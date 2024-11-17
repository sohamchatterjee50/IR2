## Conversion Code for Table Retrieval

import tensorflow._api.v2.compat.v1 as tf
from data_processing.utils.constants import _SEP, _MAX_INT
from data_processing.utils import text_utils, tf_example_utils


RetrievalConversionConfig = tf_example_utils.RetrievalConversionConfig


def _join_features(features, negative_example_features):
    """Joins the features of two tables."""

    def append_feature(values, other_values):
        values.extend(other_values)

    for k, n_v in negative_example_features.items():

        v = features[k]
        for feature_type in ["bytes_list", "float_list", "int64_list"]:
            if n_v.HasField(feature_type) != v.HasField(feature_type):
                raise ValueError(f"feature types are incomapatible: {k}")

            if n_v.HasField(feature_type):
                append_feature(
                    getattr(v, feature_type).value, getattr(n_v, feature_type).value
                )
                break

        else:
            raise ValueError(f"Unsupported feature type: {k}")


class ToRetrievalTensorflowExample(tf_example_utils.ToTrimmedTensorflowExample):

    def __init__(self, config):
        super(ToRetrievalTensorflowExample, self).__init__(config)
        self._use_document_title = config.use_document_title

    def convert(
        self,
        interaction,
        index,
        negative_example,
    ):
        """Converts question at 'index' to example."""
        table = interaction.table

        num_rows = len(table.rows)
        if num_rows >= self._max_row_id:
            num_rows = self._max_row_id - 1

        num_columns = len(table.columns)
        if num_columns >= self._max_column_id:
            num_columns = self._max_column_id - 1

        title = table.document_title
        if not self._use_document_title:
            title = ""

        title_tokens = self._tokenizer.tokenize(title)
        tokenized_table = self._tokenize_table(table)

        while True:
            try:
                _, features = self._to_trimmed_features(
                    question=None,
                    table=table,
                    question_tokens=title_tokens,
                    tokenized_table=tokenized_table,
                    num_columns=num_columns,
                    num_rows=num_rows,
                )
                break

            except ValueError:
                pass

            # Since this is retrieval we might get away with removing some cells of
            # the table.
            if num_columns >= num_rows:
                num_columns -= 1

            else:
                num_rows -= 1

            if num_columns == 0 or num_rows == 0:
                raise ValueError("Cannot fit table into sequence.")

        question = interaction.questions[index]
        features["question_id"] = tf_example_utils.create_string_feature(
            [question.id.encode("utf8")]
        )
        features["question_id_ints"] = tf_example_utils.create_int_feature(
            text_utils.str_to_ints(question.id, length=text_utils.DEFAULT_INTS_LENGTH)
        )

        q_tokens = self._tokenizer.tokenize(question.text)
        q_tokens = self._serialize_text(q_tokens)[0]
        q_tokens.append(tf_example_utils.Token(_SEP, _SEP))
        q_input_ids = self._to_token_ids(q_tokens)
        self._pad_to_seq_length(q_input_ids)
        q_input_mask = [1] * len(q_tokens)
        self._pad_to_seq_length(q_input_mask)
        features["question_input_ids"] = tf_example_utils.create_int_feature(
            q_input_ids
        )
        features["question_input_mask"] = tf_example_utils.create_int_feature(
            q_input_mask
        )

        if question:
            features["question_hash"] = tf_example_utils.create_int_feature(
                [tf_example_utils.fingerprint(question.text) % _MAX_INT]
            )

        if negative_example is not None:
            n_table = negative_example.table
            n_title_tokens = self._tokenizer.tokenize(n_table.document_title)
            n_tokenized_table = self._tokenize_table(n_table)
            n_num_rows = self._get_num_rows(n_table, drop_rows_to_fit=True)
            n_num_columns = self._get_num_columns(n_table)
            _, n_example_features = self._to_trimmed_features(
                question=None,
                table=n_table,
                question_tokens=n_title_tokens,
                tokenized_table=n_tokenized_table,
                num_columns=n_num_columns,
                num_rows=n_num_rows,
                drop_rows_to_fit=True,
            )
            _join_features(features, n_example_features)

        return tf.train.Example(features=tf.train.Features(feature=features))
