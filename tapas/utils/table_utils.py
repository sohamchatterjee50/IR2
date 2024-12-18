## Contains Parser Functions for Table Inputs

import enum
import tensorflow._api.v2.compat.v1 as tf
from tapas.utils import text_utils, dataset_utils


class TableTask(enum.Enum):
    CLASSIFICATION = 0
    PRETRAINING = 1
    RETRIEVAL = 2
    RETRIEVAL_NEGATIVES = 3


def parse_table_examples(
    max_seq_length,
    max_predictions_per_seq,
    task_type,
    add_aggregation_function_id,
    add_classification_labels,
    add_answer,
    include_id,
    add_candidate_answers,
    max_num_candidates,
    params,
):
    """Returns a parse_fn that parses tf.Example in table format.

    Args:
      max_seq_length: int: The length of the model's input maximum sequence.
      max_predictions_per_seq: Shoud be set when using TableTask.PRETRAINING.
      task_type: TableTask
      add_aggregation_function_id: bool: True if the model learns a loss to
        predict an aggregation function.
      add_classification_labels: bool: True if the model does classification.
      add_answer: bool: True to add features for the weakly supervised setting.
      include_id: bool: False if TPU is used (here it should be True).
      add_candidate_answers: bool,
      max_num_candidates: int: Should be set when add_candidate_answers is true.
      params: The parameters to pass to the dataset parser function
    """

    if task_type is TableTask.RETRIEVAL_NEGATIVES:
        # For this task every table feature encodes 2 tables.
        max_seq_length = 2 * max_seq_length

    feature_types = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "column_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "row_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "prev_label_ids": tf.FixedLenFeature(
            [max_seq_length], tf.int64, default_value=[0] * max_seq_length
        ),
        "column_ranks": tf.FixedLenFeature(
            [max_seq_length],
            tf.int64,
            default_value=[0] * max_seq_length,
        ),
        "inv_column_ranks": tf.FixedLenFeature(
            [max_seq_length],
            tf.int64,
            default_value=[0] * max_seq_length,
        ),
        "numeric_relations": tf.FixedLenFeature(
            [max_seq_length], tf.int64, default_value=[0] * max_seq_length
        ),
    }

    if task_type is TableTask.PRETRAINING:
        if max_predictions_per_seq is None:
            raise ValueError(
                "Please set max_predictions_per_seq when using TableTask.PRETRAINING."
            )
        feature_types.update(
            {
                "masked_lm_positions": tf.FixedLenFeature(
                    [max_predictions_per_seq], tf.int64
                ),
                "masked_lm_ids": tf.FixedLenFeature(
                    [max_predictions_per_seq], tf.int64
                ),
                "masked_lm_weights": tf.FixedLenFeature(
                    [max_predictions_per_seq], tf.float32
                ),
                "next_sentence_labels": tf.FixedLenFeature([1], tf.int64),
            }
        )

    elif task_type in (TableTask.CLASSIFICATION,):
        # For classification we have a label for each token.
        if task_type is TableTask.CLASSIFICATION:
            feature_types.update(
                {
                    "label_ids": tf.FixedLenFeature(
                        [max_seq_length],
                        tf.int64,
                        default_value=[0] * max_seq_length,
                    ),
                }
            )

        feature_types.update(
            {
                "question_id_ints": tf.FixedLenFeature(
                    [text_utils.DEFAULT_INTS_LENGTH],
                    tf.int64,
                    default_value=[0] * text_utils.DEFAULT_INTS_LENGTH,
                ),
            }
        )
        # Label for predicting the aggregation function.
        if add_aggregation_function_id:
            feature_types.update(
                {
                    "aggregation_function_id": tf.FixedLenFeature([1], tf.int64),
                }
            )

        if add_classification_labels:
            feature_types.update(
                {
                    "classification_class_index": tf.FixedLenFeature([1], tf.int64),
                }
            )

        # Features for the weakly supervised setting.
        if add_answer:
            feature_types.update(
                {
                    "numeric_values": tf.FixedLenFeature(
                        [max_seq_length],
                        tf.float32,
                        default_value=[0] * max_seq_length,
                    ),
                    "numeric_values_scale": tf.FixedLenFeature(
                        [max_seq_length],
                        tf.float32,
                        default_value=[0] * max_seq_length,
                    ),
                    "answer": tf.FixedLenFeature(
                        [1],
                        tf.float32,
                        default_value=[0],
                    ),
                }
            )

    elif task_type in [TableTask.RETRIEVAL, TableTask.RETRIEVAL_NEGATIVES]:
        tables_per_examples = 1

        if task_type is TableTask.RETRIEVAL_NEGATIVES:
            tables_per_examples = 2

        max_seq_length = max_seq_length // tables_per_examples
        feature_types.update(
            {
                "question_input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
                "question_input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
                "table_id_hash": tf.FixedLenFeature(
                    [tables_per_examples],
                    tf.int64,
                    default_value=[0] * tables_per_examples,
                ),
                "question_hash": tf.FixedLenFeature(
                    [1],
                    tf.int64,
                    default_value=[0],
                ),
            }
        )

        if include_id:
            feature_types.update(
                {
                    "table_id": tf.FixedLenFeature([tables_per_examples], tf.string),
                    "question_id": tf.FixedLenFeature([1], tf.string),
                }
            )

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if add_candidate_answers:
        feature_types.update(
            {
                "cand_num": tf.FixedLenFeature([], tf.int64),
                "can_aggregation_function_ids": tf.FixedLenFeature(
                    [max_num_candidates], tf.int64
                ),
                "can_sizes": tf.FixedLenFeature([max_num_candidates], tf.int64),
                "can_indexes": tf.VarLenFeature(tf.int64),
            }
        )

    if include_id:
        feature_types.update(
            {
                "question_id": tf.FixedLenFeature([1], tf.string),
            }
        )

    def _parse_fn(serialized_example):
        features = dict(
            dataset_utils.build_parser_function(feature_types, params)(
                serialized_example
            )
        )
        if add_candidate_answers:
            _preprocess_candidate_answers(
                features,
                max_num_candidates=max_num_candidates,
                max_seq_length=max_seq_length,
            )
        return features

    return _parse_fn


def _preprocess_candidate_answers(features, max_num_candidates, max_seq_length):
    """Prepares dense labels for each candidate."""
    ragged_indices = tf.RaggedTensor.from_row_lengths(
        features["can_indexes"].values, features["can_sizes"]
    )
    candidate_id = tf.ragged.row_splits_to_segment_ids(ragged_indices.row_splits)
    indices = tf.stack([candidate_id, ragged_indices.flat_values], axis=-1)
    updates = tf.ones_like(candidate_id, dtype=tf.int32)
    features["can_label_ids"] = tf.scatter_nd(
        indices=indices, updates=updates, shape=[max_num_candidates, max_seq_length]
    )
    # Variable length tensors are not supported on TPU.
    del features["can_indexes"]
