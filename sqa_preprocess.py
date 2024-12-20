## Script for Creating SQA Data

import os, hydra, random
import tensorflow._api.v2.compat.v1 as tf
from absl import logging
from argparse import Namespace
from omegaconf import DictConfig
from tapas.utils import (
    base_utils,
    pruning_utils,
    prediction_utils,
    tf_example_utils,
    number_annot_utils,
)
from tapas.task_utils import tasks, sqa_utils
from tapas.utils.constants import _MAX_TABLE_ID
from tapas.utils.file_utils import make_directories, _print, _to_tf_compression_type


tf.disable_v2_behavior()


def _create_all_examples(vocab_file, test_mode, output_dir, test_batch_size, args):
    """Converts interactions to TF examples."""
    interaction_dir = output_dir
    example_dir = os.path.join(output_dir, "tf_examples")
    make_directories(example_dir)

    _create_examples(
        interaction_dir,
        example_dir,
        vocab_file,
        "random-split-1-train",
        batch_size=None,
        test_mode=test_mode,
        args=args,
    )
    _create_examples(
        interaction_dir,
        example_dir,
        vocab_file,
        "random-split-1-dev",
        test_batch_size,
        test_mode,
        args,
    )
    _create_examples(
        interaction_dir,
        example_dir,
        vocab_file,
        "test",
        test_batch_size,
        test_mode,
        args,
    )


def _create_examples(
    interaction_dir,
    example_dir,
    vocab_file,
    filename,
    batch_size,
    test_mode,
    args,
):
    """Creates TF example for a single dataset."""

    filename = f"{filename}.tfrecord"
    interaction_path = os.path.join(interaction_dir, filename)
    example_path = os.path.join(example_dir, filename)

    # config = tf_example_utils.ClassifierConversionConfig(
    #     vocab_file=vocab_file,
    #     max_seq_length=args.max_seq_length,
    #     use_document_title=args.use_document_title,
    #     update_answer_coordinates=args.update_answer_coordinates,
    #     drop_rows_to_fit=args.drop_rows_to_fit,
    #     max_column_id=_MAX_TABLE_ID,
    #     max_row_id=_MAX_TABLE_ID,
    #     strip_column_names=False,
    #     add_aggregation_candidates=False,
    #     expand_entity_descriptions=False,
    # )

    config = tf_example_utils.RetrievalConversionConfig(
        vocab_file=vocab_file,
        max_seq_length=args.max_seq_length,
        use_document_title=args.use_document_title,
        max_column_id=_MAX_TABLE_ID,
        max_row_id=_MAX_TABLE_ID,
        strip_column_names=False,
    )

    converter = base_utils.ToRetrievalTensorflowExample(config)

    examples = []
    num_questions, num_conversion_errors = 0, 0
    for interaction in prediction_utils.iterate_interactions(interaction_path):

        number_annot_utils.add_numeric_values(interaction)
        for i in range(len(interaction.questions)):

            num_questions += 1
            try:
                examples.append(converter.convert(interaction, i))

            except ValueError as e:
                num_conversion_errors += 1
                logging.info(
                    "Can't convert interaction: %s error: %s", interaction.id, e
                )

        if test_mode and len(examples) >= 100:
            break

    _print(f"Processed: {filename}")
    _print(f"Number of questions processed: {num_questions}")
    _print(f"Number of examples: {len(examples)}")
    _print(f"Number of conversion errors: {num_conversion_errors}")

    if batch_size is None:
        random.shuffle(examples)

    else:
        # Make sure the eval sets are divisible by the test batch size since
        # otherwise examples will be dropped on TPU.
        # These examples will later be ignored when writing the predictions.
        original_num_examples = len(examples)
        while len(examples) % batch_size != 0:
            examples.append(converter.get_empty_example())

        if original_num_examples != len(examples):
            _print(f"Padded with {len(examples) - original_num_examples} examples.")

    with tf.io.TFRecordWriter(
        example_path,
        options=_to_tf_compression_type(args.compression_type),
    ) as writer:

        for example in examples:

            writer.write(example.SerializeToString())


def _get_token_selector(args):
    if not args.prune_columns:
        return None

    return pruning_utils.HeuristicExactMatchTokenSelector(
        args.bert_vocab_file,
        args.max_seq_length,
        pruning_utils.SelectionType.COLUMN,
        # Only relevant for SQA where questions come in sequence
        use_previous_answer=True,
        use_previous_questions=True,
    )


@hydra.main(version_base=None, config_path="configs", config_name="sqa_preprocess")
def main(cfg: DictConfig):

    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)

    if args.tapas_verbosity:
        tf.get_logger().setLevel(args.tapas_verbosity)

    task = tasks.Task.SQA
    output_dir = os.path.join(args.output_dir, task.name.lower())

    # Creating the data
    _print("Creating interactions ...")
    token_selector = _get_token_selector(args)
    sqa_utils.create_interactions(args.input_dir, output_dir, token_selector)
    _print("Creating TF examples ...")
    _create_all_examples(
        args.bert_vocab_file,
        args.test_mode,
        test_batch_size=args.test_batch_size,
        output_dir=output_dir,
        args=args,
    )


if __name__ == "__main__":

    main()
