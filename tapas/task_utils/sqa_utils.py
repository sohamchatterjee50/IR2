## Reads Interactions from SQA TSV files, Adds the Tables,
## and Writes them to tfrecords

import os, csv, collections
import tensorflow._api.v2.compat.v1 as tf
from absl import logging
from tapas.protos import interaction_pb2
from tapas.utils import (
    file_utils,
    interaction_utils,
    interaction_utils_parser,
)

_Mode = interaction_utils_parser.SupervisionMode


def _read_interactions(input_dir):
    """Reads interactions from TSV files."""
    filenames = [
        fn for fn in file_utils.list_directory(input_dir) if fn.endswith(".tsv")
    ]
    interaction_dict = {}
    for filename in filenames:

        filepath = os.path.join(input_dir, filename)
        with tf.io.gfile.GFile(filepath, "r") as file_handle:

            try:
                interactions = interaction_utils.read_from_tsv_file(file_handle)
                interaction_dict[filename] = interactions

            except KeyError as ke:
                logging.error(
                    "Can't read interactions from file: %s (%s)", filepath, ke
                )

    return interaction_dict


def _add_tables(input_dir, interaction_dict):
    """Adds table protos to all interactions."""
    table_files = set()
    for interactions in interaction_dict.values():

        for interaction in interactions:

            table_files.add(interaction.table.table_id)

    table_dict = {}
    for index, table_file in enumerate(sorted(table_files)):

        logging.log_every_n(
            logging.INFO, "Read %4d / %4d table files", 100, index, len(table_files)
        )
        table_path = os.path.join(input_dir, table_file)
        with tf.io.gfile.GFile(table_path, "r") as table_handle:
            table = interaction_pb2.Table()
            rows = list(csv.reader(table_handle))
            headers, rows = rows[0], rows[1:]

            for header in headers:

                table.columns.add().text = header

            for row in rows:

                new_row = table.rows.add()
                for cell in row:

                    new_row.cells.add().text = cell

            table.table_id = table_file
            table_dict[table_file] = table

    for interactions in interaction_dict.values():

        for interaction in interactions:

            interaction.table.CopyFrom(table_dict[interaction.table.table_id])


def _write_report(
    report_filename,
    supervision_modes,
    counters,
):
    """Creates the report_filename file containing statistics about conversion."""
    contents = ["Total\tValid\tFailed\tFile"]
    for dataset, counter in counters.items():

        failed = counter.pop("failed", 0)
        valid = counter.pop("valid", 0)
        total = failed + valid
        contents.append("\t".join(map(str, (total, valid, failed, dataset))))

    # Prints an overview to the stdout.
    logging.info("\n".join(contents))

    with tf.io.gfile.GFile(report_filename, "w") as report_file:
        for dataset, supervision_mode in sorted(supervision_modes.items()):
            report_file.write(
                "# Dataset: {} supervision_mode: {}\n".format(dataset, supervision_mode)
            )
        report_file.write("\n")
        report_file.write("\n".join(contents))
        report_file.write("\n")

        for dataset, counter in counters.items():

            if not counter:
                continue

            report_file.write("# Detailed error statistics for {}:\n".format(dataset))

            for key, value in sorted(counter.items()):

                report_file.write("# {}\t{}\n".format(key, value))


def _parse_questions(interaction_dict, supervision_modes, report_filename):
    """Adds numeric value spans to all questions."""
    counters = collections.defaultdict(collections.Counter)
    for key, interactions in interaction_dict.items():

        for interaction in interactions:

            questions = []

            for original_question in interaction.questions:
                try:
                    question = interaction_utils_parser.parse_question(
                        interaction.table, original_question, supervision_modes[key]
                    )
                    counters[key]["valid"] += 1

                except ValueError as exc:
                    question = interaction_pb2.Question()
                    question.CopyFrom(original_question)
                    question.answer.is_valid = False
                    counters[key]["failed"] += 1
                    counters[key]["failed-" + str(exc)] += 1

                questions.append(question)

            del interaction.questions[:]
            interaction.questions.extend(questions)

    _write_report(report_filename, supervision_modes, counters)


def _write_tfrecord(
    interactions,
    filepath,
    token_selector,
):
    with tf.io.TFRecordWriter(filepath + ".tfrecord") as writer:

        for interaction in interactions:

            if token_selector is not None:
                interaction = token_selector.annotated_interaction(interaction)

            writer.write(interaction.SerializeToString())


def _get_output_filename(output_dir, input_file):

    basename = os.path.splitext(input_file)[0]
    return os.path.join(output_dir, basename)


def create_interactions(
    input_dir,
    output_dir,
    token_selector,
):
    """Converts data in SQA format to Interaction protos.

    Args:
      supervision_modes: Import for WikiSQL, decide if supervision is removed.
      input_dir: SQA data.
      output_dir: Where interactions will be written.
      token_selector: Optional helper class to keep more relevant tokens in input.
    """
    file_utils.make_directories(output_dir)
    supervision_modes = collections.defaultdict(lambda: _Mode.NONE)

    interaction_dict = _read_interactions(input_dir)
    _add_tables(input_dir, interaction_dict)
    _parse_questions(
        interaction_dict, supervision_modes, os.path.join(output_dir, "report.tsv")
    )
    for filename, interactions in interaction_dict.items():

        _write_tfrecord(
            interactions, _get_output_filename(output_dir, filename), token_selector
        )
