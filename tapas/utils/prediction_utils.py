## Utilitity Functions for Prediction Handling

import os, ast, csv
import numpy as np
import pandas as pd
import tensorflow._api.v2.compat.v1 as tf
from tapas.protos import interaction_pb2


def parse_coordinates(raw_coordinates):
    """Parses cell coordinates from text."""
    return {ast.literal_eval(x) for x in ast.literal_eval(raw_coordinates)}


def iterate_predictions(prediction_file):
    """Iterates through a TSV prediction file."""
    with tf.io.gfile.GFile(prediction_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            if "logits_cls" in row:
                # Only for binary problems the logit will be a float scalar.
                if row["logits_cls"].startswith("["):
                    row["logits_cls"] = np.fromstring(
                        row["logits_cls"][1:-1], sep=" "
                    ).tolist()

                else:
                    row["logits_cls"] = float(row["logits_cls"])

            yield row


def is_tfrecord(filename):
    extension = os.path.splitext(filename)[1]
    return extension in [".tfrecord", ".tfrecords"]


def iterate_interactions(interactions_file):
    """Get interactions from file."""
    for value in tf.python_io.tf_record_iterator(interactions_file):
        interaction = interaction_pb2.Interaction()
        interaction.ParseFromString(value)
        yield interaction


def parse_interaction_id(text):
    return text[: text.rindex("-")]


def table_to_panda_frame(table):

    contents = [[cell.text for cell in row.cells] for row in table.rows]
    headers = [f"{column.text}_{index}" for index, column in enumerate(table.columns)]

    return pd.DataFrame(contents, columns=headers)
