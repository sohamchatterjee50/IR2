## Utilities for Data Conversions for Different Tasks / Datasets

import os, collections
from tapas.utils import interaction_utils_parser
from tapas.task_utils import tasks, sqa_utils, wikisql_utils

_Mode = interaction_utils_parser.SupervisionMode


def get_interaction_dir(output_dir):
    return os.path.join(output_dir, "interactions")


def get_train_filename(task):
    """Gets the name of the file with training data."""
    if task in [
        tasks.Task.WIKISQL,
        tasks.Task.WIKISQL_SUPERVISED,
        tasks.Task.NQ_RETRIEVAL,
    ]:
        return "train"

    if task == tasks.Task.SQA:
        return "random-split-1-train"

    raise ValueError(f"Unknown task: {task.name}")


def get_dev_filename(task):
    """Gets the name of the file with development data."""
    if task in [
        tasks.Task.WIKISQL,
        tasks.Task.WIKISQL_SUPERVISED,
        tasks.Task.NQ_RETRIEVAL,
    ]:
        return "dev"

    if task == tasks.Task.SQA:
        return "random-split-1-dev"

    raise ValueError(f"Unknown task: {task.name}")


def get_test_filename(task):
    """Gets the name of the file with test data."""
    if task in [
        tasks.Task.SQA,
        tasks.Task.WIKISQL,
        tasks.Task.WIKISQL_SUPERVISED,
        tasks.Task.NQ_RETRIEVAL,
    ]:
        return "test"

    raise ValueError(f"Unknown task: {task.name}")


def get_supervision_modes(task):
    """Gets the correct supervision mode for each task."""
    if task == tasks.Task.WIKISQL:
        # We tried using REMOVE_ALL_STRICT but didn't find it to improve results.
        return {
            "train.tsv": _Mode.REMOVE_ALL,
            "dev.tsv": _Mode.NONE,
            "test.tsv": _Mode.NONE,
        }

    if task == tasks.Task.WTQ:
        # Remove ambiguous examples at training time.
        # (Examples where the answer occurs in multiple cells.)
        modes = {}
        modes["train.tsv"] = _Mode.REMOVE_ALL_STRICT
        modes["test.tsv"] = _Mode.REMOVE_ALL
        for i in range(1, 5 + 1):

            modes[f"random-split-{i}-train.tsv"] = _Mode.REMOVE_ALL_STRICT
            modes[f"random-split-{i}-dev.tsv"] = _Mode.REMOVE_ALL

        return modes

    if task in [
        tasks.Task.SQA,
        tasks.Task.WIKISQL_SUPERVISED,
    ]:
        return collections.defaultdict(lambda: _Mode.NONE)

    raise ValueError(f"Unknown task: {task.name}")


def create_interactions(
    task,
    input_dir,
    output_dir,
    token_selector,
):  # pylint: disable=g-doc-args
    """Converts original task data to interactions.

    Interactions will be written to f'{output_dir}/interactions'. Other files
    might be written as well.

    Args:
      task: The current task.
      input_dir: Data with original task data.
      output_dir: Outputs are written to this directory.
      token_selector: Optional helper class to keep more relevant tokens in input.
    """

    if task == tasks.Task.SQA:
        tsv_dir = input_dir

    elif task == tasks.Task.WIKISQL:
        wikisql_utils.convert(input_dir, output_dir)
        tsv_dir = output_dir

    elif task == tasks.Task.WIKISQL_SUPERVISED:
        wikisql_utils.convert(input_dir, output_dir)
        tsv_dir = output_dir

    else:
        raise ValueError(f"Unknown task: {task.name}")

    sqa_utils.create_interactions(
        get_supervision_modes(task),
        tsv_dir,
        get_interaction_dir(output_dir),
        token_selector,
    )
