## Contains Task Definitions

import enum


class Task(enum.Enum):
    """Fine-tuning tasks supported by Tapas."""

    SQA = 0
    WIKISQL = 1
    WIKISQL_SUPERVISED = 2
    NQ_RETRIEVAL = 3
