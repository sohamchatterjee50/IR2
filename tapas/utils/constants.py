## General Constants File
import enum
from typing import TypeVar

# General Constants
_NS = "main"
_SEP = "[SEP]"
_MAX_INT = 2**32 - 1

# For Text Index
T = TypeVar("T")

# For Number Utils
# Some of the definitions here follow guidelines written in DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L414.
_MAX_DATE_NGRAM_SIZE = 5

_NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]

_ORDINAL_WORDS = [
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fith",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
]

_ORDINAL_SUFFIXES = ["st", "nd", "rd", "th"]
_MIN_YEAR = 1700
_MAX_YEAR = 2016
_INF = float("INF")

# For Number Annotation Utils
_DATE_TUPLE_SIZE = 3

# For Interpret Utils
_MAX_NUM_TRIALS = 100
_FLOAT_TOLERANCE = 1.0e-2
_MAX_NUM_CANDIDATES = 500
_MAX_INDICES_TO_EXPLORE = 10

# For Attention Utils
_INFINITY = 10000  # Same value used in BERT

# For Retriever Model
# Used to mask the logits of the repeated elements
_INF_RET = 10000.0

# For Eval Retriever Utils
_NUM_NEIGHBORS = 100

# Constants for Dopa Tables Projects
EMPTY_TEXT = "EMPTY"
NUM_TYPE = "number"
DATE_TYPE = "date"


class Relation(enum.Enum):
    HEADER_TO_CELL = 1  # Header -> cell
    CELL_TO_HEADER = 2  # Cell -> header
    QUERY_TO_HEADER = 3  # Query -> headers
    QUERY_TO_CELL = 4  # Query -> cells
    ROW_TO_CELL = 5  # Row -> cells
    CELL_TO_ROW = 6  # Cells -> row
    EQ = 7  # Annotation value == cell value
    LT = 8  # Annotation value < cell value
    GT = 9  # Annotation value > cell value
