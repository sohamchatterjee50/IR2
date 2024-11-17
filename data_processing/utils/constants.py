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
_DATE_TUPLE_SIZE = 3

# For Interpret Utils
_MAX_NUM_TRIALS = 100
_FLOAT_TOLERANCE = 1.0e-2
_MAX_NUM_CANDIDATES = 500
_MAX_INDICES_TO_EXPLORE = 10

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
