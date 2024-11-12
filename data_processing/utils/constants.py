## Constants used by Dopa Tables Project
import enum

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
