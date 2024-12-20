## Utility Functions for Proto Interaction Creation

import ast, csv
from tapas.utils import text_utils
from tapas.protos import interaction_pb2

_AggregationFunction = interaction_pb2.Answer.AggregationFunction

# TSV headers.
_ID = "id"
_ANNOTATOR = "annotator"
_POSITION = "position"
_QUESTION = "question"
_TABLE_FILE = "table_file"
_ANSWER_TEXT = "answer_text"
_ANSWER_COORDINATES = "answer_coordinates"
_AGGREGATION = "aggregation"
_ANSWER_FLOAT_VALUE = "float_answer"
_ANSWER_CLASS_INDEX = "class_index"


def _parse_answer_coordinates(answer_coordinate_str, answer):
    """Populates the answer_coordinates field of `answer` by parsing `answer_coordinate_str`.

    Args:
      answer_coordinate_str: A string representation of a Python list of tuple
        strings.
        For example: "['(1, 4)','(1, 3)', ...]"
      answer: an Answer object.
    """

    try:
        coords = ast.literal_eval(answer_coordinate_str)
        for row_index, column_index in sorted(
            ast.literal_eval(coord) for coord in coords
        ):
            answer.answer_coordinates.add(
                row_index=row_index, column_index=column_index
            )

    except SyntaxError:
        raise ValueError("Unable to evaluate %s" % answer_coordinate_str)


def _parse_answer_text(answer_text, answer):
    """Populates the answer_texts field of `answer` by parsing `answer_text`.

    Args:
      answer_text: A string representation of a Python list of strings.
        For example: "[u'test', u'hello', ...]"
      answer: an Answer object.
    """
    try:
        for value in ast.literal_eval(answer_text):
            answer.answer_texts.append(value)
    except SyntaxError:
        raise ValueError("Unable to evaluate %s" % answer_text)


def read_from_tsv_file(file_handle):
    """Parses a TSV file in SQA format into a list of interactions.

    Args:
      file_handle:  File handle of a TSV file in SQA format.

    Returns:
      Questions grouped into interactions.
    """
    questions = {}
    for row in csv.DictReader(file_handle, delimiter="\t"):

        sequence_id = text_utils.get_sequence_id(row[_ID], row[_ANNOTATOR])
        key = sequence_id, row[_TABLE_FILE]
        if key not in questions:
            questions[key] = {}

        position = int(row[_POSITION])

        answer = interaction_pb2.Answer()
        _parse_answer_coordinates(row[_ANSWER_COORDINATES], answer)
        _parse_answer_text(row[_ANSWER_TEXT], answer)

        if _AGGREGATION in row:
            agg_func = row[_AGGREGATION].upper().strip()
            if agg_func:
                answer.aggregation_function = _AggregationFunction.Value(agg_func)

        if _ANSWER_FLOAT_VALUE in row:
            float_value = row[_ANSWER_FLOAT_VALUE]
            if float_value:
                answer.float_value = float(float_value)

        if _ANSWER_CLASS_INDEX in row:
            class_index = row[_ANSWER_CLASS_INDEX]
            if class_index:
                answer.class_index = int(class_index)

        questions[key][position] = interaction_pb2.Question(
            id=text_utils.get_question_id(sequence_id, position),
            original_text=row[_QUESTION],
            answer=answer,
        )

    interactions = []
    for (sequence_id, table_file), question_dict in sorted(
        questions.items(), key=lambda sid: sid[0]
    ):

        question_list = [
            question
            for _, question in sorted(question_dict.items(), key=lambda pos: pos[0])
        ]
        interactions.append(
            interaction_pb2.Interaction(
                id=sequence_id,
                questions=question_list,
                table=interaction_pb2.Table(table_id=table_file),
            )
        )

    return interactions
