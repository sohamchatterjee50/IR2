## Utilities to use for replacing those in `nq_preprocess_utils`
## For SQA data handling.

import os, json, html, urllib
import apache_beam as beam
from tapas.utils.constants import _NS
from tapas.protos import interaction_pb2


def get_filenames(path, split):
    """Reads JSONL files from the given path."""
    filepath = os.path.join(path, f"{split.name}.jsonl")
    yield filepath


def process_line(line_split):
    """Parses json and yields result dictionary."""
    beam.metrics.Metrics.counter(_NS, "Lines").inc()
    line, split = line_split
    # The input is already in the target format, so no need for complex parsing
    data = json.loads(line)
    result = {
        "example_id": data["id"],
        "contained": True,  # Since we know each line contains a table
        "tables": [data["table"]],
        "interactions": [data],
    }
    result["split"] = split
    return result


def to_table(result):
    """Convert dictionary tables to Table protos."""
    for table_dict in result["tables"]:
        table = interaction_pb2.Table()
        table.table_id = table_dict["tableId"]
        table.document_title = table_dict["documentTitle"]
        table.document_url = (
            table_dict["documentUrl"] if table_dict["documentUrl"] else ""
        )

        # Convert columns
        for col in table_dict["columns"]:

            column = table.columns.add()
            column.text = col["text"]

        # Convert rows
        for row_dict in table_dict["rows"]:
            row = table.rows.add()
            for cell in row_dict["cells"]:
                new_cell = row.cells.add()
                new_cell.text = cell["text"]

        # Add alternative URLs if they exist
        if table_dict.get("alternativeDocumentUrls"):
            table.alternative_document_urls.extend(
                table_dict["alternativeDocumentUrls"]
            )

        beam.metrics.Metrics.counter(_NS, "Tables").inc()
        yield table


def to_interaction(result):
    """Convert dictionary interactions to Interaction protos."""
    for interaction_dict in result["interactions"]:

        interaction = interaction_pb2.Interaction()
        interaction.id = f'{result["split"]}_{interaction_dict["id"]}'

        # Convert table
        interaction.table.CopyFrom(
            next(to_table({"tables": [interaction_dict["table"]]}))
        )

        # Convert questions
        for q in interaction_dict["questions"]:

            question = interaction.questions.add()
            question.id = f"{interaction.id}_{q['id'].split('_')[-1]}"
            question.original_text = q["originalText"]

            # Handle answer
            if "answer" in q:
                answer = q["answer"]
                if "answerTexts" in answer:
                    # Convert string representation of list to actual list
                    answer_texts = (
                        eval(answer["answerTexts"])
                        if isinstance(answer["answerTexts"], str)
                        else answer["answerTexts"]
                    )
                    question.answer.answer_texts.extend(answer_texts)

        beam.metrics.Metrics.counter(_NS, "Interactions").inc()
        yield interaction


def get_version(table):
    """Get version number from URL or fallback to 0 if no URL."""
    if not table.document_url:
        return 0

    query = urllib.parse.urlparse(html.unescape(table.document_url)).query
    parsed_query = urllib.parse.parse_qs(query)
    try:
        value = parsed_query["oldid"][0]
        return int(value)
    
    except (KeyError, IndexError):
        return 0
