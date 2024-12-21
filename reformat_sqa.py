import os, csv, json, hydra
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from omegaconf import DictConfig


def to_jsonl(data_list: list, output_file: str):
    """Write data to JSONL format."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out_file:

        for item in tqdm(data_list, desc="Writing JSONL Data", total=len(data_list)):

            out_file.write(json.dumps(item) + "\n")


def create_out_file(input_path: str, output_base: str) -> str:
    """Create output filepath maintaining directory structure."""
    # Get relative path by removing common prefix
    rel_path = os.path.relpath(input_path, start=os.path.dirname(input_path))

    # Change extension to .jsonl
    base_name = os.path.splitext(rel_path)[0]
    new_path = os.path.join(output_base, f"{base_name}.jsonl")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    return new_path


def load_table(file_path: str) -> dict:
    """Load CSV table into structured format."""
    with open(file_path, "r", encoding="utf-8") as csv_file:

        reader = csv.reader(csv_file)
        rows = list(reader)

    columns = [{"text": col} for col in rows[0]]
    table_rows = []
    for row in rows[1:]:

        cells = [{"text": cell} for cell in row]
        table_rows.append({"cells": cells})

    return {
        "columns": columns,
        "rows": table_rows,
    }


def load_interactions(file_path: str) -> list:
    """Load interactions from TSV file."""
    try:
        df = pd.read_csv(file_path, sep="\t")
        return df.to_dict(orient="records")

    except FileNotFoundError:
        print(f"Warning: Input file not found: {file_path}")
        return []


def convert_sqa_to_nq(data_list: list, name: str, input_dir: str) -> list:
    """Convert SQA data to NQ-like format."""
    nq_data = []
    for entry in tqdm(data_list, desc="Processing for {name}", total=len(data_list)):

        table_file_path = os.path.join(
            input_dir, "table_csv", os.path.basename(entry["table_file"])
        )

        try:
            table_data = load_table(table_file_path)

        except FileNotFoundError:
            print(f"Warning: Table file not found: {table_file_path}")
            continue

        formatted_entry = {
            "id": entry["id"],
            "table": {
                "columns": table_data["columns"],
                "rows": table_data["rows"],
                "tableId": entry["id"],
                "documentTitle": entry["table_file"],
                "documentUrl": None,
                "alternativeDocumentUrls": [],
            },
            "questions": [
                {
                    "id": f"{entry['id']}_{entry['annotator']}_{entry['position']}",
                    "originalText": entry["question"],
                    "answer": {"answerTexts": entry["answer_text"]},
                }
            ],
        }
        nq_data.append(formatted_entry)

    return nq_data


@hydra.main(version_base=None, config_path="configs", config_name="reformat_sqa")
def main(cfg: DictConfig):

    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)

    # Ensure input and output directories exist
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for dirpath, _, filenames in tqdm(
        os.walk(args.input_dir), desc="Processing files..."
    ):

        for name in filenames:

            if not name.endswith(".tsv"):
                print(f"Skipping non-TSV file: {name}")
                continue

            input_path = os.path.join(dirpath, name)
            output_path = create_out_file(input_path, args.output_dir)

            # Process the file
            sqa_data = load_interactions(input_path)
            if sqa_data:
                nq_data = convert_sqa_to_nq(sqa_data, name, args.input_dir)

                if nq_data:
                    to_jsonl(nq_data, output_path)
                    print(f"Processed {input_path} -> {output_path}")

                else:
                    print(f"No valid data generated for {input_path}")

            else:
                print(f"Failed to load data from {input_path}")


if __name__ == "__main__":

    main()
