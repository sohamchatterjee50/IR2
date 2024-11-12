## Data Conversion to Tensorflow examples

import argparse
from data_processing.utils import base_utils
from data_processing.utils import create_data
from data_processing.utils import io_utils
from apache_beam.runners.direct import direct_runner


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_interactions_dir",
        type=str,
        required=True,
        help="Directory with inputs.",
    )
    parser.add_argument("--input_tables_dir", type=str, help="Directory with inputs.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory with outputs."
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="The vocabulary file that the BERT model was trained on.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=True,
        help="Max length of a sequence in word pieces.",
    )
    parser.add_argument(
        "--max_column_id", type=int, required=True, help="Max column id to extract."
    )
    parser.add_argument(
        "--max_row_id", type=int, required=True, help="Max row id to extract."
    )
    parser.add_argument(
        "--cell_trim_length",
        type=int,
        default=-1,
        help="If > 0: Trim cells so that the length is <= this value.",
    )
    parser.add_argument(
        "--use_document_title",
        action="store_true",
        help="Include document title text in the tf example.",
    )
    parser.add_argument(
        "--converter_impl",
        type=str,
        default="PYTHON",
        choices=["PYTHON", "JAVA", "CPP"],
        help="Implementation to map interactions to tf examples.",
    )
    args = parser.parse_args()

    return args


def run(inputs, outputs, input_format, args):

    # Only require this runner as it will all be done locally
    direct_runner.DirectRunner().run(
        create_data.build_retrieval_pipeline(
            input_files=inputs,
            input_format=input_format,
            output_files=outputs,
            config=base_utils.RetrievalConversionConfig(
                vocab_file=args.vocab_file,
                max_seq_length=args.max_seq_length,
                max_column_id=args.max_column_id,
                max_row_id=args.max_row_id,
                strip_column_names=False,
                cell_trim_length=args.cell_trim_length,
                use_document_title=args.use_document_title,
            ),
            converter_impl=args.converter_impl,
        )
    ).wait_until_finish()


def main(args):
    inputs, outputs = io_utils.get_inputs_and_outputs(
        args.input_interactions_dir, args.output_dir
    )
    if not inputs:
        raise ValueError(f"Input dir is empty: '{args.input_interactions_dir}'")

    run(inputs, outputs, create_data.InputFormat.INTERACTION, args)

    if args.input_tables_dir is not None:
        table_inputs, table_outputs = io_utils.get_inputs_and_outputs(
            args.input_tables_dir, args.output_dir
        )
        run(table_inputs, table_outputs, create_data.InputFormat.TABLE)


if __name__ == "__main__":

    args = config()
    main(args)
