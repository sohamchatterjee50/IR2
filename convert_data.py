## Data Conversion to Tensorflow examples

import hydra
from argparse import Namespace
from omegaconf import DictConfig
from apache_beam.runners.direct import direct_runner
from tapas.utils import io_utils, base_utils, create_utils


# Extra utility to convert from string to ConverterImplType
def str_to_type(arg: str):

    if arg.upper() == "PYTHON":
        return create_utils.ConverterImplType.PYTHON

    raise ValueError(f"Value '{arg}' not supported!")


def run(inputs, outputs, input_format, args):

    # Only require this runner from the original repo as it will all be done locally
    direct_runner.DirectRunner().run(
        create_utils.build_retrieval_pipeline(
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
            converter_impl=str_to_type(args.converter_impl),
        )
    ).wait_until_finish()


@hydra.main(version_base=None, config_path="configs", config_name="convert_data")
def main(cfg: DictConfig):

    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)

    inputs, outputs = io_utils.get_inputs_outputs(
        args.input_interactions_dir, args.output_dir
    )
    if not inputs:
        raise ValueError(f"Input directory is empty: '{args.input_interactions_dir}'")

    run(inputs, outputs, create_utils.InputFormat.INTERACTION, args)

    # In case tables exist also
    if args.input_tables_dir is not None:
        table_inputs, table_outputs = io_utils.get_inputs_outputs(
            args.input_tables_dir, args.output_dir
        )
        run(table_inputs, table_outputs, create_utils.InputFormat.TABLE, args)


if __name__ == "__main__":

    main()
