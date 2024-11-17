# Helper functions for Mapping Input to Output Filenames

import os
import tensorflow._api.v2.compat.v1 as tf


def _is_supported(filename):

    extension = os.path.splitext(filename)[-1]
    return extension in [
        ".txtpb.gz",
        ".txtpb",
        ".tfrecord",
        ".tfrecords",
    ]


def _check_basename(
    basenames,
    basename,
    input_dir,
):
    if basename in basenames:
        raise ValueError(
            "Basename should be unique:" f"basename: {basename}, input_dir: {input_dir}"
        )
    basenames.add(basename)


def get_inputs_and_outputs(input_dir, output_dir):
    """Reads files from 'input_dir' and creates corresponding paired outputs.

    Args:
      input_dir: Where to read inputs from.
      output_dir: Where to read outputs from.

    Returns:
      inputs and outputs.
    """

    # Initialization
    basenames = set()
    inputs, outputs = [], []
    input_files = tf.io.gfile.listdir(input_dir)

    for filename in input_files:

        if not _is_supported(filename):
            print(f"Skipping unsupported file: {filename}")
            continue

        basename, _ = os.path.splitext(filename)
        _check_basename(basenames, basename, input_dir)
        inputs.append(filename)
        output = f"{basename}.tfrecord"
        outputs.append(output)

    inputs = [os.path.join(input_dir, i) for i in inputs]
    outputs = [os.path.join(output_dir, o) for o in outputs]

    return inputs, outputs
