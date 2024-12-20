## Helper Functions for Local File Handling

import tensorflow._api.v2.compat.v1 as tf
from absl import logging


def make_directories(path: str):
    """Create directory recursively. Don't do anything if directory exits."""
    tf.io.gfile.makedirs(path)


def list_directory(path: str):
    """List directory contents."""
    return tf.io.gfile.listdir(path)


def _print(message):

    logging.info(message)
    print(message)


def _warn(msg):
    print(f"Warning: {msg}")
    logging.warn(msg)


def _to_tf_compression_type(
    compression_type: str,
):
    if not compression_type:
        return tf.io.TFRecordCompressionType.NONE

    if compression_type == "GZIP":
        return tf.io.TFRecordCompressionType.GZIP

    if compression_type == "ZLIB":
        return tf.io.TFRecordCompressionType.ZLIB

    raise ValueError(f"Unknown compression type: {compression_type}")
