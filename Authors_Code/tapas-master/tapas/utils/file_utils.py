# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper function for dealing with local files."""

from typing import List, Text
import tensorflow._api.v2.compat.v1 as tf


def make_directories(path):
    """Create directory recursively. Don't do anything if directory exits."""
    tf.io.gfile.makedirs(path)


def list_directory(path):
    """List directory contents."""
    return tf.io.gfile.listdir(path)
