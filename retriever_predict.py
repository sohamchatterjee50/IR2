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
"""Table retriever experiment."""

import csv
import functools
import os
import traceback
from typing import Text, Optional
from argparse import Namespace

from tapas.models import retriever_model
from tapas.utils import eval_retriever_utils
from tapas.utils import experiment_utils  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import hydra
from omegaconf import DictConfig
tf.disable_v2_behavior()

def _get_test_input_fn(name, input_file, args):
  """Gets input_fn for eval/predict modes."""
  if input_file is None:
    return None
  input_fn = functools.partial(
      retriever_model.input_fn,
      name=name,
      file_patterns=input_file,
      data_format=args.data_format,
      is_training=False,
      max_seq_length=args.max_seq_length,
      compression_type=args.compression_type,
      use_mined_negatives=args.use_mined_negatives,
      include_id=True)
  return input_fn


def _predict_and_export_metrics(
    mode, input_fn, checkpoint_path, step, estimator, output_dir, args
):
    """Exports model predictions and calculates precision@k."""
    tf.logging.info("Running predictor for step %d.", step)
    result = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)
    output_predict_file = os.path.join(output_dir, f"{mode}_results_{step}.tsv")
    write_predictions(result, output_predict_file)

    # Compute precision@k.
    if not args.evaluated_checkpoint_step or not args.evaluated_checkpoint_metric:
        # p_at_k = eval_retriever_utils.eval_precision_at_k(
        #     query_prediction_files=output_predict_file,
        #     table_prediction_files=output_predict_file,
        #     make_tables_unique=True,
        # )
        metrics_at_k = eval_retriever_utils.eval_metrics_at_k(
            query_prediction_files=output_predict_file,
            table_prediction_files=output_predict_file,
            make_tables_unique=True,
        )
        experiment_utils.save_metrics(output_dir, mode, step, metrics_at_k)

def _predict_and_export_metrics(
    mode, input_fn, checkpoint_path, step, estimator, output_dir, args
):
    """Exports model predictions and calculates precision@k."""
    tf.logging.info("Running predictor for step %d.", step)
    result = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)
    output_predict_file = os.path.join(output_dir, f"{mode}_results_{step}.tsv")
    write_predictions(result, output_predict_file)

    # Compute precision@k.
    if not args.evaluated_checkpoint_step or not args.evaluated_checkpoint_metric:
        # p_at_k = eval_retriever_utils.eval_precision_at_k(
        #     query_prediction_files=output_predict_file,
        #     table_prediction_files=output_predict_file,
        #     make_tables_unique=True,
        # )
        metrics_at_k = eval_retriever_utils.eval_metrics_at_k(
            query_prediction_files=output_predict_file,
            table_prediction_files=output_predict_file,
            make_tables_unique=True,
        )
        experiment_utils.save_metrics(output_dir, mode, step, metrics_at_k)

def write_predictions(predictions,
                      output_predict_file):
  """Writes predictions to an output TSV file.

  Predictions header: [query_id, query_rep, table_id, table_rep]
  Args:
    predictions: model predictions
    output_predict_file: Path for wrinting the predicitons.
  """
  with tf.io.gfile.GFile(output_predict_file, "w") as write_file:
    header = [
        "query_id",
        "query_rep",
        "table_id",
        "table_rep",
    ]
    writer = csv.DictWriter(write_file, fieldnames=header, delimiter="\t")
    writer.writeheader()

    for prediction in predictions:
      query_id = prediction["query_id"]
      table_id = prediction["table_id"]
      query_rep = prediction["query_rep"]
      table_rep = prediction["table_rep"]

      prediction_to_write = {
          "query_id": query_id[0].decode("utf-8"),
          "query_rep": query_rep.tolist(),
          "table_id": table_id[0].decode("utf-8"),
          "table_rep": table_rep.tolist(),
      }
      writer.writerow(prediction_to_write)


@hydra.main(version_base=None, config_path="configs", config_name="predict_retriever")
def main(cfg: DictConfig):

    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)
    bert_config = experiment_utils.bert_config_from_flags()
    total_steps = experiment_utils.num_train_steps()
    retriever_config = retriever_model.RetrieverConfig(
        bert_config=bert_config,
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=total_steps,
        num_warmup_steps=experiment_utils.num_warmup_steps(),
        grad_clipping=args.grad_clipping,
        down_projection_dim=args.down_projection_dim,
        init_from_single_encoder=args.init_from_single_encoder,
        max_query_length=args.max_query_length,
        mask_repeated_tables=args.mask_repeated_tables,
        mask_repeated_questions=args.mask_repeated_questions,
        #use_out_of_core_negatives=args.use_out_of_core_negatives,
        ignore_table_content=args.ignore_table_content,
        disabled_features=args.disabled_features,
        use_mined_negatives=args.use_mined_negatives,
    )

    model_fn = retriever_model.model_fn_builder(retriever_config)
    estimator = experiment_utils.build_estimator(model_fn)

    if args.do_train:
        tf.io.gfile.makedirs(args.model_dir)
        bert_config.to_json_file(os.path.join(args.model_dir, "bert_config.json"))
        retriever_config.to_json_file(
            os.path.join(args.model_dir, "tapas_config.json"))
        train_input_fn = functools.partial(
            retriever_model.input_fn,
            name="train",
            file_patterns=args.input_file_train,
            data_format=args.data_format,
            is_training=True,
            max_seq_length=args.max_seq_length,
            compression_type=args.compression_type,
            use_mined_negatives=args.use_mined_negatives,
            include_id=False)
        estimator.train(input_fn=train_input_fn, max_steps=total_steps)

    eval_input_fn = _get_test_input_fn("eval", args.input_file_eval, args)
    if args.do_eval:
        if eval_input_fn is None:
            raise ValueError("No input_file_eval specified!")
        for _, checkpoint in experiment_utils.iterate_checkpoints(
            model_dir=estimator.model_dir,
            total_steps=total_steps,
            marker_file_prefix=os.path.join(
                estimator.model_dir, f"eval_{args.eval_name}"
            ),
            minutes_to_sleep=args.minutes_to_sleep_before_predictions,
        ):
            tf.logging.info("Running eval: %s", args.eval_name)
            try:
                result = estimator.evaluate(
                    input_fn=eval_input_fn,
                    steps=args.num_eval_steps,
                    name=args.eval_name,
                    checkpoint_path=checkpoint,
                )
                tf.logging.info("Eval result:\n%s", result)
            except (ValueError, tf.errors.NotFoundError):
                tf.logging.error(
                    "Error getting predictions for checkpoint %s: %s",
                    checkpoint,
                    traceback.format_exc(),
                )

    if args.do_predict:
        predict_input_fn = _get_test_input_fn("predict", args.input_file_predict, args)
        if args.prediction_output_dir:
            prediction_output_dir = args.prediction_output_dir
            tf.io.gfile.makedirs(prediction_output_dir)
        else:
            prediction_output_dir = estimator.model_dir

        marker_file_prefix = os.path.join(prediction_output_dir, "predict")
        # When two separate jobs are launched we don't want conflicting markers.
        if predict_input_fn is not None:
            marker_file_prefix += "_test"
        if eval_input_fn is not None:
            marker_file_prefix += "_dev"

        single_step = args.evaluated_checkpoint_step
        # if args.evaluated_checkpoint_metric:
        #     single_step = experiment_utils.get_best_step_for_metric(
        #         estimator.model_dir, args.evaluated_checkpoint_metric
        #     )
        # for current_step, checkpoint in experiment_utils.iterate_checkpoints(
        #     model_dir=estimator.model_dir,
        #     total_steps=total_steps,
        #     marker_file_prefix=marker_file_prefix,
        #     single_step=single_step,
        # ):
        for current_step, checkpoint in [(0, "model.ckpt-11300")]:
            checkpoint = os.path.join(args.model_dir, checkpoint)
            print(checkpoint)
            try:
                if predict_input_fn is not None:
                    _predict_and_export_metrics(
                        mode="predict",
                        input_fn=predict_input_fn,
                        checkpoint_path=checkpoint,
                        step=current_step,
                        estimator=estimator,
                        output_dir=prediction_output_dir,
                        args=args
                    )

                if eval_input_fn is not None:
                    _predict_and_export_metrics(
                        mode="eval",
                        input_fn=eval_input_fn,
                        checkpoint_path=checkpoint,
                        step=current_step,
                        estimator=estimator,
                        output_dir=prediction_output_dir,
                        args=args
                    )
            except (ValueError, tf.errors.NotFoundError):
                tf.logging.error(
                    "Error getting predictions for checkpoint %s: %s",
                    checkpoint,
                    traceback.format_exc(),
                )


if __name__ == "__main__":
  main()