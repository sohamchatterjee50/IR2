## Utilities to help with Experimentation

import os, json, time, datetime
import tensorflow._api.v2.compat.v1 as tf
from absl import logging
from argparse import Namespace
from tapas.models.bert import modeling
from tapas.utils import calc_metric_utils
from tensorflow._api.v2.compat.v1 import estimator as tf_estimator
import omegaconf


# To get arguments from .yaml configuration
def get_args():
    cfg = omegaconf.OmegaConf.load("configs/experiments.yaml")
    general = cfg.get("base")
    args = Namespace(**general)
    return args


args = get_args()


def get_num_eval_steps():

    return args.num_eval_steps


def get_learning_rate():

    return args.learning_rate


def bert_config_from_flags():
    """Reads the BERT config from flags."""
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
    if args.bert_config_attention_probs_dropout_prob is not None:
        bert_config.attention_probs_dropout_prob = (
            args.bert_config_attention_probs_dropout_prob
        )

    if args.bert_config_hidden_dropout_prob is not None:
        bert_config.hidden_dropout_prob = args.bert_config_hidden_dropout_prob

    if args.bert_config_initializer_range is not None:
        bert_config.initializer_range = args.bert_config_initializer_range

    if args.bert_config_softmax_temperature is not None:
        bert_config.softmax_temperature = args.bert_config_softmax_temperature

    return bert_config


def num_train_steps():

    if args.num_train_examples is None:
        return None

    return args.num_train_examples // args.train_batch_size


def num_warmup_steps():

    num_steps = num_train_steps()
    if num_steps is None:
        return None

    return int(num_steps * args.warmup_ratio)


def build_estimator(model_fn, model_dir):
    """Builds a TPUEstimator using the common experiment flags."""

    is_per_host = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf_estimator.tpu.RunConfig(
        cluster=None,
        master=args.master,
        model_dir=model_dir,
        tf_random_seed=args.tf_random_seed,
        save_checkpoints_steps=args.save_checkpoints_steps,
        keep_checkpoint_max=args.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=args.keep_checkpoint_every_n_hours,
        tpu_config=tf_estimator.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=1,
            per_host_input_for_training=is_per_host,
        ),
    )
    # As TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    return tf_estimator.tpu.TPUEstimator(
        params={
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "drop_remainder": False,
            "max_eval_count": args.max_eval_count,
        },
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=args.train_batch_size // args.gradient_accumulation_steps,
        eval_batch_size=args.eval_batch_size,
        predict_batch_size=args.predict_batch_size,
    )


def iterate_checkpoints(
    model_dir,
    marker_file_prefix,
    total_steps,
    single_step=None,
    minutes_to_sleep=5,
):
    """Iterate over checkpoints as they appear until the final one is reached.

    By default iterates over all checkpoints until completion, if `single_step`
    argument is used only that checkpoint is returned, whether it exists or not.
    The `marker_file_prefix` is used to write empty files as a checkpoint is
    successfully yielded to prevent repeated work if the job is restarted.

    Args:
      model_dir: Location where checkpoints live.
      marker_file_prefix: Location to write an empty file that marks the
        checkpoint as processed successfully. Ignored when `single_step` is used.
      total_steps: After each read over checkpoints, finish if `total_steps` is
        reached or `None` is passed.
      single_step: If specified, only return the checkpoint for this step.
      minutes_to_sleep: Number of minutes to sleep between iterations.

    Yields:
      A tuple with a step number and checkpoint path
    """
    if single_step is not None:
        checkpoint = os.path.join(model_dir, f"model.ckpt-{single_step}")
        yield single_step, checkpoint
        return

    done = set()
    while True:
        state = tf.train.get_checkpoint_state(model_dir)
        checkpoints = state.all_model_checkpoint_paths if state is not None else []
        found_pending_checkpoint = False
        for checkpoint in checkpoints:

            step = int(os.path.basename(checkpoint).split("-")[1])
            if step in done:
                continue

            done.add(step)
            if not tf.gfile.Exists(f"{checkpoint}.index"):
                tf.logging.info(f"Skipping step {step} since checkpoint is missing")
                continue

            marker_file = f"{marker_file_prefix}-{step}.done"
            if tf.gfile.Exists(marker_file):
                tf.logging.info(f"Skipping step {step} since marker file was found")
                continue

            yield step, checkpoint
            # To force the file to be created we need to write something in it.
            with tf.io.gfile.GFile(marker_file, "w") as f:
                f.write(datetime.datetime.now().isoformat() + "\n")

            found_pending_checkpoint = True
            # We will restart the loop since the some checkpoints might have been
            # deleted in the meantime.
            break

        if checkpoints and (total_steps is None or step >= total_steps):
            tf.logging.info(f"Checkpoint loop finished after step {step}")
            return

        if not found_pending_checkpoint and minutes_to_sleep > 0:
            tf.logging.info(f"Sleeping {minutes_to_sleep} mins before next loop")
            time.sleep(minutes_to_sleep * 60)


def save_metrics(
    model_dir,
    mode,
    step,
    metrics,
):
    """Save metrics to file and TensorBoard."""
    calc_metric_utils.write_to_tensorboard(metrics, step, os.path.join(model_dir, mode))
    metric_file_path = os.path.join(model_dir, f"{mode}_metrics_{step}.json")
    logging.info("Writing metrics to: %s", metric_file_path)

    with tf.io.gfile.GFile(metric_file_path, "w") as f:
        metrics = dict(metrics, step=step)
        f.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def get_best_step_for_metric(
    model_dir,
    metric,
):
    """Finds measurements for a metric and returns the step with the best value.

    Args:
      model_dir: Location where the model was trained.
      metric: Metric to select.

    Raises:
      ValueError: if no metric files are found.

    Returns:
      Step that maximizes the metric.
    """
    filepattern = os.path.join(model_dir, "eval_metrics_*.json")
    measurements = []
    for path in tf.io.gfile.glob(filepattern):

        with tf.io.gfile.GFile(path) as f:

            measurement = json.load(f)
            measurements.append((measurement[metric], measurement["step"]))

    if not measurements:
        raise ValueError(f"Metrics missing in {model_dir}")

    return max(measurements)[1]
