## Script for all Reader-related experiments

import os, enum, time, hydra, random, functools
import tensorflow._api.v2.compat.v1 as tf
from absl import logging
from argparse import Namespace
from omegaconf import DictConfig
from tapas.task_utils import tasks, task_utils
from tapas.models import classifier_model
from tapas.models.bert import modeling
from tapas.utils import (
    prediction_utils,
    calc_metric_utils,
    hparam_utils,
    number_annot_utils,
    pruning_utils,
    tf_example_utils,
    e2e_utils,
)
from tapas.utils.constants import (
    _MAX_TABLE_ID,
    _MAX_PREDICTIONS_PER_SEQ,
    _CELL_CLASSIFICATION_THRESHOLD,
)
from tensorflow._api.v2.compat.v1 import estimator as tf_estimator


tf.disable_v2_behavior()


class Mode(enum.Enum):
    CREATE_DATA = 1
    TRAIN = 2
    PREDICT_AND_EVALUATE = 3
    EVALUATE = 4
    PREDICT = 5


class TestSet(enum.Enum):
    DEV = 1
    TEST = 2


# File Utils ############################
def make_directories(path):
    """Create directory recursively. Don't do anything if directory exits."""
    tf.io.gfile.makedirs(path)


def list_directory(path):
    """List directory contents."""
    return tf.io.gfile.listdir(path)


#########################################


def _create_measurements_for_metrics(
    metrics,
    global_step,
    model_dir,
    name,
):
    """Reports metrics."""
    for label, value in metrics.items():
        _print(f"{name} {label}: {value:0.4f}")
    logdir = os.path.join(model_dir, name)
    calc_metric_utils.write_to_tensorboard(metrics, global_step, logdir)


def _print(msg):
    print(msg)
    logging.info(msg)


def _warn(msg):
    print(f"Warning: {msg}")
    logging.warn(msg)


def _create_all_examples(
    task, vocab_file, test_mode, output_dir, test_batch_size, args
):
    """Converts interactions to TF examples."""
    interaction_dir = task_utils.get_interaction_dir(output_dir)
    example_dir = os.path.join(output_dir, "tf_examples")
    make_directories(example_dir)

    _create_examples(
        task,
        interaction_dir,
        example_dir,
        vocab_file,
        task_utils.get_train_filename(task),
        batch_size=None,
        test_mode=test_mode,
        args=args,
    )
    _create_examples(
        task,
        interaction_dir,
        example_dir,
        vocab_file,
        task_utils.get_dev_filename(task),
        test_batch_size,
        test_mode,
        args,
    )
    _create_examples(
        task,
        interaction_dir,
        example_dir,
        vocab_file,
        task_utils.get_test_filename(task),
        test_batch_size,
        test_mode,
        args,
    )


def _to_tf_compression_type(
    compression_type,
):
    if not compression_type:
        return tf.io.TFRecordCompressionType.NONE

    if compression_type == "GZIP":
        return tf.io.TFRecordCompressionType.GZIP

    if compression_type == "ZLIB":
        return tf.io.TFRecordCompressionType.ZLIB

    raise ValueError(f"Unknown compression type: {compression_type}")


def _create_examples(
    task,
    interaction_dir,
    example_dir,
    vocab_file,
    filename,
    batch_size,
    test_mode,
    args,
):
    """Creates TF example for a single dataset."""

    filename = f"{filename}.tfrecord"
    interaction_path = os.path.join(interaction_dir, filename)
    example_path = os.path.join(example_dir, filename)

    config = tf_example_utils.ClassifierConversionConfig(
        vocab_file=vocab_file,
        max_seq_length=args.max_seq_length,
        use_document_title=args.use_document_title,
        update_answer_coordinates=args.update_answer_coordinates,
        drop_rows_to_fit=args.drop_rows_to_fit,
        max_column_id=_MAX_TABLE_ID,
        max_row_id=_MAX_TABLE_ID,
        strip_column_names=False,
        add_aggregation_candidates=False,
        expand_entity_descriptions=False,
    )
    converter = tf_example_utils.ToClassifierTensorflowExample(config)

    examples = []
    num_questions, num_conversion_errors = 0, 0
    for interaction in prediction_utils.iterate_interactions(interaction_path):

        number_annot_utils.add_numeric_values(interaction)
        for i in range(len(interaction.questions)):

            num_questions += 1
            try:
                examples.append(converter.convert(interaction, i))

            except ValueError as e:
                num_conversion_errors += 1
                logging.info(
                    "Can't convert interaction: %s error: %s", interaction.id, e
                )

        if test_mode and len(examples) >= 100:
            break

    _print(f"Processed: {filename}")
    _print(f"Num questions processed: {num_questions}")
    _print(f"Num examples: {len(examples)}")
    _print(f"Num conversion errors: {num_conversion_errors}")

    if batch_size is None:
        random.shuffle(examples)

    else:
        original_num_examples = len(examples)
        while len(examples) % batch_size != 0:
            examples.append(converter.get_empty_example())

        if original_num_examples != len(examples):
            _print(f"Padded with {len(examples) - original_num_examples} examples.")

    with tf.io.TFRecordWriter(
        example_path,
        options=_to_tf_compression_type(args.compression_type),
    ) as writer:

        for example in examples:

            writer.write(example.SerializeToString())


def _get_train_examples_file(task, output_dir):
    return os.path.join(
        output_dir, "tf_examples", f"{task_utils.get_train_filename(task)}.tfrecord"
    )


def _get_test_filename(task, test_set):
    if test_set == TestSet.TEST:
        return task_utils.get_test_filename(task)

    if test_set == TestSet.DEV:
        return task_utils.get_dev_filename(task)

    raise ValueError(f"Unknown test set: {test_set}")


def _get_test_examples_file(
    task,
    output_dir,
    test_set,
):
    filename = _get_test_filename(task, test_set)
    return os.path.join(output_dir, "tf_examples", f"{filename}.tfrecord")


def _get_test_interactions_file(
    task,
    output_dir,
    test_set,
):
    filename = _get_test_filename(task, test_set)
    return os.path.join(output_dir, "interactions", f"{filename}.tfrecord")


def _get_test_prediction_file(
    task,
    model_dir,
    test_set,
    is_sequence,
    global_step,
):
    """Get prediction filename for different tasks and setups."""
    suffix = "" if global_step is None else f"_{global_step}"
    if is_sequence:
        suffix = f"_sequence{suffix}"

    filename = _get_test_filename(task, test_set)
    return os.path.join(model_dir, f"{filename}{suffix}.tsv")


def _get_token_selector(args):
    if not args.prune_columns:
        return None

    return pruning_utils.HeuristicExactMatchTokenSelector(
        args.bert_vocab_file,
        args.max_seq_length,
        pruning_utils.SelectionType.COLUMN,
        # Only relevant for SQA where questions come in sequence
        use_previous_answer=True,
        use_previous_questions=True,
    )


def _train_and_predict(
    task,
    test_batch_size,
    train_batch_size,
    gradient_accumulation_steps,
    bert_config_file,
    init_checkpoint,
    test_mode,
    mode,
    output_dir,
    model_dir,
    loop_predict,
    args,
):
    """Trains, produces test predictions and eval metric."""
    make_directories(model_dir)

    if task == tasks.Task.SQA:
        num_aggregation_labels = 0
        num_classification_labels = 0
        use_answer_as_supervision = False

    elif task in [tasks.Task.WIKISQL, tasks.Task.WIKISQL_SUPERVISED]:
        num_aggregation_labels = 4
        num_classification_labels = 0
        use_answer_as_supervision = task != tasks.Task.WIKISQL_SUPERVISED

    elif task == tasks.Task.NQ_RETRIEVAL:
        num_aggregation_labels = 0
        num_classification_labels = 2
        use_answer_as_supervision = False

    else:
        raise ValueError(f"Unknown task: {task.name}")

    do_model_aggregation = num_aggregation_labels > 0
    do_model_classification = num_classification_labels > 0

    hparams = hparam_utils.get_hparams(task)
    if test_mode:
        if train_batch_size is None:
            train_batch_size = 1

        test_batch_size = 1
        num_train_steps = 10
        num_warmup_steps = 1

    else:
        if train_batch_size is None:
            train_batch_size = hparams["train_batch_size"]

        num_train_examples = hparams["num_train_examples"]
        num_train_steps = int(num_train_examples / train_batch_size)
        num_warmup_steps = int(num_train_steps * hparams["warmup_ratio"])

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    if "bert_config_attention_probs_dropout_prob" in hparams:
        bert_config.attention_probs_dropout_prob = hparams.get(
            "bert_config_attention_probs_dropout_prob"
        )

    if "bert_config_hidden_dropout_prob" in hparams:
        bert_config.hidden_dropout_prob = hparams.get("bert_config_hidden_dropout_prob")

    tapas_config = classifier_model.TapasClassifierConfig(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=hparams["learning_rate"],
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        positive_weight=10.0,
        num_aggregation_labels=num_aggregation_labels,
        num_classification_labels=num_classification_labels,
        aggregation_loss_importance=1.0,
        use_answer_as_supervision=use_answer_as_supervision,
        answer_loss_importance=1.0,
        use_normalized_answer_loss=False,
        huber_loss_delta=hparams.get("huber_loss_delta"),
        temperature=hparams.get("temperature", 1.0),
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function=(
            classifier_model.AverageApproximationFunction.RATIO
        ),
        cell_select_pref=hparams.get("cell_select_pref"),
        answer_loss_cutoff=hparams.get("answer_loss_cutoff"),
        grad_clipping=hparams.get("grad_clipping"),
        disabled_features=[],
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=False,
        disable_per_token_loss=hparams.get("disable_per_token_loss", False),
        mask_examples_without_labels=hparams.get("mask_examples_without_labels", False),
        init_cell_selection_weights_to_zero=(
            hparams["init_cell_selection_weights_to_zero"]
        ),
        select_one_column=hparams["select_one_column"],
        allow_empty_column_selection=hparams["allow_empty_column_selection"],
        span_prediction=classifier_model.SpanPredictionMode(
            hparams.get("span_prediction", classifier_model.SpanPredictionMode.NONE)
        ),
        disable_position_embeddings=False,
        reset_output_cls=args.reset_output_cls,
        reset_position_index_per_cell=args.reset_position_index_per_cell,
        table_pruning_config_file=args.table_pruning_config_file,
    )

    model_fn = classifier_model.model_fn_builder(tapas_config)
    is_per_host = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf_estimator.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=model_dir,
        tf_random_seed=args.tf_random_seed,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=4.0,
        tpu_config=tf_estimator.tpu.TPUConfig(
            iterations_per_loop=args.iterations_per_loop,
            num_shards=8,
            per_host_input_for_training=is_per_host,
        ),
    )

    # As TPU is not available, we use the normal Estimator on CPU/GPU.
    estimator = tf_estimator.tpu.TPUEstimator(
        params={"gradient_accumulation_steps": gradient_accumulation_steps},
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size // gradient_accumulation_steps,
        eval_batch_size=None,
        predict_batch_size=test_batch_size,
    )

    if mode == Mode.TRAIN:
        _print("Training")
        bert_config.to_json_file(os.path.join(model_dir, "bert_config.json"))
        tapas_config.to_json_file(os.path.join(model_dir, "tapas_config.json"))
        train_input_fn = functools.partial(
            classifier_model.input_fn,
            name="train",
            file_patterns=_get_train_examples_file(task, output_dir),
            data_format="tfrecord",
            compression_type=args.compression_type,
            is_training=True,
            max_seq_length=args.max_seq_length,
            max_predictions_per_seq=_MAX_PREDICTIONS_PER_SEQ,
            add_aggregation_function_id=do_model_aggregation,
            add_classification_labels=do_model_classification,
            add_answer=use_answer_as_supervision,
            include_id=False,
        )
        estimator.train(
            input_fn=train_input_fn,
            max_steps=tapas_config.num_train_steps,
        )

    elif mode == Mode.PREDICT_AND_EVALUATE or mode == Mode.PREDICT:

        # Starts a continous eval that starts with the latest checkpoint and runs
        # until a checkpoint with 'num_train_steps' is reached.
        prev_checkpoint = None
        while True:
            checkpoint = estimator.latest_checkpoint()

            if not loop_predict and not checkpoint:
                raise ValueError(f"No checkpoint found at {model_dir}.")

            if loop_predict and checkpoint == prev_checkpoint:
                _print("Sleeping 5 mins before predicting")
                time.sleep(5 * 60)
                continue

            current_step = int(os.path.basename(checkpoint).split("-")[1])
            _predict(
                estimator,
                task,
                output_dir,
                model_dir,
                do_model_aggregation,
                do_model_classification,
                use_answer_as_supervision,
                global_step=current_step,
                args=args,
            )

            if mode == Mode.PREDICT_AND_EVALUATE:
                _eval(
                    task=task,
                    output_dir=output_dir,
                    model_dir=model_dir,
                    global_step=current_step,
                    args=args,
                )

            if not loop_predict or current_step >= tapas_config.num_train_steps:
                _print(f"Evaluation finished after training step {current_step}.")
                break

            prev_checkpoint = checkpoint

    else:
        raise ValueError(f"Unexpected mode: {mode}.")


def _predict(
    estimator,
    task,
    output_dir,
    model_dir,
    do_model_aggregation,
    do_model_classification,
    use_answer_as_supervision,
    global_step,
    args,
):
    """Writes predictions for dev and test."""
    for test_set in TestSet:

        _predict_for_set(
            estimator,
            do_model_aggregation,
            do_model_classification,
            use_answer_as_supervision,
            example_file=_get_test_examples_file(
                task,
                output_dir,
                test_set,
            ),
            prediction_file=_get_test_prediction_file(
                task,
                model_dir,
                test_set,
                is_sequence=False,
                global_step=global_step,
            ),
            other_prediction_file=_get_test_prediction_file(
                task,
                model_dir,
                test_set,
                is_sequence=False,
                global_step=None,
            ),
            args=args,
        )
    if task == tasks.Task.SQA:

        for test_set in TestSet:
            _predict_sequence_for_set(
                estimator,
                do_model_aggregation,
                use_answer_as_supervision,
                example_file=_get_test_examples_file(task, output_dir, test_set),
                prediction_file=_get_test_prediction_file(
                    task,
                    model_dir,
                    test_set,
                    is_sequence=True,
                    global_step=global_step,
                ),
                other_prediction_file=_get_test_prediction_file(
                    task,
                    model_dir,
                    test_set,
                    is_sequence=True,
                    global_step=None,
                ),
                args=args,
            )


def _predict_for_set(
    estimator,
    do_model_aggregation,
    do_model_classification,
    use_answer_as_supervision,
    example_file,
    prediction_file,
    other_prediction_file,
    args,
):
    """Gets predictions and writes them to TSV file."""
    predict_input_fn = functools.partial(
        classifier_model.input_fn,
        name="predict",
        file_patterns=example_file,
        data_format="tfrecord",
        compression_type=args.compression_type,
        is_training=False,
        max_seq_length=args.max_seq_length,
        max_predictions_per_seq=_MAX_PREDICTIONS_PER_SEQ,
        add_aggregation_function_id=do_model_aggregation,
        add_classification_labels=do_model_classification,
        add_answer=use_answer_as_supervision,
        include_id=False,
    )
    result = estimator.predict(input_fn=predict_input_fn)
    prediction_utils.write_predictions(
        result,
        prediction_file,
        do_model_aggregation=do_model_aggregation,
        do_model_classification=do_model_classification,
        cell_classification_threshold=_CELL_CLASSIFICATION_THRESHOLD,
        output_token_probabilities=False,
        output_token_answers=True,
    )
    tf.io.gfile.copy(prediction_file, other_prediction_file, overwrite=True)


def _predict_sequence_for_set(
    estimator,
    do_model_aggregation,
    use_answer_as_supervision,
    example_file,
    prediction_file,
    other_prediction_file,
    args,
):
    """Runs realistic sequence evaluation for SQA."""
    examples_by_position = prediction_utils.read_classifier_dataset(
        predict_data=example_file,
        data_format="tfrecord",
        compression_type=args.compression_type,
        max_seq_length=args.max_seq_length,
        max_predictions_per_seq=_MAX_PREDICTIONS_PER_SEQ,
        add_aggregation_function_id=do_model_aggregation,
        add_classification_labels=False,
        add_answer=use_answer_as_supervision,
    )  # pytype: disable=wrong-arg-types
    result = prediction_utils.compute_prediction_sequence(
        estimator=estimator, examples_by_position=examples_by_position
    )
    prediction_utils.write_predictions(
        result,
        prediction_file,
        do_model_aggregation,
        do_model_classification=False,
        cell_classification_threshold=_CELL_CLASSIFICATION_THRESHOLD,
        output_token_probabilities=False,
        output_token_answers=True,
    )
    tf.io.gfile.copy(prediction_file, other_prediction_file, overwrite=True)


def _eval(
    task,
    output_dir,
    model_dir,
    args,
    global_step=None,
):
    """Evaluate dev and test predictions."""
    for test_set in TestSet:

        _eval_for_set(
            model_dir=model_dir,
            name=test_set.name.lower(),
            task=task,
            interaction_file=_get_test_interactions_file(
                task,
                output_dir,
                test_set,
            ),
            prediction_file=_get_test_prediction_file(
                task,
                model_dir,
                test_set,
                is_sequence=False,
                global_step=None,
            ),
            global_step=global_step,
            args=args,
        )

        if task == tasks.Task.SQA:
            _eval_for_set(
                model_dir=model_dir,
                name=f"{test_set.name.lower()}_seq",
                task=task,
                interaction_file=_get_test_interactions_file(
                    task,
                    output_dir,
                    test_set,
                ),
                prediction_file=_get_test_prediction_file(
                    task,
                    model_dir,
                    test_set,
                    is_sequence=True,
                    global_step=None,
                ),
                global_step=global_step,
                args=args,
            )


def _eval_for_set(
    model_dir, name, task, interaction_file, prediction_file, global_step, args
):
    """Computes eval metric from predictions."""
    if not tf.io.gfile.exists(prediction_file):
        _warn(f"Can't evaluate for {name} because {prediction_file} doesn't exist.")
        return
    test_examples = calc_metric_utils.read_data_examples_from_interactions(
        interaction_file
    )
    calc_metric_utils.read_predictions(
        predictions_path=prediction_file,
        examples=test_examples,
    )
    if task in [
        tasks.Task.SQA,
        tasks.Task.WIKISQL,
        tasks.Task.WIKISQL_SUPERVISED,
    ]:
        denotation_accuracy = calc_metric_utils.calc_denotation_accuracy(
            examples=test_examples,
            denotation_errors_path=None,
            predictions_file_name=None,
        )
        if global_step is not None:
            _create_measurements_for_metrics(
                {"denotation_accuracy": denotation_accuracy},
                global_step=global_step,
                model_dir=model_dir,
                name=name,
            )

    elif task == tasks.Task.NQ_RETRIEVAL:
        e2e_metrics = e2e_utils.evaluate_retrieval_e2e(
            interaction_file=interaction_file,
            prediction_file=prediction_file,
            vocab_file=args.bert_vocab_file,
        ).to_dict()
        e2e_metrics = {key: val for key, val in e2e_metrics.items() if val is not None}
        if global_step is not None:
            _create_measurements_for_metrics(
                e2e_metrics,
                global_step=global_step,
                model_dir=model_dir,
                name=name,
            )

    else:
        raise ValueError(f"Unknown task: {task.name}")


def _check_options(output_dir, task, mode):
    """Checks against some invalid options so we can fail fast."""

    if mode == Mode.CREATE_DATA:
        return

    if mode == Mode.PREDICT_AND_EVALUATE or mode == Mode.EVALUATE:
        interactions = _get_test_interactions_file(
            task,
            output_dir,
            test_set=TestSet.DEV,
        )
        if not tf.io.gfile.exists(interactions):
            raise ValueError(f"No interactions found: {interactions}")

    tf_examples = _get_test_examples_file(
        task,
        output_dir,
        test_set=TestSet.DEV,
    )
    if not tf.io.gfile.exists(tf_examples):
        raise ValueError(f"No TF examples found: {tf_examples}")

    _print(f"is_built_with_cuda: {tf.test.is_built_with_cuda()}")
    _print(f"is_gpu_available: {tf.test.is_gpu_available()}")
    _print(f'GPUs: {tf.config.experimental.list_physical_devices("GPU")}')


@hydra.main(version_base=None, config_path="configs", config_name="reader_main")
def main(cfg: DictConfig):

    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)

    if args.tapas_verbosity:
        tf.get_logger().setLevel(args.tapas_verbosity)

    task = tasks.Task[args.task]
    output_dir = os.path.join(args.output_dir, task.name.lower())
    model_dir = args.model_dir or os.path.join(output_dir, "model")
    mode = Mode[args.mode.upper()]
    _check_options(output_dir, task, mode)

    if mode == Mode.CREATE_DATA:
        # Retrieval interactions are model dependant and are created in advance.
        if task != tasks.Task.NQ_RETRIEVAL:
            _print("Creating interactions ...")
            token_selector = _get_token_selector()
            task_utils.create_interactions(
                task, args.input_dir, output_dir, token_selector
            )

        _print("Creating TF examples ...")
        _create_all_examples(
            task,
            args.bert_vocab_file,
            args.test_mode,
            test_batch_size=args.test_batch_size,
            output_dir=output_dir,
            args=args,
        )

    elif mode in (Mode.TRAIN, Mode.PREDICT_AND_EVALUATE, Mode.PREDICT):
        _print("Training or predicting ...")
        _train_and_predict(
            task=task,
            test_batch_size=args.test_batch_size,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            bert_config_file=args.bert_config_file,
            init_checkpoint=args.init_checkpoint,
            test_mode=args.test_mode,
            mode=mode,
            output_dir=output_dir,
            model_dir=model_dir,
            loop_predict=args.loop_predict,
            args=args,
        )

    elif mode == Mode.EVALUATE:
        _eval(
            task=task,
            output_dir=output_dir,
            model_dir=model_dir,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":

    main()
