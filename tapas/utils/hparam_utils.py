## Contains the Best Hyper-parameter Configurations for Different Tasks

from tapas.task_utils import tasks


def get_sqa_hparams():
    return get_hparams(tasks.Task.SQA)


def get_wikisql_super_hparams():
    return get_hparams(tasks.Task.WIKISQL_SUPERVISED)


def get_wikisql_hparams():
    return get_hparams(tasks.Task.WIKISQL)


def get_hparams(task):
    """Gets hpraram dictionary for a given tasks."""
    if task == tasks.Task.SQA:
        return {
            "init_cell_selection_weights_to_zero": False,
            "learning_rate": 5e-5 * (128 / 512),
            "num_train_examples": 200000 * 128,
            "select_one_column": True,
            "allow_empty_column_selection": False,
            "train_batch_size": 128,
            "warmup_ratio": 0.01,
        }

    params = {
        "grad_clipping": 10.0,
        "num_train_examples": 50000 * 512,
        "train_batch_size": 512,
    }

    if task == tasks.Task.WIKISQL:
        params.update(
            {
                "answer_loss_cutoff": 0.185567,
                "cell_select_pref": 0.611754,
                "huber_loss_delta": 1265.74,
                "init_cell_selection_weights_to_zero": False,
                "learning_rate": 0.0000617164,
                "select_one_column": False,
                "allow_empty_column_selection": False,
                "temperature": 0.107515,
                "warmup_ratio": 0.142400,
            }
        )

    elif task == tasks.Task.WIKISQL_SUPERVISED:
        params.update(
            {
                "answer_loss_cutoff": 36.4519,
                "cell_select_pref": 0.903421,
                "huber_loss_delta": 222.088,
                "init_cell_selection_weights_to_zero": True,
                "learning_rate": 0.0000412331,
                "select_one_column": True,
                "allow_empty_column_selection": True,
                "temperature": 0.763141,
                "warmup_ratio": 0.168479,
            }
        )

    elif task == tasks.Task.NQ_RETRIEVAL:
        params.update(
            {
                "disable_per_token_loss": False,
                "grad_clipping": 10.0,
                "num_classification_labels": 2,
                "num_aggregation_labels": 0,
                "init_cell_selection_weights_to_zero": False,
                "num_train_examples": 50000 * 512,
                "select_one_column": False,
                "allow_empty_column_selection": True,
                "train_batch_size": 512,
                "compute_e2e_retrieval_metrics": True,
                "compute_denotation_accuracy": False,
                "mask_examples_without_labels": True,
                "bert_config_attention_probs_dropout_prob": 0.034,
                "bert_config_hidden_dropout_prob": 0.2,
                "learning_rate": 1e-06,
                "warmup_ratio": 0.0,
                "span_prediction": "span",
            }
        )

    else:
        raise ValueError(f"Unknown task: {task.name}")

    return params
