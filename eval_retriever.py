## Evaluates recall@k scores for Table Retriever Predictions
## (This also generates the KNN files for reader experiments)
import hydra
from argparse import Namespace
from omegaconf import DictConfig
from tapas.utils import eval_table_retriever_utils


@hydra.main(version_base=None, config_path="configs", config_name="eval_retriever")
def main(cfg: DictConfig):

    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)

    if args.prediction_files_global:
        eval_table_retriever_utils.eval_recall_at_k(
            args.prediction_files_local,
            args.prediction_files_global,
            make_tables_unique=True,
            retrieval_results_file_path=args.retrieval_results_file_path,
        )

    else:
        eval_table_retriever_utils.eval_recall_at_k(
            args.prediction_files_local,
            args.prediction_files_local,
            make_tables_unique=True,
            retrieval_results_file_path=args.retrieval_results_file_path,
        )


if __name__ == "__main__":

    main()
