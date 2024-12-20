import hydra
import pandas as pd
from argparse import Namespace
from omegaconf import DictConfig
from tapas.utils.bm25_utils import (
    create_bm25_index,
    iterate_tables,
    iterate_interactions,
)
from tapas.utils.file_utils import _print


def evaluate(index, max_table_rank, thresholds, interactions, rows):
    """Evaluates index against interactions."""
    ranks = []
    for nr, interaction in enumerate(interactions):

        for question in interaction.questions:

            scored_hits = index.retrieve(question.original_text)
            reference_table_id = interaction.table.table_id
            for rank, (table_id, _) in enumerate(scored_hits[:max_table_rank]):

                if table_id == reference_table_id:
                    ranks.append(rank)
                    break

        if nr % (len(interactions) // 10) == 0:
            _print(f"Processed {nr:5d} / {len(interactions):5d}.")

    def recall_at_th(threshold):
        return sum(1 for rank in ranks if rank < threshold) / len(interactions)

    values = [f"{recall_at_th(threshold):.4}" for threshold in thresholds]
    rows.append(values)


def create_index(tables, title_multiplicator):

    return create_bm25_index(
        tables,
        title_multiplicator=title_multiplicator,
    )


def get_hparams():

    hparams = []
    for multiplier in [1, 2]:
        hparams.append({"multiplier": multiplier, "use_bm25": False})

    for multiplier in [10, 15]:
        hparams.append({"multiplier": multiplier, "use_bm25": True})

    return hparams


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):

    hydra_args = cfg.get("bm25")
    args = Namespace(**hydra_args)

    max_table_rank = args.max_table_rank
    thresholds = [1, 5, 10, 15, max_table_rank]

    for interaction_file in args.interaction_files:
        _print(f"Test set: {interaction_file}")
        interactions = list(iterate_interactions(interaction_file))

        for use_local_index in [True, False]:
            rows, row_names = [], []

            for hparams in get_hparams():
                name = "local" if use_local_index else "global"
                name += "_bm25" if hparams["use_bm25"] else "_tfidf"
                name += f'_tm{hparams["multiplier"]}'

                _print(name)
                if use_local_index:
                    index = create_index(
                        tables=(i.table for i in interactions),
                        title_multiplicator=hparams["multiplier"],
                    )

                else:
                    index = create_index(
                        tables=iterate_tables(args.table_file),
                        title_multiplicator=hparams["multiplier"],
                    )

                _print("... index created.")
                evaluate(index, max_table_rank, thresholds, interactions, rows)
                row_names.append(name)

                df = pd.DataFrame(rows, columns=thresholds, index=row_names)
                _print(df.to_string())


if __name__ == "__main__":

    main()
