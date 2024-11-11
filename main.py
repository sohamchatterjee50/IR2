import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):

    general = cfg.get("general")
    for key, val in general.items():
        print(f"Key: {key}, Val: {val}, Type: {type(val)}")


if __name__ == "__main__":

    main()
