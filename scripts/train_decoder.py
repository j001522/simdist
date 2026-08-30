"""Script to train a debug pixel decoder on a frozen world-model checkpoint."""

import hydra
from omegaconf import DictConfig, OmegaConf

from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()

from simdist.utils.paths import get_train_decoder_hydra_config
from simdist.modeling import decoder_trainer


@hydra.main(**get_train_decoder_hydra_config())
def main(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    decoder_trainer.train(dict_cfg)


if __name__ == "__main__":
    main()
