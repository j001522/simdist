"""Script to merge generated data into a dataset and compute scaler params."""

import hydra
from omegaconf import DictConfig, OmegaConf

from simdist.utils.paths import get_process_data_hydra_config
from simdist.utils import config as config_utils


@hydra.main(**get_process_data_hydra_config())
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)

    # Image extero (list dims, e.g. UR5e RGB) -> manip processor (h5py-direct, Isaac-free,
    # JPEG passthrough). Flat-vector extero (Go2) -> the original DataProcessor.
    if config_utils.extero_obs_image_shapes_from_sys_config(dict_cfg["system"]):
        from simdist.data.manip_data_processor import ManipulationDataProcessor

        data_processor = ManipulationDataProcessor(dict_cfg)
    else:
        from simdist.data.data_processor import DataProcessor

        data_processor = DataProcessor(dict_cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
