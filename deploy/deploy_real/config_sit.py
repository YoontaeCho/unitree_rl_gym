try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ModuleNotFoundError:
    LEGGED_GYM_ROOT_DIR='../..'
import numpy as np
import yaml
from config import Config

class SitConfig(Config):
    def __init__(self, file_path) -> None:
        super().__init__(file_path)
        self.slow_bound = self._config["slow_bound"]
        self.max_velocity = self._config["max_velocity"]
        self.target_height = self._config["target_height"]
        self.obs_dim = self._config["obs_dim"]
        self.lower_joint = self._config["lower_joint"]