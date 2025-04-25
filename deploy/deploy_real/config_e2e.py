try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ModuleNotFoundError:
    LEGGED_GYM_ROOT_DIR='../..'
import numpy as np
import yaml
from config import Config

class E2EConfig(Config):
    def __init__(self, file_path) -> None:
        super().__init__(file_path)

        self.slow_bound = self._config["slow_bound"]
        self.max_velocity = self._config["max_velocity"]
        self.target_height = self._config["target_height"]
        self.sit_obs_dim = self._config["sit_obs_dim"]
        self.eetrack_obs_dim = self._config["eetrack_obs_dim"]
        self.lower_joint = self._config["lower_joint"]

        self.sit_policy_path = self._config["sit_policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        self.eetrack_policy_path = self._config["eetrack_policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        # ver2
        self.eetrack_joint_offsets = self._config["eetrack_joint_offsets"]
        if 'rest_joint' in self._config:
            self.rest_joint = self._config['rest_joint']

        if 'arm_joint' in self._config:
            self.arm_joint = self._config['arm_joint']