try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ModuleNotFoundError:
    LEGGED_GYM_ROOT_DIR='../..'
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self._config = config

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]
            
            self.kps = config.get("kps", None)
            self.kds = config.get("kds", None)


            if 'motor_joint' in config:
                self.motor_joint = config['motor_joint']
            else:
                self.motor_joint=[]

            if 'arm_joint' in config:
                self.arm_joint = config['arm_joint']
            else:
                self.arm_joint=[]

            if 'non_arm_joint' in config:
                self.non_arm_joint = config['non_arm_joint']
            else:
                self.non_arm_joint=[]

            if 'lab_joint' in config:
                self.lab_joint = config['lab_joint']
            else:
                self.lab_joint=[]
                
            if 'lab_joint_eetrack' in config:
                self.lab_joint_eetrack = config['lab_joint_eetrack']
            else:
                self.lab_joint_eetrack=[]
                
            if 'lab_joint_offsets' in config:
                self.lab_joint_offsets = config['lab_joint_offsets']
            else:
                self.lab_joint_offsets=[]

            if "lab_joint_offsets_sit" in config:
                self.lab_joint_offsets_sit = config["lab_joint_offsets_sit"]
            else:
                self.lab_joint_offsets_sit = []

            self.locomotion_motor_joint_offsets = config.get("locomotion_motor_joint_offsets", None)
                
            self.ang_vel_scale = config.get("ang_vel_scale", None)
            self.dof_pos_scale = config.get("dof_pos_scale", None)
            self.dof_vel_scale = config.get("dof_vel_scale", None)
            self.action_scale = config.get("action_scale", None)
            self.cmd_scale = config.get("cmd_scale", None)
            self.max_cmd = config.get("max_cmd", None)
            if self.cmd_scale:
                self.cmd_scale = np.array(self.cmd_scale, dtype=np.float32)
            if self.max_cmd:
                self.max_cmd = np.array(self.max_cmd, dtype=np.float32)

            self.num_actions = config.get("num_actions", None)
            self.num_obs = config.get("num_obs", None)

            self.initial_smoothing = config.get("initial_smoothing", None)
            self.later_smoothing = config.get("later_smoothing", None)
            self.kpkd_smoothing =config.get("kpkd_smoothing", None)

            self.ik_joint = config.get("ik_joint", None)
            self.eetrack_right_arm_kps = config.get("eetrack_right_arm_kps", None)
            self.eetrack_right_arm_kds = config.get("eetrack_right_arm_kds", None)

            self.locomotion_policy_path = config.get("locomotion_policy_path", None)
            self.locomotion_obs_dim = config.get("locomotion_obs_dim", None)
            if self.locomotion_policy_path:
                self.locomotion_policy_path = self.locomotion_policy_path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            self.navigation_policy_path = config.get("navigation_policy_path", None)
            if self.navigation_policy_path:
                self.navigation_policy_path = self.navigation_policy_path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            
            self.sit_policy_path = config.get("sit_policy_path", None)
            if self.sit_policy_path:
                self.sit_policy_path = self.sit_policy_path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            
            self.target_height = config.get("target_height", 0.3)
            self.max_velocity = config.get("max_velocity", 0.1)


            self.mot_joint_offsets = config.get("mot_joint_offsets", None)
            self.slow_bound = config.get("slow_bound", 0.2)

            self.sit_kds = config.get("sit_kds", None)

            self.sit_smoothing = config.get("sit_smoothing", 1)