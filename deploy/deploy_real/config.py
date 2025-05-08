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

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            # self.lowcmd_topic = config["lowcmd_topic"]
            # self.lowstate_topic = config["lowstate_topic"]

            if 'leg_joint2motor_idx' in config:
                self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            if 'joint2motor_idx' in config:
                self.joint2motor_idx = config["joint2motor_idx"]
            
            self.kps = config.get("kps", None)
            self.kds = config.get("kds", None)
            self.sit_kps = config.get("sit_kps", None)
            self.sit_kds = config.get("sit_kds", None)
            self.eetrack_kps = config.get("eetrack_kps", None)
            self.eetrack_kds = config.get("eetrack_kds", None)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            if 'arm_waist_joint2motor_idx' in config:
                self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
                self.arm_waist_kps = config["arm_waist_kps"]
                self.arm_waist_kds = config["arm_waist_kds"]
                self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)
            else:
                self.arm_waist_joint2motor_idx = []
                self.arm_waist_kps = []
                self.arm_waist_kds = []
                self.arm_waist_target = []

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
                
            # self.ang_vel_scale = config["ang_vel_scale"]
            # self.dof_pos_scale = config["dof_pos_scale"]
            # self.dof_vel_scale = config["dof_vel_scale"]
            # self.action_scale = config["action_scale"]
            # self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            # self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            # self.num_actions = config["num_actions"]
            self.num_obs = config.get("num_obs", None)
            # self.joint_pos_target_path = config["joint_pos_target_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            # self.upper_body_joint = config["upper_body_joint"]
            # self.lower_body_joint = config["lower_body_joint"]
            self.jpa_joint = config.get("jpa_joint", None)
            self.rjpa_joint = config.get("rjpa_joint", None)

            self.exp_name = config.get("exp_name", None)

            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            self.slow_bound = config.get("slow_bound", None)
            self.max_velocity = config.get("max_velocity", None)
            self.target_height = config.get("target_height", None)

            self.initial_smoothing = config.get("initial_smoothing", None)
            self.later_smoothing = config.get("later_smoothing", None)
            self.kpkd_smoothing =config.get("kpkd_smoothing", None)

            self.ik_joint = config.get("ik_joint", None)
            self.eetrack_right_arm_kps = config.get("eetrack_right_arm_kps", None)
            self.eetrack_right_arm_kds = config.get("eetrack_right_arm_kds", None)

            self.locomotion_policy_path = config.get("locomotion_policy_path", None)
            self.locomotion_obs_dim = config.get("locomotion_obs_dim", None)
            self.locomotion_policy_path = self.locomotion_policy_path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
