import os
import torch as th
import numpy as np
import math_utils
from pathlib import Path
from typing import Union
from legged_gym import LEGGED_GYM_ROOT_DIR

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
from common.command_helper_ros import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.remote_controller import RemoteController, KeyMap
from config_sit import SitConfig as Config
from common.crc import CRC
from enum import Enum
from deploy.deploy_real.hoseforce import HoseForceEstimator

import utils_stage as us
from icecream import ic

class Mode(Enum):
    wait = 0
    initial = 1
    hold = 2
    finish = 3
    null = 4


axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)


class GlobalClock:
    def __init__(self, node):
        self.node = node

    def get_time(self):
        return self.node.get_clock().now()


clock = None


def index_map(k_to, k_from):
    """
    Returns an index mapping from k_from to k_to.

    Given k_to=a, k_from=b,
    returns an index map "a_from_b" such that
    array_a[a_from_b] = array_b

    Missing values are set to -1.
    """
    index_dict = {k: i for i, k in enumerate(k_to)}  # O(len(k_from))
    return [index_dict.get(k, -1) for k in k_from]  # O(len(k_to))


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        
        # log path
        self.logpath = Path('/tmp/hose_force/')
        self.logpath.mkdir(parents=True, exist_ok=True)

        # == build index map ==
        self.mot_from_lab = index_map(self.config.motor_joint, self.config.lab_joint)
        self.lab_from_mot = index_map(self.config.lab_joint, self.config.motor_joint)
        
        # num joints
        self.num_joints = len(self.config.motor_joint)
        
        self.initial_pos = np.zeros(self.num_joints)
        
        # log trajectory
        self.timestamp = np.array([])
        self.q_traj = np.zeros((0, self.num_joints))
        self.q_target_traj = np.zeros((0, self.num_joints))
        self.dq_traj = np.zeros((0, self.num_joints))
        self.tau_traj = np.zeros((0, self.num_joints))

        # log ee force
        self.ee_force_traj = np.zeros((0, 6)) # [Fx, Fy, Fz, Mx, My, Mz]
        self.extforce = HoseForceEstimator(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0_replace_with_welder.urdf',
            end_effector='welder',
            config=self.config)

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = LowCmdHG()
            self.low_state = LowStateHG()

            self.lowcmd_publisher_ = self._node.create_publisher(LowCmdHG, 'lowcmd', 10)
            self.lowstate_subscriber = self._node.create_subscription(
                LowStateHG, 'lowstate', self.LowStateHgHandler, 10)
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

        elif config.msg_type == "go":
            raise ValueError(f"{config.msg_type} is not implemented yet.")

        else:
            raise ValueError("Invalid msg_type")

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        # FIXME: you can change the initial mode here
        self.mode = Mode.wait

        self._mode_change = True
        self._terminate = False
        self._timer = self._node.create_timer(self.config.control_dt, self.run_wrapper)
        
        try:
            rp.spin(self._node)
        except KeyboardInterrupt:
            self.log_metrics_and_trajectories()
            print("Log saved.")
        finally:
            self._node.destroy_timer(self._timer)
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self._node.destroy_node()
            rp.shutdown()
            print("Exit")

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.mode_machine = self.mode_machine_
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.publish(cmd)
            
    def save_initial_pos(self):
        """
        Do nothing but save the initial position when 'start' is pressed.
        """
        if self.remote_controller.button[KeyMap.start] == 1:
            self._mode_change = True
            self.mode = Mode.hold
            print("Initial position saved.")
            for mot_idx in range(self.num_joints):
                self.initial_pos[mot_idx] = self.low_state.motor_state[mot_idx].q
            ic(self.initial_pos)

    def run_hold_position(self):
        """
        Keep current position.
        Calculate end effector force & torque based on the estimated joint torque and Jacobian.
        """
        if self.remote_controller.button[KeyMap.A] == 1:
            self._mode_change = True
            self.mode = Mode.finish

        # log current data
        timestamp = clock.get_time().nanoseconds / 1e9
        curr_q = np.zeros(self.num_joints)
        curr_dq = np.zeros(self.num_joints)
        curr_tau = np.zeros(self.num_joints)
        curr_q[self.lab_from_mot] = [self.low_state.motor_state[mot_idx].q for mot_idx in range(self.num_joints)]
        curr_dq[self.lab_from_mot] = [self.low_state.motor_state[mot_idx].dq for mot_idx in range(self.num_joints)]
        curr_tau[self.lab_from_mot] = [self.low_state.motor_state[mot_idx].tau_est for mot_idx in range(self.num_joints)]
        self.timestamp = np.append(self.timestamp, timestamp)
        self.q_traj = np.vstack((self.q_traj, curr_q))
        self.dq_traj = np.vstack((self.dq_traj, curr_dq))
        self.tau_traj = np.vstack((self.tau_traj, curr_tau))

        # target dof pos : initial position
        target_dof_pos = np.zeros(self.num_joints)
        for mot_idx in range(self.num_joints):
            target_dof_pos[mot_idx] = self.initial_pos[mot_idx]
            
        # get ee force
        q_mot = np.zeros(self.num_joints)
        tau_mot = np.zeros(self.num_joints)
        for mot_idx in range(self.num_joints):
            q_mot[mot_idx] = self.low_state.motor_state[mot_idx].q
            tau_mot[mot_idx] = self.low_state.motor_state[mot_idx].tau_est
        F_eef = self.extforce.compute_force(q_mot, tau_mot)
        self.ee_force_traj = np.vstack((self.ee_force_traj, F_eef))
        
        # Build low cmd
        for mot_idx in range(self.num_joints):
            self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].dq = 0.0
            self.low_cmd.motor_cmd[mot_idx].kp = float(self.config.kps[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].kd = float(self.config.kds[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].tau = 0.0
        
        # send the command
        self.send_cmd(self.low_cmd)        
            
    def log_metrics_and_trajectories(self):
        self.timestamp = self.timestamp - self.timestamp[0]
        
        # Save trajectories
        trajectories = {
            "timestamp": self.timestamp,
            "q_traj": self.q_traj,
            "q_target_traj" : self.q_target_traj,
            "dq_traj": self.dq_traj,
            "tau_traj": self.tau_traj,
            "ee_force_traj": self.ee_force_traj,
            "initial_pos": self.initial_pos,
        }

        # Save the log with experiment name
        timestamp = clock.get_time().nanoseconds / 1e9
        file_name = f"{self.logpath}/hose_test_{str(timestamp).split('.')[0]}.npy"
        np.save(file_name, trajectories)
        print(f"Log saved at {file_name}")
        
        # totally terminate
        self._mode_change = True
        self.mode = Mode.null
        

    def run_wrapper(self):
        if self.mode == Mode.wait:
            if self.low_state.crc != 0:
                self.mode = Mode.initial
                self.low_cmd.mode_machine = self.mode_machine_
                print("Successfully connected to the robot.")
        elif self.mode == Mode.initial:
            if self._mode_change:
                print("Entered to initial state.")
                print("Move robot to desired position and press start.")
                self._mode_change = False
            self.save_initial_pos()
        elif self.mode == Mode.hold:
            if self._mode_change:
                print("Collecting EE force...")
                print("Press Button A to finish.")
                self._mode_change = False
            self.run_hold_position()
        elif self.mode == Mode.finish:
            if self._mode_change:
                print("Finish.")
                self._mode_change = False
            self.log_metrics_and_trajectories()
        elif self.mode == Mode.null:
            self._terminate = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="config file name in the configs folder",
        default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    controller = Controller(config)