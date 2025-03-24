from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union, List
import numpy as np
import time
import torch
import torch as th
from pathlib import Path

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo

from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import PoseStamped

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped
from common.command_helper_ros import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
# from config import Config
from config_sit import SitConfig as Config
from common.crc import CRC
from enum import Enum
import pinocchio as pin
from ikctrl import IKCtrl, xyzw2wxyz
from yourdfpy import URDF

import math_utils
import random as rd
from act_to_dof import ActToDof

import utils_metric as um
import utils_stage as us
class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    null = 5



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


def load_action(path: str, interval_len: int=4, env_id:int=0) -> np.ndarray:
    """
    sample random interval from episode.
    N : episode length. (N * 0.02 [s] = total time [s])
    M : number of joints. (maybe 29)
    variables:
        episode : shape(N, M)
        interval_len : interval of episode in second.
        action : shape(interval_len / 0.02, M)
    """
    sim_traj_and_metrics = torch.load(path,map_location=torch.device('cpu'))

    # sim_traj : shape(N, E, M)
    #   - N : episode length
    #   - E : number of episodes
    #   - M : number of joints
    sim_traj = sim_traj_and_metrics["traj"]["joint_pos_target_traj"][:, env_id, :]
    sim_metric = sim_traj_and_metrics["metrics"]

    episode = sim_traj.numpy().astype(np.float32)

    episode_len_int = int(len(episode) * 0.02)
    if episode_len_int <= interval_len:
        action = episode
    else:
        start = np.random.uniform(low=0, high=episode_len_int - interval_len)
        end = start + interval_len
        print(start, end)
        action = episode[int(start / 0.02): int(end / 0.02), :]
        
    return action

import os

from deploy_real_stand import Controller as StandController

class Controller(StandController):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        ### Mapping helpers.
        self.obsmap = us.Stage2Observation(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            config, self.tf_buffer)
        self.actmap = us.SimpleAction(config)
        self.vhcommand = us.VelocityHeightCommand(config)
    
    def run_policy(self):
        logpath = Path('/tmp/metric_test/')
        logpath.mkdir(parents=True, exist_ok=True)

        if self.remote_controller.button[KeyMap.select] == 1:
            self._mode_change = True
            self.mode = Mode.null
            return
        self.counter += 1


        # COMMAND
        world_from_pelvis = us.body_pose(
                self.tf_buffer,
                'pelvis',
                'world',
                rot_type='quat'
            )
        xyz, quat_wxyz = world_from_pelvis

        if self.terminate_by_pelvis_condition(xyz, quat_wxyz):
            raise ValueError

        root_state_w = np.zeros(7)
        root_state_w[0:3] = xyz
        root_state_w[3:7] = quat_wxyz


        if True:
            height_command = self.vhcommand(current_pelvis_height_w = xyz[2])
        else :
            # IF you want to start with the standing stage on the initial period of the episode.
            if self.counter <= 100:
                height_command = np.array([0., 0.79])
            else:
                height_command = self.vhcommand(current_pelvis_height_w = xyz[2])

        world_quat = np.asarray((1., 0., 0., 0.))

        self.target_pose = np.concatenate([xyz[:2], np.array([height_command[1]]), world_quat])
        self.publish_target() # Just for visualization.

        # For standing.
        self.obs = self.obsmap(self.low_state, height_command)
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone()
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        target_dof_pos = self.actmap(self.action, self.obs)
        
        
        # FIXME(hh) If you want smoothing
        if self.config.later_smoothing:
            if self.prev_joint_pos_target is not None:
                if self.counter < 100:
                    target_dof_pos = self.config.initial_smoothing * target_dof_pos + \
                                    (1-self.config.initial_smoothing) * self.prev_joint_pos_target
                else:
                    target_dof_pos = self.config.later_smoothing * target_dof_pos + \
                                    (1-self.config.later_smoothing) * self.prev_joint_pos_target
            self.prev_joint_pos_target = target_dof_pos

        # Calculate metrics
        curr_q, curr_dq, curr_ddq, curr_tau = self.get_motor_state(self.low_state)
        # self.calculate_metrics(curr_q, curr_dq, curr_ddq, curr_tau, logpath)

        # FIXME(hh) 2nd smoothing, select only upper body joints
        # Build low cmd
        for i in self.mot_from_lab:
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0 * float(self.config.kps[i])
            self.low_cmd.motor_cmd[i].kd = 0.0 * float(self.config.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0 
            
        # send the command
        self.send_cmd(self.low_cmd)

    def run_wrapper(self):
        # print("hello", self.mode,
        # self.mode == Mode.zero_torque)
        if self.mode == Mode.wait:
            if self.low_state.crc != 0:
                self.mode = Mode.zero_torque
                self.low_cmd.mode_machine = self.mode_machine_
                print("Successfully connected to the robot.")
        elif self.mode == Mode.zero_torque:
            if self._mode_change:
                print("Enter zero torque state.")
                print("Waiting for the start signal...")
                self._mode_change = False
            self.zero_torque_state()
        elif self.mode == Mode.default_pos:
            if self._mode_change:
                print("Moving to default pos.")
                self._mode_change = False
                self.prepare_default_pos()
            self.move_to_default_pos()
        elif self.mode == Mode.damping:
            if self._mode_change:
                print("Enter default pos state.")
                print("Waiting for the Button A signal...")
                self._mode_change = False
            self.default_pos_state()
        elif self.mode == Mode.policy:
            if self._mode_change:
                print("Run policy.")
                self._mode_change = False
                self.counter = 0
            self.run_policy()
        elif self.mode == Mode.null:
            self._terminate = True

        # time.sleep(self.config.control_dt)


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
