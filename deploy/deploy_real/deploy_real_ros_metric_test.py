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
from config import Config
from common.crc import CRC
from enum import Enum
import pinocchio as pin
from ikctrl import IKCtrl, xyzw2wxyz
from yourdfpy import URDF

import math_utils
import random as rd
from act_to_dof import ActToDof


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

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Metric test
        self.prev_joint_pos_target = None
        self.smoothing = self.config.smoothing
        self.loaded_action = load_action(self.config.joint_pos_target_path)
        self.prev_q = None
        self.prev_dq = None
        self.prev_ddq = None
        self.prev_tau = None
        self.prev_prev_dq = None
        self._pos_diff = []
        self._pos_jitter = []
        self._torque_diff = []
        self.exp_name = os.path.basename(self.config.joint_pos_target_path)


        # == build index map ==
        self.mot_from_upper_body = index_map(self.config.motor_joint,
                                             self.config.upper_body_joint)
        self.mot_from_lower_body = index_map(self.config.motor_joint,
                                             self.config.lower_body_joint)
        self.mot_from_lab = index_map(self.config.motor_joint,
                                             self.config.lab_joint)
        self.lab_from_mot = index_map(self.config.lab_joint,
                                      self.config.motor_joint)
        self.config.default_angles = np.asarray(self.config.lab_joint_offsets)[
            self.lab_from_mot
        ]

        # Data buffers
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.tf_broadcaster = TransformBroadcaster(self._node)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type

            self.low_cmd = LowCmdHG()
            self.low_state = LowStateHG()

            self.lowcmd_publisher_ = self._node.create_publisher(LowCmdHG,
                                                                 'lowcmd', 10)
            self.lowstate_subscriber = self._node.create_subscription(
                LowStateHG, 'lowstate', self.LowStateHgHandler, 10)
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

        elif config.msg_type == "go":
            raise ValueError(f"{config.msg_type} is not implemented yet.")

        else:
            raise ValueError("Invalid msg_type")

        self.goalpath_publisher = self._node.create_publisher(
                PathMsg, 'goalpath', 10)
        self.truepath_publisher = self._node.create_publisher(
                PathMsg, 'truepath', 10)
        self.goalpath = PathMsg()
        self.truepath = PathMsg()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.mode = Mode.policy

        self._mode_change = True
        self._timer = self._node.create_timer(
            self.config.control_dt, self.run_wrapper)
        self._terminate = False
        try:
            rp.spin(self._node)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
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

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.mode_machine = self.mode_machine_
        cmd.crc = CRC().Crc(cmd)
        size = len(cmd.motor_cmd)
        self.lowcmd_publisher_.publish(cmd)

    def wait_for_low_state(self):
        while self.low_state.crc == 0:
            print(self.low_state)
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        if self.remote_controller.button[KeyMap.start] == 1:
            self._mode_change = True
            self.mode = Mode.default_pos
        else:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)

    def prepare_default_pos(self):
        # move time 2s
        total_time = 2
        self.counter = 0
        self._num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        self._kps = [float(kp) for kp in kps]
        self._kds = [float(kd) for kd in kds]
        self._default_pos = np.concatenate(
            (self.config.default_angles, self.config.arm_waist_target), axis=0)
        self._dof_size = len(dof_idx)
        self._dof_idx = dof_idx

        self._init_dof_pos = np.zeros(29)
        for i in range(29):
            self._init_dof_pos[i] = self.low_state.motor_state[i].q

    def move_to_default_pos(self):
        # move to default pos
        if self.counter < self._num_step:
            alpha = self.counter / self._num_step
            # FIXME(hh) only use upper-body
            for j in self.mot_from_upper_body:
                motor_idx = j
                target_pos = self.config.default_angles[j]

                self.low_cmd.motor_cmd[motor_idx].q = (
                    self._init_dof_pos[j] * (1 - alpha) + target_pos * alpha)
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0

            for i in self.mot_from_lower_body:
                self.low_cmd.motor_cmd[i].q = float(
                    0.0
                )
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.0
                self.low_cmd.motor_cmd[i].tau = 0.0
            self.send_cmd(self.low_cmd)
            self.counter += 1
        else:
            self._mode_change = True
            self.mode = Mode.damping

    def default_pos_state(self):
        # FIXME(hh) only use upper-body joints
        if self.remote_controller.button[KeyMap.A] != 1:
            for i in self.mot_from_upper_body:
                self.low_cmd.motor_cmd[i].q = float(
                    0.0
                )
                self.low_cmd.motor_cmd[i].kp = 40.0
                self.low_cmd.motor_cmd[i].kd = 5.0

            for i in self.mot_from_lower_body:
                self.low_cmd.motor_cmd[i].q = float(
                    0.0
                )
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.0
            self.send_cmd(self.low_cmd)
        else:
            self._mode_change = True
            self.mode = Mode.policy

    def get_motor_state(self, low_state):
        """
        get motor states of upper body joints
        """
        q_mot = []
        dq_mot = []
        ddq_mot = []
        tau_mot = []
        for i_mot in self.mot_from_upper_body:
            q_mot.append(low_state.motor_state[i_mot].q)
            dq_mot.append(low_state.motor_state[i_mot].dq)
            ddq_mot.append(low_state.motor_state[i_mot].ddq)
            tau_mot.append(low_state.motor_state[i_mot].tau_est)
        
        return np.asarray(q_mot), np.asarray(dq_mot), np.asarray(ddq_mot), np.asarray(tau_mot)
        # return np.asarray(q_mot), None, np.asarray(ddq_mot), np.asarray(tau_mot)


    def pos_diff(self, curr_q, prev_q):
        """calculate pos diff metric"""
        self._pos_diff.append(np.average(np.abs(curr_q - prev_q)))
    
    def torque_diff(self, curr_tau, prev_tau):
        """calculate torque diff metric"""
        self._torque_diff.append(np.average(np.abs(curr_tau - prev_tau)))
    
    def pos_jitter(self, curr_dq, prev_dq, prev_prev_dq):
        """calculate pos jitter metric"""
        self._pos_jitter.append(np.average(np.abs((curr_dq - prev_dq) - (prev_dq - prev_prev_dq))))


    def run_policy(self):
        logpath = Path('/tmp/metric_test/')
        logpath.mkdir(parents=True, exist_ok=True)

        if self.remote_controller.button[KeyMap.select] == 1:
            self._mode_change = True
            self.mode = Mode.null
            return
        self.counter += 1

        
        # FIXME(hh) If we have previously collected joint pos targets.

        target_dof_pos = self.loaded_action[self.counter][self.lab_from_mot]
        
        # FIXME(hh) If you want smoothing
        if self.smoothing:
            if self.prev_joint_pos_target is not None:
                target_dof_pos = self.smoothing * target_dof_pos + \
                                 (1-self.smoothing) * self.prev_joint_pos_target
            self.prev_joint_pos_target = target_dof_pos

        curr_q, curr_dq, curr_ddq, curr_tau = self.get_motor_state(self.low_state)
        if self.prev_prev_dq is not None:
            self.pos_diff(curr_q, self.prev_q)
            # print(curr_dq)
            # print(curr_q)
            self.pos_jitter(curr_dq, self.prev_dq, self.prev_prev_dq)
            self.torque_diff(curr_tau, self.prev_tau)
            # Save merics.
            np.save(F'{logpath}/metric_{self.exp_name}.npy',
                    np.array(
                        [np.average(self._pos_diff), 
                         np.average(self._pos_jitter), 
                         np.average(self._torque_diff)]
                         ))
        if self.counter == len(self.loaded_action) - 1:
            print("--------------------------METRICS--------------------------")
            print("pos_diff", np.average(self._pos_diff))
            print("pos_jitter", np.average(self._pos_jitter))
            print("torque_diff", np.average(self._torque_diff))
        self.prev_q, self.prev_dq, self.prev_ddq, self.prev_tau = curr_q, curr_dq, curr_ddq, curr_tau
        self.prev_prev_dq = self.prev_dq
        # FIXME(hh) 2nd smoothing, select only upper body joints
        # Build low cmd
        for i in self.mot_from_upper_body:
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 1.0 * float(self.config.kps[i])
            self.low_cmd.motor_cmd[i].kd = 1.0 * float(self.config.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0 # * float(target_dof_eff[i]) FIXME(hh) we were not applying effort.
        
        for i in self.mot_from_lower_body:
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
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
