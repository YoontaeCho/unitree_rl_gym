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

        self.config = config
        self.remote_controller = RemoteController()
        
        # Load policy
        print(config.policy_path)
        self.policy = torch.jit.load(config.policy_path)
        self.policy.eval()


        ### Mapping helpers.


        # smoothing
        self.prev_joint_pos_target = None
        # self.smoothing = self.config.smoothing
        
        # log path
        self.logpath = Path('/tmp/eetrack_sit/')
        self.logpath.mkdir(parents=True, exist_ok=True)
        
        # log trajectory
        self.timestamp = np.array([])
        self.q_traj = np.zeros((0, 29))
        self.dq_traj = np.zeros((0, 29))
        self.tau_traj = np.zeros((0, 29))

        # == build index map ==
        self.mot_from_lab = index_map(self.config.motor_joint,
                                             self.config.lab_joint)
        self.lab_from_mot = index_map(self.config.lab_joint,
                                      self.config.motor_joint)
        self.config.default_angles = np.asarray(self.config.lab_joint_offsets)[
            self.lab_from_mot
        ]
        self.mot_from_arm = index_map(self.config.motor_joint, self.config.arm_joints)

        # Data buffers
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)


        self.ikctrl = IKCtrl(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            config.arm_joint,
            frame='left_rubber_hand')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.tf_broadcaster = TransformBroadcaster(self._node)


        self.obsmap = us.Stage2Observation(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            config, self.tf_buffer)
        self.actmap = us.SimpleAction(config, self.ikctrl)
        self.vhcommand = us.VelocityHeightCommand(config)

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
        
        # log trajectory
        # joint order = lab joint config order
        timestamp = clock.get_time().nanoseconds / 1e9
        curr_q = []
        curr_dq = []
        curr_tau = []
        for i_mot in self.mot_from_lab:
            curr_q.append(self.low_state.motor_state[i_mot].q)
            curr_dq.append(self.low_state.motor_state[i_mot].dq)
            curr_tau.append(self.low_state.motor_state[i_mot].tau_est)
        self.timestamp = np.append(self.timestamp, timestamp)
        self.q_traj = np.vstack((self.q_traj, curr_q))
        self.dq_traj = np.vstack((self.dq_traj, curr_dq))
        self.tau_traj = np.vstack((self.tau_traj, curr_tau))

        self.terminate_by_joint_acc()

    def terminate_by_joint_acc(self):
        arm_dqs = self.dq_traj[:, self.mot_from_arm]
        is_terminate = np.abs(arm_dqs[-1,:] - arm_dqs[-2,:]) > 2.
        print("is_terminate : ",is_terminate)
        if is_terminate:
            raise ValueError("Terminate by joint acc")
    
    def run_policy(self):
        if self.remote_controller.button[KeyMap.A] == 1:
            self._mode_change = True
            self.mode = Mode.finish
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
        obs_tensor = obs_tensor.detach().clone().float()
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
        # curr_q, curr_dq, curr_ddq, curr_tau = self.get_motor_state(self.low_state)
        # self.calculate_metrics(curr_q, curr_dq, curr_ddq, curr_tau, logpath)

        # FIXME(hh) 2nd smoothing, select only upper body joints
        # Build low cmd
        for i in range(len(self.mot_from_lab)):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = self.config.kpkd_smoothing * float(self.config.kps[i])
            self.low_cmd.motor_cmd[i].kd = self.config.kpkd_smoothing * float(self.config.kds[i])
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
        elif self.mode == Mode.finish:
            if self._mode_change:
                print("Finish.")
                self._mode_change = False
            self.log_metrics_and_trajectories()
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
