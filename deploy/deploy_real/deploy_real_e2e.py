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
from ikctrl import IKCtrl

import utils_stage as us
import utils_eetrack as ue


class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    finish = 5
    null = 6


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


def body_pose(
        tf_buffer,
        frame: str,
        ref_frame: str = 'pelvis',
        stamp=None,
        rot_type: str = 'axa'):
    """ --> tf does not exist """
    if stamp is None:
        stamp = rp.time.Time()
        # stamp = clock.get_time()
    try:
        # t = "ref{=pelvis}_from_frame" transform
        t = tf_buffer.lookup_transform(
            ref_frame,  # to
            frame,  # from
            stamp)
    except TransformException as ex:
        print(f'Could not transform {frame} to {ref_frame}: {ex}')
        raise

    txn = t.transform.translation
    rxn = t.transform.rotation

    xyz = np.array([txn.x, txn.y, txn.z])
    quat_wxyz = np.array([rxn.w, rxn.x, rxn.y, rxn.z])

    xyz = np.array(xyz)
    if rot_type == 'axa':
        axa = axis_angle_from_quat(quat_wxyz)
        axa = wrap_to_pi(axa)
        return (xyz, axa)
    elif rot_type == 'quat':
        return (xyz, quat_wxyz)
    raise ValueError(f"Unknown rot_type: {rot_type}")


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
        
        # Load policy
        print(config.policy_path)
        self.sit_policy = th.jit.load(config.policy_path) # FIXME sit_policy_path
        self.sit_policy.eval()

        # smoothing
        self.prev_joint_pos_target = None
        
        # log path
        self.logpath = Path('/tmp/eetrack_stand/')
        self.logpath.mkdir(parents=True, exist_ok=True)

        # == build index map ==
        self.mot_from_lab = index_map(self.config.motor_joint, self.config.lab_joint)
        self.lab_from_mot = index_map(self.config.lab_joint, self.config.motor_joint)
        self.mot_from_lower = index_map(self.config.motor_joint, self.config.lower_joint)
        
        # num joints
        self.num_joints = len(self.config.motor_joint)
        
        # log trajectory (1000Hz)
        self.timestamp_high_freq = np.array([])
        self.q_traj = np.zeros((0, self.num_joints))
        self.dq_traj = np.zeros((0, self.num_joints))
        self.tau_traj = np.zeros((0, self.num_joints))

        # log trajectory (50Hz)
        self.timestamp_low_freq = np.array([])
        self.observations = np.zeros((0, self.config.obs_dim))
        self.actions = np.zeros((0, self.num_joints))
        self.raw_joint_pos_targets = np.zeros((0, self.num_joints))
        self.joint_pos_targets = np.zeros((0, self.num_joints))


        # counter
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.tf_broadcaster = TransformBroadcaster(self._node)

        ### Mapping helpers.
        self.sit_obsmap = us.Stage2Observation(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            config, self.tf_buffer)
        self.ikctrl = IKCtrl(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            config.arm_joint,
            frame='left_rubber_hand')
        self.sit_actmap = us.SimpleAction(config, self.ikctrl)
        self.vhcommand = us.VelocityHeightCommand(config)


        # eetrack
        self.eetrack_actmap = us.SimpleEETrackAction(config, self.ikctrl)
        self.eetrack_command = None
        self.eetrack_policy = None
        self.eetrack_obsmap = None

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

        self.mode = Mode.policy

        self.sitting = False
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
        
        # log trajectory
        # joint order = lab joint config order
        timestamp_high_freq = clock.get_time().nanoseconds / 1e9
        curr_q = np.zeros(self.num_joints)
        curr_dq = np.zeros(self.num_joints)
        curr_tau = np.zeros(self.num_joints)
        curr_q[self.lab_from_mot] = [self.low_state.motor_state[mot_idx].q for mot_idx in range(self.num_joints)]
        curr_dq[self.lab_from_mot] = [self.low_state.motor_state[mot_idx].dq for mot_idx in range(self.num_joints)]
        curr_tau[self.lab_from_mot] = [self.low_state.motor_state[mot_idx].tau_est for mot_idx in range(self.num_joints)]
        self.timestamp_high_freq = np.append(self.timestamp_high_freq, timestamp_high_freq)
        self.q_traj = np.vstack((self.q_traj, curr_q))
        self.dq_traj = np.vstack((self.dq_traj, curr_dq))
        self.tau_traj = np.vstack((self.tau_traj, curr_tau))

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.mode_machine = self.mode_machine_
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.publish(cmd)

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

        self._kps = [float(kp) for kp in self.config.kps]
        self._kds = [float(kd) for kd in self.config.kds]

        self._init_dof_pos = np.zeros(self.num_joints)
        for mot_idx in range(self.num_joints):
            self._init_dof_pos[mot_idx] = self.low_state.motor_state[mot_idx].q

    def move_to_default_pos(self):
        # move to default pos with smoothing
        if self.counter < self._num_step:
            alpha = self.counter / self._num_step
            for motor_idx in range(self.num_joints):
                target_pos = self.config.default_angles[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].q = (self._init_dof_pos[motor_idx] * (1 - alpha) + target_pos * alpha)
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            self.counter += 1
        else:
            self._mode_change = True
            self.mode = Mode.policy

    def default_pos_state(self):
        if self.remote_controller.button[KeyMap.A] != 1:
            for motor_idx in range(self.num_joints):
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
        else:
            self._mode_change = True
            self.mode = Mode.policy

        
    def terminate_by_pelvis_condition(self, xyz, quat, limit_euler_angle=[0.9, 1.0]) -> bool:
        """
        limit euler angle : roll 51.57', pitch 57.3'.
        """
        euler = math_utils.wrap_to_pi(
            th.stack(math_utils.euler_xyz_from_quat(th.as_tensor(quat.reshape(1, quat.shape[0]))), dim=-1)
        )
        out_of_limit = th.logical_or(
            th.abs(euler[..., 0]) > limit_euler_angle[0],
            th.abs(euler[..., 1]) > limit_euler_angle[1],
        )
        if out_of_limit.item() :
            print("Terminated by pelvis condition.")
            print(f"euler: {euler}")
        return out_of_limit.item()
    

    def run_policy(self):
        # If the button A is pressed, then finish the policy.
        if self.remote_controller.button[KeyMap.A] == 1:
            self._mode_change = True
            self.mode = Mode.finish
            return
        
        self.task = None


        self.counter += 1

        world_from_pelvis = body_pose(
            self.tf_buffer,
            'pelvis',
            'world',
            rot_type='quat'
        )
        xyz, quat_wxyz = world_from_pelvis
        root_state_w = np.zeros(7)
        root_state_w[0:3] = xyz
        root_state_w[3:7] = quat_wxyz
        
        # Add termination condition.
        if self.terminate_by_pelvis_condition(xyz, quat_wxyz):
            raise ValueError("Terminated by pelvis condition.")

        if self.task == "sit":
        # Press down button to sit
            curr_keymap = self.remote_controller.button[KeyMap.down] == 1
            if curr_keymap:
                self.sitting = True

            height_command = self.vhcommand(current_pelvis_height_w = xyz[2], sitting=self.sitting)

            # For stage 1 & 2.
            self.obs = self.sit_obsmap(self.low_state, height_command)

            obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
            obs_tensor = obs_tensor.detach().clone().float()
            self.action = self.sit_policy(obs_tensor).detach().numpy().squeeze()

            # target_dof_pos : motor joint ordered
            target_dof_pos = self.sit_actmap(self.action, self.obs)
            
        elif self.task == "eetrack":
            if self.eetrack_command is None:
                self.eetrack_command = ue.eetrack(th.from_numpy(root_state_w)[None],
                                   self.tf_buffer,
                                   clock,
                                   height=-0.4
                                   )
            hands_command = self.eetrack_command.get_command(
                th.from_numpy(root_state_w)[None])[0].detach().cpu().numpy()

            self.obs = self.eetrack_obsmap(self.low_state, hands_command)

            obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
            obs_tensor = obs_tensor.detach().clone().float()
            self.action = self.eetrack_policy(obs_tensor).detach().numpy().squeeze()

            # target_dof_pos : motor joint ordered
            target_dof_pos = self.eetrack_actmap(self.action, self.obs)

        else:
            raise ValueError("Unknown task")

        raw_target_dof_pos = target_dof_pos.copy()

        # smoothing for all joints
        if self.config.later_smoothing:
            if self.prev_joint_pos_target is not None:
                if self.counter < 100:
                    target_dof_pos = self.config.initial_smoothing * target_dof_pos + \
                                    (1-self.config.initial_smoothing) * self.prev_joint_pos_target
                else:
                    target_dof_pos = self.config.later_smoothing * target_dof_pos + \
                                    (1-self.config.later_smoothing) * self.prev_joint_pos_target
            self.prev_joint_pos_target = target_dof_pos

        # observation dumping
        self.dump_observations_and_joint_pos_target(raw_target_dof_pos, target_dof_pos)

        # FIXME(hh) kpkd coefficient smoothing
        # Build low cmd
        for mot_idx in range(self.num_joints):
            self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].dq = 0.0
            self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(self.config.kps[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(self.config.kds[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].tau = 0.0
        
        # send the command
        self.send_cmd(self.low_cmd)

    def dump_observations_and_joint_pos_target(self, raw_target_dof_pos, target_dof_pos):
        # log timestamp
        timestamp_low_freq = clock.get_time().nanoseconds / 1e9
        self.timestamp_low_freq = np.append(self.timestamp_low_freq, timestamp_low_freq)
        
        # log raw joint pos target (before smoothing)
        raw_target_dof_pos_lab = np.zeros(29)
        raw_target_dof_pos_lab[self.lab_from_mot] = raw_target_dof_pos
        self.raw_joint_pos_targets = np.vstack((self.raw_joint_pos_targets, raw_target_dof_pos_lab))
        
        # log joint pos target
        target_dof_pos_lab = np.zeros(29)
        target_dof_pos_lab[self.lab_from_mot] = target_dof_pos
        self.joint_pos_targets = np.vstack((self.joint_pos_targets, target_dof_pos_lab))
        
        # log observation
        self.observations = np.vstack((self.observations, self.obs))
        
        # log action
        self.actions = np.vstack((self.actions, self.action))
        
            
    def log_metrics_and_trajectories(self):     
        # Calculate pos diff & torque diff metrics
        pos_diff = np.average(np.abs(self.q_traj[1:] - self.q_traj[:-1]))
        torque_diff = np.average(np.abs(self.tau_traj[1:] - self.tau_traj[:-1]))
        print("\n--------------------------METRICS--------------------------")
        print("total_time", self.counter * self.config.control_dt)
        print("pos_diff", pos_diff)
        print("torque_diff", torque_diff)
        print("----------------------------------------------------------")
        
        # Normalize timestamps with respect to the first timestamp_high_freq
        self.timestamp_low_freq = self.timestamp_low_freq - self.timestamp_high_freq[0]
        self.timestamp_high_freq = self.timestamp_high_freq - self.timestamp_high_freq[0]
        
        # Save metrics and trajectories
        metrics = {
            "pos_diff": pos_diff,
            "torque_diff": torque_diff
        }
        trajectories = {
            "timestamp_high_freq": self.timestamp_high_freq,
            "q_traj": self.q_traj,
            "dq_traj": self.dq_traj,
            "tau_traj": self.tau_traj,
            "timestamp_low_freq": self.timestamp_low_freq,
            "observations": self.observations,
            "actions": self.actions,
            "raw_joint_pos_targets": self.raw_joint_pos_targets,
            "joint_pos_targets" : self.joint_pos_targets
        }
        log_data = {
            "metrics": metrics,
            "trajectories": trajectories
        }

        # Save the log with experiment name
        model = os.path.basename(self.config.policy_path).split('.')[0]
        setting = f"init_{str(self.config.initial_smoothing).replace('.', '_')}_"
        setting += f"later_{str(self.config.later_smoothing).replace('.', '_')}_"
        setting += f"kpkd_{str(self.config.kpkd_smoothing).replace('.', '_')}"
        
        exp = f"height_{str(self.config.target_height).replace('.', '_')}" 

        timestamp = clock.get_time().nanoseconds / 1e9
        
        file_name = f"{self.logpath}/log_{model}_{setting}_{exp}_{str(timestamp).split('.')[0]}.npy"
        np.save(file_name, log_data)
        print(f"Log saved at {file_name}")
        
        # totally terminate
        self._mode_change = True
        self.mode = Mode.null
        

    def run_wrapper(self):
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
                print("Press Button A to finish.")
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