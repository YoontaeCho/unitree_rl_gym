import os
import torch as th
import numpy as np
import math_utils
from pathlib import Path
from typing import Union, Literal
from legged_gym import LEGGED_GYM_ROOT_DIR
from scipy.spatial.transform import Rotation as R

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo
from rclpy.duration import Duration

from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from pyroki_ros.action import TrajOptSingleEE

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped
from common.command_helper_ros import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.remote_controller import RemoteController, KeyMap
from config_0820 import Config
from common.crc import CRC
from enum import Enum
from ikctrl import IKCtrl
from std_msgs.msg import Bool, Empty

import utils_robot as ur
import utils_trajcmd as ut
import utils_stage as us
import utils_eetrack_tag_contact_align as ue
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray
import subprocess

class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    finish = 5
    null = 6

yaw_quat = math_utils.as_np(math_utils.yaw_quat)
quat_apply = math_utils.as_np(math_utils.quat_apply)
axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)


axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)
quat_from_euler_xyz = math_utils.as_np(math_utils.quat_from_euler_xyz)
euler_xyz_from_quat = math_utils.as_np(math_utils.euler_xyz_from_quat)
yaw_quat = math_utils.as_np(math_utils.yaw_quat)
matrix_from_quat = math_utils.as_np(math_utils.matrix_from_quat)
subtract_frame_transforms = math_utils.as_np(math_utils.subtract_frame_transforms)
quat_apply_inverse = math_utils.as_np(math_utils.quat_apply_inverse)

class GlobalClock:
    def __init__(self, node):
        self.node = node

    def get_time(self):
        return self.node.get_clock().now()


clock = None

from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import TransformException

_last_tf = None  # module/class-level cache

def map_transform(t, rot_type="quat"):
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

    
def body_pose(
        tf_buffer,
        frame: str,
        ref_frame: str = 'pelvis',
        stamp=None,
        rot_type: str = 'axa',
        return_stamp=False,):
    """ --> tf does not exist """
    # global _last_tf
    if stamp is None:
        stamp = rp.time.Time()
        # stamp = clock.get_time()
    try:
        # t = "ref{=pelvis}_from_frame" transform
        t = tf_buffer.lookup_transform(
            ref_frame,  # to
            frame,  # from
            stamp)
        # _last_tf = t
    except TransformException as ex:
        print(f'Could not transform {frame} to {ref_frame}: {ex}')
        # return map_transform(_last_tf, rot_type)

    txn = t.transform.translation
    rxn = t.transform.rotation

    xyz = np.array([txn.x, txn.y, txn.z])
    quat_wxyz = np.array([rxn.w, rxn.x, rxn.y, rxn.z])

    xyz = np.array(xyz)
    if rot_type == 'axa':
        axa = axis_angle_from_quat(quat_wxyz)
        axa = wrap_to_pi(axa)
        if return_stamp:
            return (xyz, axa, t.header.stamp)
        else:
            return (xyz, axa)
    elif rot_type == 'quat':
        if return_stamp:
            return (xyz, quat_wxyz, t.header.stamp)
        else:
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
    

def interpolate_array_by_float_index(array, index):
    # Get the lower and upper bounds
    left_idx = int(np.floor(index))
    right_idx = int(np.ceil(index))
    
    # Get the values at the left and right indices
    left_value = array[left_idx]
    right_value = array[right_idx]
    
    # Calculate the fractional distance between the left and right indices
    fraction = index - left_idx
    
    # Linear interpolation formula
    interpolated_value = left_value + (right_value - left_value) * fraction
    
    return interpolated_value


def interpolate_position(pos1, pos2, n_segments):
    increments = (pos2 - pos1) / n_segments
    interp_pos = [pos1 + increments * p for p in range(n_segments)]
    interp_pos.append(pos2)
    return interp_pos


def interpolate_quaternion(quat1, quat2, n_segments):
    quat1 = th.from_numpy(quat1[None, None, ...])
    quat2 = quat2[None, ...]
    t = th.linspace(0, 1, n_segments + 1).view(1, -1, 1)
    interp_q = math_utils.slerp_vectorized(quat1, quat2, t)
    interp_q = interp_q[0]
    return interp_q


from common.rotation_helper import get_gravity_orientation, transform_imu_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        
        # Load policy

        # smoothing
        self.prev_joint_pos_target = None
        
        # log path
        self.logpath = Path('/tmp/e2e/')
        self.logpath.mkdir(parents=True, exist_ok=True)

        # == build index map ==
        self.mot_from_lab = index_map(self.config.motor_joint, self.config.lab_joint)
        self.lab_from_mot = index_map(self.config.lab_joint, self.config.motor_joint)
        
        # num joints
        self.num_joints = len(self.config.motor_joint)

        self.tasks = np.zeros((0,1))
        
        # log trajectory (1000Hz)
        self.timestamp_high_freq = np.array([])
        self.q_traj = np.zeros((0, self.num_joints))
        self.dq_traj = np.zeros((0, self.num_joints))
        self.tau_traj = np.zeros((0, self.num_joints))

        # log trajectory (50Hz)
        self.timestamp_low_freq = np.array([])
        self.locomotion_observations = np.zeros((0, self.config.locomotion_obs_dim)) # TODO
        # self.eetrack_observations = np.zeros((0, self.config.eetrack_obs_dim))
        self.locomotion_vel_traj = np.zeros((0, 3))

        # self.locomotion_actions = np.zeros((0, 15))
        self.locomotion_actions = np.zeros((0, 14))
        self.raw_joint_pos_targets = np.zeros((0, self.num_joints))
        self.joint_pos_targets = np.zeros((0, self.num_joints))

        self.current_joint_pos = np.zeros(self.num_joints)
        self.root_states_w = np.zeros((0, 7))

        # counter
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer(cache_time=rp.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self._node)

        self.tf_broadcaster = TransformBroadcaster(self._node)

        ########################## Locomotion ##########################
        self.locomotion_robot = ur.Robot(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0_replace_with_welder.urdf')
        # self.locomotion_actmap = ul.LocomotionAction_0820(config, self.locomotion_robot)
        self.locomotion_actmap = ut.TrajCmdAction(config, self.locomotion_robot)
        self.locomotion_policy = th.jit.load(config.locomotion_policy_path) # TODO
        self.locomotion_policy.eval()
        self.locomotion_obsmap = ut.TrajCmdObservation(config, self.tf_buffer # TODO
        )
        self.locomotion_velocity_command = None
        self.locomotion_phase_command = None
        self.locomotion_last_action = np.zeros(29)

        self.stop_locomotion = False

        self.act_joint = config.ik_joint
        self.ikctrl = IKCtrl('../../resources/robots/g1_description/g1_29dof_rev_1_0_zed2i_with_welder_v3.urdf',
                             self.act_joint,
                             frame='end_effector')
        self.lim_lo_pin = self.ikctrl.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.ikctrl.robot.model.upperPositionLimit
        
        # == build index map ==
        self.pin_from_mot = np.zeros(29, dtype=np.int32) # FIXME(ycho): hardcoded
        self.mot_from_pin = np.zeros(43, dtype=np.int32) # FIXME(ycho): hardcoded
        self.mot_from_act = np.zeros(len(self.act_joint), dtype=np.int32) # FIXME(ycho): hardcoded
        for i_mot, j in enumerate( self.config.motor_joint):
            i_pin = (self.ikctrl.robot.index(j) - 1)
            self.pin_from_mot[i_mot] = i_pin
            self.mot_from_pin[i_pin] = i_mot
            if j in self.act_joint:
                i_act = self.act_joint.index(j)
                self.mot_from_act[i_act] = i_mot

        q_mot = np.array(config.mot_joint_offsets)
        q_pin = np.zeros_like(self.ikctrl.cfg.q)
        q_pin[self.pin_from_mot] = q_mot

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

        print("Waiting for the robot to be ready...")
        self.mode = Mode.wait
        self.task : Literal["locomotion", "navigation", "sit", "eetrack"] = "locomotion"
        # self.task : Literal["locomotion", "navigation", "sit", "eetrack"] = "sit"
        self.prev_task = self.task
        self.task_counter = 0
        self._mode_change = True
        self._terminate = False

        # calls run_wrapper every self.config.control_dt seconds
        self._timer = self._node.create_timer(self.config.control_dt, self.run_wrapper)
        self.stop_time = None
        self.goal_changed = False

        try:
            rp.spin(self._node)
        except KeyboardInterrupt:
            self.log_metrics_and_trajectories()
            print("Log saved.")
        finally:
            subprocess.run(["pkill", "-f", "align"])
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
        return
        self.lowcmd_publisher_.publish(cmd)

    def zero_torque_state(self):
        if self.remote_controller.button[KeyMap.start] == 1:
            self._mode_change = True
            self.mode = Mode.damping
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
        if self.counter < self._num_step: # NOTE (bk) what's this if statement for?
            alpha = self.counter / self._num_step
            target_pos = np.zeros(self.num_joints)
            target_pos[self.mot_from_lab] = self.config.lab_joint_offsets
            for motor_idx in range(self.num_joints): 
                self.low_cmd.motor_cmd[motor_idx].q = (self._init_dof_pos[motor_idx] * (1 - alpha) + target_pos[motor_idx] * alpha)
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            self.counter += 1
        else:
            # NOTE (bk) why are we switching to policy mode here?
            self._mode_change = True
            self.mode = Mode.policy

    def default_pos_state(self):
        if self.remote_controller.button[KeyMap.Y] != 1:
            # NOTE (bk) what does this code snippet do? perhaps it sends the robot to default pos?
            # return
            target_pos = np.zeros(self.num_joints)
            target_pos[self.mot_from_lab] = self.config.lab_joint_offsets
            for motor_idx in range(self.num_joints):
                self.low_cmd.motor_cmd[motor_idx].q = target_pos[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
        else:
            self._mode_change = True
            self.mode = Mode.policy

    
    def publish_tf(self, header_frame, child_frame, pos, quat):
        t = TransformStamped()

        # Format header
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = header_frame #'world'
        t.child_frame_id = child_frame

        # Populate translation
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])

        # Set world_from_pelvis quaternion based on IMU state
        qw, qx, qy, qz = [float(x) for x in quat]
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def terminate_by_pelvis_condition(self, xyz, quat, limit_euler_angle=[1.7, 1.7]) -> bool:
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

        
    def run_locomotion_policy(self, traj_cmd):
        self.obs = self.locomotion_obsmap(self.low_state, traj_cmd, last_action=self.locomotion_last_action)
        obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone().float()
        self.locomotion_action = self.locomotion_policy(obs_tensor).detach().numpy().squeeze()
        self.locomotion_last_action = self.locomotion_action

        self.locomotion_target_dof_pos = self.locomotion_actmap(self.locomotion_action)
    
    def get_pelvis_from_world(self):
        if self.task in ["locomotion", "navigation"]:
            try:
                world_from_pelvis = body_pose(
                    self.tf_buffer,
                    'pelvis',
                    'world',
                    rot_type='quat',
                    # stamp=rp.time.Time()
                )
                # print(world_from_pelvis)
            except:
                try:
                    p_a, q_a = body_pose(
                        self.tf_buffer,
                        'camera_init',
                        'world',
                        rot_type='quat',
                        # stamp=rp.time.Time()
                    )
                except Exception as e:
                    print(e)
                try:
                    p_b, q_b = body_pose(
                        self.tf_buffer,
                        'body',
                        'camera_init',
                        rot_type='quat',
                        # stamp=rp.time.Time()
                    )
                except Exception as e:
                    print(e)
                try:
                    p_c, q_c = body_pose(
                        self.tf_buffer,
                        'body_z_from_mid_sole_link',
                        'body',
                        rot_type='quat',
                        # stamp=rp.time.Time()
                    )
                except Exception as e:
                    print(e)
                try:
                    p_d, q_d = body_pose(
                        self.tf_buffer,
                        'pelvis',
                        'body_z_from_mid_sole_link',
                        rot_type='quat',
                        # stamp=rp.time.Time()
                    )
                except Exception as e:
                    print(e)
                
                p_a_b, q_a_b = combine_frame_transforms(
                    p_a, q_a,
                    p_b, q_b
                )
                p_a_c, q_a_c = combine_frame_transforms(
                    p_a_b, q_a_b,
                    p_c, q_c
                )
                p_a_d, q_a_d = combine_frame_transforms(
                    p_a_c, q_a_c,
                    p_d, q_d
                )
                world_from_pelvis = (p_a_d, q_a_d)
                # print(world_from_pelvis)

        else:
            world_from_pelvis = body_pose(
                self.tf_buffer,
                'pelvis',
                'mid_sole_link',
                rot_type='quat',
                # stamp=rp.time.Time()
            )

        return world_from_pelvis
    

    def init_trajcmd(self, body_pos_w, body_quat_w, goal_pos_b, 
                     num_verts : int = 101, 
                     num_traj_samples: int = 10, 
                     traj_sample_time_step: float = 0.5
                     ):
        """
        args : 
        - start pos w
        - start quat w 
        - goal pos w 
        - num_verts = 101

        return :
        - None. Generates verts.
        """
        self.traj_init_counter = self.counter
        self.goal_changed = False
        self.num_verts = num_verts
        self.traj_duration = num_verts * 0.02
        self.num_segs = self.num_verts - 1
        self.num_traj_samples = num_traj_samples
        self.traj_sample_time_step = traj_sample_time_step

        body_pos_w = body_pos_w.reshape(1,3)
        body_quat_w = body_quat_w.reshape(1,4)
        goal_pos_b = goal_pos_b.reshape(1,3)


        goal_pos_w = body_pos_w + quat_apply(yaw_quat(body_quat_w), goal_pos_b)

        # t = th.linspace(0.0, 1.0, self.num_verts, device=self.device).view(self.num_verts, 1) # N_v x 1
        # start = body_pos_w[:, :2].view(1, 2).expand(num_verts, 2) # N_v x 2
        # goal = goal_pos_w[:, :2].view(1, 2).expand(num_verts, 2) # N_v x 2

        # self.verts = th.zeros((self.num_verts, 3), device=self.device) # N_v x 3
        # self.verts[:, :2] = (1.0 - t) * start + t * goal

        t = np.linspace(0.0, 1.0, self.num_verts).reshape(self.num_verts, 1)  # (N_v, 1)
        start = np.repeat(body_pos_w[:, :2], self.num_verts, axis=0)  # (N_v, 2)
        goal = np.repeat(goal_pos_w[:, :2], self.num_verts, axis=0)   # (N_v, 2)

        self.verts = np.zeros((self.num_verts, 3))
        self.verts[:, :2] = (1.0 - t) * start + t * goal

        self.traj_command_w = np.zeros((num_traj_samples, 3))
        self.traj_command_b = np.zeros_like(self.traj_command_w)


    def _get_pos_b(self, pos_w):
        # breakpoint()
        return quat_apply_inverse(
            yaw_quat(np.broadcast_to(self.root_state_w[3:], (pos_w.shape[0], 4))),
            pos_w
        )

    def _calc_traj_samples(self, times: np.ndarray) -> np.ndarray:
        """
        단일 env 기준: traj_id는 보통 0.
        times: (S,) 샘플 시각 (초)
        return: (S, 3) world 좌표 샘플
        """
        traj_dur = self.traj_duration
        num_verts = self.num_verts
        num_segs = self.num_segs

        # [0, 1] 로 정규화된 phase
        traj_phase = np.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs

        seg_id0 = np.floor(seg_idx).astype(np.int64)  # (S,)
        seg_id1 = np.ceil(seg_idx).astype(np.int64)   # (S,)

        # 다음 세그먼트 id를 저장(배열로 보관)
        self.next_segment_id = seg_id1

        # 선형 보간 계수
        lerp = (seg_idx - seg_id0)[:, None]  # (S,1)

        pos0 = self.verts[seg_id0]  # (S,3)
        pos1 = self.verts[seg_id1]  # (S,3)

        # 선형 보간
        pos = (1.0 - lerp) * pos0 + lerp * pos1  # (S,3)
        return pos

    def update_trajcmd(self):
        """
        단일 env 버전.
        num_traj_samples 개를, traj_sample_time_step 간격으로 현재 시점부터 샘플링.
        """

        # 현재 에피소드 경과 시간(예: 0.02s * counter)
        timestep_begin = float(self.counter - self.traj_init_counter) * 0.02

        # 샘플 시각들: [t0, t0+dt, t0+2dt, ...]
        steps = np.arange(self.num_traj_samples, dtype=float)
        traj_timesteps = timestep_begin + steps * self.traj_sample_time_step  # (S,)

        # 단일 env -> traj_id = 0
        traj_samples = self._calc_traj_samples(times=traj_timesteps)  # (S,3)

        # 결과 기록
        # traj_command_w, traj_command_b 는 (S,3) shape 라고 가정
        self.traj_command_w[...] = traj_samples
        self.traj_command_b[...] = self._get_pos_b(traj_samples)

        for i in range(self.num_traj_samples):
            # print(f"traj cmd sample {i}: w {self.traj_command_w[i]}, b {self.traj_command_b[i]}")
            self.publish_tf(
                header_frame='world',
                child_frame=f'target_{i}',
                pos=self.traj_command_w[i],
                quat=np.array([1.0, 0.0, 0.0, 0.0])
            )

        # 필요 시 반환
        return self.traj_command_b[...,:2].flatten()

    


    ############################# MAIN LOOP #############################
    def run_policy(self):
        # If the button A is pressed, then finish the policy.
        if self.remote_controller.button[KeyMap.A] == 1:
            self._mode_change = True
            self.mode = Mode.finish
            return

        self.counter += 1

        world_from_pelvis = self.get_pelvis_from_world()

        xyz, quat_wxyz = world_from_pelvis
        self.root_state_w = np.zeros(7)
        self.root_state_w[0:3] = xyz
        self.root_state_w[3:7] = quat_wxyz

        # Add termination condition.
        if self.terminate_by_pelvis_condition(xyz, quat_wxyz):
            raise ValueError("Terminated by pelvis condition.")
        

        if self.counter == 1:
            self.init_trajcmd(
                body_pos_w=xyz,
                body_quat_w=quat_wxyz,
                goal_pos_b = np.array([0.0, 0.0, 0.0]),
                num_verts = 101,
                num_traj_samples = 10,
                traj_sample_time_step = 0.5
            )

        ############################################ SWITCH MODE FROM LOCOMOTION TO NAVIVATION ############################################
        if self.remote_controller.button[KeyMap.Y] == 1:
            print("============== locomotion mode activated ==============")
            self.task = "locomotion" 
            # self.zed_stop_publisher.publish(Empty())
        if self.remote_controller.button[KeyMap.X] == 1:
            print("============== Go 1m forward ==============")
            self.goal_pos_b = np.array([1.0, 0.0, 0.0])  # 1m forward in body frame
            self.goal_changed = True
        if self.remote_controller.button[KeyMap.B] == 1:
            print("============== Go 1m backward ==============")
            self.goal_pos_b = np.array([-1.0, 0.0, 0.0])  # 1m backward in body frame
            self.goal_changed = True
        
        if self.goal_changed:
            print("New goal received.")
            self.init_trajcmd(
                body_pos_w=xyz,
                body_quat_w=quat_wxyz,
                goal_pos_b = self.goal_pos_b,
                num_verts = 101,
                num_traj_samples = 10,
                traj_sample_time_step = 0.5
            )

        ############################################ LOCOMOTION ############################################
        target_tau = np.zeros(29, dtype=np.float32)
        if self.task == "locomotion":
            trajcmd = self.update_trajcmd()
            self.run_locomotion_policy(trajcmd)
            target_dof_pos = self.locomotion_target_dof_pos


        kps = np.array(self.config.kps).astype(np.float32).copy()
        kds = np.array(self.config.kds).astype(np.float32).copy()

        for mot_idx in range(self.num_joints):
            self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].dq = 0.0
            self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(kps[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(kds[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].tau = float(target_tau[mot_idx])
        
        # observation dumping
        self.dump_observations_and_joint_pos_target()

         
        # send the command
        self.send_cmd(self.low_cmd)
        self.prev_task = self.task

    def dump_observations_and_joint_pos_target(self):
        # log timestamp
        timestamp_low_freq = clock.get_time().nanoseconds / 1e9
        self.timestamp_low_freq = np.append(self.timestamp_low_freq, timestamp_low_freq)

        self.root_states_w = np.vstack((self.root_states_w, self.root_state_w))
             
    def log_metrics_and_trajectories(self):     
        # Calculate pos diff & torque diff metrics
        pos_diff = np.average(np.abs(self.q_traj[1:] - self.q_traj[:-1]))
        torque_diff = np.average(np.abs(self.tau_traj[1:] - self.tau_traj[:-1]))
        print("\n--------------------------METRICS--------------------------")
        print("total_time", self.counter * self.config.control_dt)
        print("pos_diff", pos_diff)
        print("torque_diff", torque_diff)
        print("stop time ", self.stop_time)
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
            "root_states_w" : self.root_states_w,
        }
        log_data = {
            "metrics": metrics,
            "trajectories": trajectories
        }

        # Save the log with experiment name
        # sit_model = os.path.basename(self.config.sit_policy_path).split('.')[0]
        # eetrack_model = os.path.basename(self.config.locomotion_policy_path).split('.')[0]
        locomotion_model = os.path.basename(self.config.locomotion_policy_path).split('.')[0]
        timestamp = clock.get_time().nanoseconds / 1e9
        
        file_name = f"{self.logpath}/log_{locomotion_model}_{str(timestamp).split('.')[0]}.npy"
        np.save(file_name, log_data)
        print(f"Log saved at {file_name}")
        
        # totally terminate
        self._mode_change = True
        self.mode = Mode.null
        

    def run_wrapper(self):
        if self.mode == Mode.wait:
            print("Waiting for the robot to be ready...")
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
                print("Waiting for the Button Y signal...")
                self._mode_change = False
            self.default_pos_state()
        elif self.mode == Mode.policy:
            if self._mode_change:
                print("Run Policy.\n")
                print("--------------[ Basic Guidelines ]---------------")
                print("[Navigation] Press Button {X} to switch goal Forward 1m.")
                print("-------------------------------------------------")
                print("[Navigation] Press Button {B} to switch goal Backward -1m.")
                print("-------------------------------------------------")
                print("[Exit] Press Button {A} to finish.")
                print("-------------------------------------------------")

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
