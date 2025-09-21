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
from std_msgs.msg import Bool

import utils_robot as ur
import utils_locomotion as ul
import utils_stage as us
import utils_eetrack_tag as ue
from scipy.spatial.transform import Rotation as R
from collections import deque

from std_msgs.msg import Float64MultiArray

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
        rot_type: str = 'axa'):
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
        t = TransformStamped()

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

def moving_average(x, window_size):
    x = np.stack(x)
    window_size = min(len(x), window_size)
    kernel = np.ones((window_size,)) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=0, arr=x)[0]


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        
        # num joints
        self.num_joints = len(self.config.motor_joint)

        # counter
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        # self.tf_listener = TransformListener(self.tf_buffer, self._node, spin_thread=True)

        self.tf_broadcaster = TransformBroadcaster(self._node)

        self.calib_dof_pos = np.array(
            [0.01986018,  0.0104822,   0.0023168,   0.01389957, -0.00005328,  0.0004273,
             0.02347599,  0.00330658,  0.00467377,  0.02251543, -0.00389107, -0.0011151,
            -0.00116509, -0.08396117,  0.00196204,  0.15910257,  0.05286242, -0.00417051,
             0.93385875, -0.00079096,  0.00742447,  0.00111214, -0.33684063, -0.44240966,
            -0.44957623,  0.00799348,  0.40574992,  0.54905778,  0.7704879]
        )

        self.zed_optical_frame = "zed2i_left_camera_optical_frame"
        self.zed_optical_frame_calib = "zed2i_left_camera_optical_frame_calib"
        self.tag_ids = [10, 11, 12]
        self.tag_id_to_offset = [
            [[0.04, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[-0.04, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ]
        self.tag_mean_frame = "tag_mean"

        self.tag_mean_pos_cam_deque = deque(maxlen=200)
        self.tag_mean_rpy_cam_deque = deque(maxlen=200)
        self.tag_mean_inv_pos_cam_deque = deque(maxlen=200)
        self.tag_mean_inv_rpy_cam_deque = deque(maxlen=200)


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

        self._mode_change = True
        self._terminate = False

        # calls run_wrapper every self.config.control_dt seconds
        self._timer = self._node.create_timer(self.config.control_dt, self.run_wrapper)
        self.stop_time = None
        try:
            rp.spin(self._node)
        except KeyboardInterrupt:
            pass
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
            # NOTE (bk) why are we switching to policy mode here?
            self._mode_change = True
            self.mode = Mode.policy

    def default_pos_state(self):
        if self.remote_controller.button[KeyMap.R2] != 1:
            # NOTE (bk) what does this code snippet do? perhaps it sends the robot to default pos?
            for motor_idx in range(self.num_joints):
                q = self.low_state.motor_state[motor_idx].q
                self.low_cmd.motor_cmd[motor_idx].q = (
                    np.clip(self.config.locomotion_motor_joint_offsets[motor_idx] - q, -0.1, 0.1) + q
                )
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
        else:
            self._mode_change = True
            self.mode = Mode.policy

    def publish_tag_mean_frame(self, pos, rpy):
        t = TransformStamped()

        # Format header
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = self.zed_optical_frame
        t.child_frame_id = self.tag_mean_frame

        quat = quat_from_euler_xyz(np.array(rpy[0]), np.array(rpy[1]), np.array(rpy[2]))

        # Populate translation
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]

        t.transform.rotation.w = quat[0]
        t.transform.rotation.x = quat[1]
        t.transform.rotation.y = quat[2]
        t.transform.rotation.z = quat[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)
        

    def publish_calib_frame(self, pos, rpy, parent_frame):
        t = TransformStamped()

        # Format header
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = self.zed_optical_frame_calib

        quat = quat_from_euler_xyz(np.array(rpy[0]), np.array(rpy[1]), np.array(rpy[2]))

        # Populate translation
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]

        t.transform.rotation.w = quat[0]
        t.transform.rotation.x = quat[1]
        t.transform.rotation.y = quat[2]
        t.transform.rotation.z = quat[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)


    def run_policy(self):
        ############################# MAIN LOOP #############################
        # If the button A is pressed, then finish the policy.
        if self.remote_controller.button[KeyMap.A] == 1:
            self._mode_change = True
            self.mode = Mode.finish
            return

        self.counter += 1

        qj = np.array([self.low_state.motor_state[mot_idx].q for mot_idx in range(self.num_joints)])
        
        target_dof_pos = (self.calib_dof_pos - qj).clip(-0.03, 0.03) + qj

        tags_cam_opt = [
            body_pose(self.tf_buffer, f"tag_{tag_id}_from_opt_frame", self.zed_optical_frame, rot_type="quat")
            for tag_id in self.tag_ids
        ]
        opt_tag_center_cam_pos_list = []
        opt_tag_center_cam_rpy_list = []
        opt_tag_center_inv_cam_pos_list = []
        opt_tag_center_inv_cam_rpy_list = []
        for tag_cam_opt, offset in zip(tags_cam_opt, self.tag_id_to_offset):
            tag_center_pos, tag_center_quat = combine_frame_transforms(
                tag_cam_opt[0],
                tag_cam_opt[1],
                np.array(offset[0]),
                np.array(offset[1]),
            )
            tag_center_inv_pos, tag_center_inv_quat = subtract_frame_transforms(tag_center_pos, tag_center_quat)
            opt_tag_center_cam_pos_list.append(tag_center_pos)
            opt_tag_center_cam_rpy_list.append(np.concatenate(euler_xyz_from_quat(tag_center_quat[None])))
            opt_tag_center_inv_cam_pos_list.append(tag_center_inv_pos)
            opt_tag_center_inv_cam_rpy_list.append(np.concatenate(euler_xyz_from_quat(tag_center_inv_quat[None])))

        opt_tag_center_cam_pos_mean = np.mean(opt_tag_center_cam_pos_list, axis=0)
        opt_tag_center_cam_rpy_mean = np.mean(opt_tag_center_cam_rpy_list, axis=0)
        opt_tag_center_inv_cam_pos_mean = np.mean(opt_tag_center_inv_cam_pos_list, axis=0)
        opt_tag_center_inv_cam_rpy_mean = np.mean(opt_tag_center_inv_cam_rpy_list, axis=0)

        # Mean
        self.publish_tag_mean_frame(opt_tag_center_cam_pos_mean, opt_tag_center_cam_rpy_mean)
        self.publish_calib_frame(opt_tag_center_inv_cam_pos_mean, opt_tag_center_inv_cam_rpy_mean, parent_frame=self.tag_mean_frame)

        # Moving average
        # self.tag_mean_pos_cam_deque.append(opt_tag_center_cam_pos_mean)
        # self.tag_mean_rpy_cam_deque.append(opt_tag_center_cam_rpy_mean)
        # self.tag_mean_inv_pos_cam_deque.append(opt_tag_center_inv_cam_pos_mean)
        # self.tag_mean_inv_rpy_cam_deque.append(opt_tag_center_inv_cam_rpy_mean)

        # opt_tag_center_cam_pos_ma = moving_average(self.tag_mean_pos_cam_deque, 200)
        # opt_tag_center_cam_rpy_ma = moving_average(self.tag_mean_rpy_cam_deque, 200)
        # opt_tag_center_inv_cam_pos_ma = moving_average(self.tag_mean_inv_pos_cam_deque, 200)
        # opt_tag_center_inv_cam_rpy_ma = moving_average(self.tag_mean_inv_rpy_cam_deque, 200)

        # self.publish_tag_mean_frame(opt_tag_center_cam_pos_ma, opt_tag_center_cam_rpy_ma)
        # self.publish_calib_frame(opt_tag_center_inv_cam_pos_ma, opt_tag_center_inv_cam_rpy_ma, parent_frame=self.tag_mean_frame)

        cam_opt_calib_parent_pos, cam_opt_calib_parent_quat = body_pose(
            self.tf_buffer,
            self.zed_optical_frame_calib,
            "zed2i_left_camera_frame",
            rot_type="quat",
        )
        print("Calib translation from parent:", cam_opt_calib_parent_pos)
        print("Calib rpy from parent:", np.concatenate(euler_xyz_from_quat(cam_opt_calib_parent_quat[None])))

        kps = np.array(self.config.kps).astype(np.float32).copy()
        kds = np.array(self.config.kds).astype(np.float32).copy()
        kps[-7:] = self.config.eetrack_right_arm_kps
        kds[-7:] = self.config.eetrack_right_arm_kds

        for mot_idx in range(self.num_joints):
            self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].dq = 0.0
            self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(kps[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(kds[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].tau = 0.0
         
        # send the command
        self.send_cmd(self.low_cmd)


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
                print("Waiting for the Button R2 signal...")
                self._mode_change = False
            self.default_pos_state()
        elif self.mode == Mode.policy:
            if self._mode_change:
                print("--------------Running Calibration----------------.\n")
                self._mode_change = False
                self.counter = 0
            self.run_policy()
        elif self.mode == Mode.finish:
            if self._mode_change:
                print("Finish.")
                self._mode_change = False
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
