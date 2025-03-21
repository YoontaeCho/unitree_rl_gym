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


from common.xml_helper import extract_link_data


def compute_com(tf_buffer, com_data, body_frames: List[str]):
    """compute com of body frames"""
    mass_list = []
    com_list = []

    # iterate for frames
    for frame in body_frames:
        try:
            frame_data = com_data[frame]
        except KeyError:
            continue

        try:
            link_pos, link_wxyz = body_pose(tf_buffer,
                                            frame, rot_type='quat')
        except TransformException:
            continue

        com_pos_b, com_wxyz = frame_data['pos'], frame_data['quat']

        # compute com from world coordinates
        # NOTE 'math_utils' package will be brought from isaaclab
        com_pos = link_pos + quat_rotate(link_wxyz, com_pos_b)
        com_list.append(com_pos)

        # get math
        mass = frame_data['mass']
        mass_list.append(mass)

    com = sum([m * pos for m, pos in zip(mass_list, com_list)]) / sum(mass_list)
    return com


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


def interpolate_position(pos1, pos2, n_segments):
    increments = (pos2 - pos1) / n_segments
    interp_pos = [pos1 + increments * p for p in range(n_segments)]
    interp_pos.append(pos2)
    return interp_pos


class eetrack:
    def __init__(self, root_state_w, tf_buffer):
        self.tf_buffer = tf_buffer
        # self.eetrack_midpt = root_state_w.clone()
        # self.eetrack_midpt[..., 1] += 0.3
        self.eetrack_midpt = (
            root_state_w[..., :3] +
            quat_rotate(root_state_w[0, 3:7].detach().cpu().numpy(),
                        np.array([0.3, 0.0, 0.0]))[None]
        )
        self.eetrack_end = None
        self.eetrack_subgoal = None
        self.number_of_subgoals = 60
        self.eetrack_line_length = 0.3
        self.device = "cpu"
        self.waypoints = self.create_eetrack(root_state_w)
        self.eetrack_subgoal = self.create_subgoal(
                root_state_w,
                self.waypoints)
        self.sg_idx = 0
        # first subgoal sampling time = 1.0s
        # self.init_time = rp.time.Time()#.nanoseconds / 1e9 + 1.0
        self.init_time = clock.get_time()

    def create_eetrack(self, root_state_w):
        self.eetrack_start = self.eetrack_midpt.clone()
        self.eetrack_end = self.eetrack_midpt.clone()
        is_hor = rd.choice([True, False])
        eetrack_offset = rd.uniform(-0.5, 0.5)
        # For testing
        is_box = True
        is_hor = True

        eetrack_offset = 0.0
        if is_box:
            waypoints = []

            dx = (self.eetrack_line_length) / 2.
            dy = (self.eetrack_line_length) / 2.

            deltas = [
                    [0, +dy, +dx + 0.1],
                    [0, +dy, -dx + 0.1],
                    [0, -dy, -dx + 0.1],
                    [0, -dy, +dx + 0.1],
                    [0, +dy, +dx + 0.1]
            ]

            for delta in deltas:
                waypoint = self.eetrack_midpt.clone()
                waypoint += math_utils.quat_rotate(
                    root_state_w[..., 3:7].float(),
                    th.as_tensor(delta, dtype=th.float32)[None]
                )
                waypoints.append( waypoint )
            return waypoints

        elif is_hor:
            dx = (self.eetrack_line_length) / 2.
            dz = eetrack_offset
            delta_body0 = [0, +dx, dz]
            delta_body1 = [0, -dx, dz]

            self.eetrack_start += math_utils.quat_rotate(
                root_state_w[..., 3:7].float(),
                th.as_tensor(delta_body0, dtype=th.float32)[None]
            )
            self.eetrack_end += math_utils.quat_rotate(
                root_state_w[..., 3:7].float(),
                th.as_tensor(delta_body1, dtype=th.float32)[None]
            )
            # self.eetrack_start[..., 2] += eetrack_offset
            # self.eetrack_end[..., 2] += eetrack_offset
            # self.eetrack_start[..., 0] -= (self.eetrack_line_length) / 2.
            # self.eetrack_end[..., 0] += (self.eetrack_line_length) / 2.
        else:
            # self.eetrack_start[..., 0] += eetrack_offset
            # self.eetrack_end[..., 0] += eetrack_offset
            # self.eetrack_start[..., 2] += (self.eetrack_line_length) / 2.
            # self.eetrack_end[..., 2] -= (self.eetrack_line_length) / 2.
            dx = eetrack_offset
            dz = (self.eetrack_line_length) / 2.
            delta_body0 = [0, dx, +dz]
            delta_body1 = [0, dx, -dz]
            self.eetrack_start += math_utils.quat_rotate(
                root_state_w[..., 3:7],
                th.as_tensor(delta_body0)[None]
            )
            self.eetrack_end += math_utils.quat_rotate(
                root_state_w[..., 3:7],
                th.as_tensor(delta_body1)[None]
            )

        return self.eetrack_start, self.eetrack_end

    def create_direction(self):
        angle_from_eetrack_line = torch.rand(1, device=self.device) * np.pi
        angle_from_xyplane_in_global_frame = torch.rand(
            1, device=self.device) * np.pi - np.pi / 2
        # For testing
        angle_from_eetrack_line = torch.rand(1, device=self.device) * np.pi / 2
        angle_from_xyplane_in_global_frame = torch.rand(
            1, device=self.device) * 0
        roll = torch.zeros(1, device=self.device)
        pitch = angle_from_xyplane_in_global_frame
        yaw = angle_from_eetrack_line
        euler = torch.stack([roll, pitch, yaw], dim=1)
        quat = math_utils.quat_from_euler_xyz(
            euler[:, 0], euler[:, 1], euler[:, 2])
        return quat

    def create_subgoal(self, root_state_w, waypoints):
        qs = []
        for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
            eetrack_subgoals = interpolate_position(
                p0, p1, self.number_of_subgoals)
            eetrack_subgoals = [
                (
                    l.clone().to(self.device, dtype=torch.float32)
                    if isinstance(l, torch.Tensor)
                    else torch.tensor(l, device=self.device, dtype=torch.float32)
                )
                for l in eetrack_subgoals
            ]
            eetrack_subgoals = torch.stack(eetrack_subgoals, axis=1)

            eetrack_ori = self.create_direction().unsqueeze(
                1).repeat(1, self.number_of_subgoals + 1, 1)
            if True:
                eetrack_ori[..., :] = root_state_w[..., None, 3:7]
            # welidng_subgoals -> Nenv x Npoints x (3 + 4)
            q = torch.cat([eetrack_subgoals, eetrack_ori], dim=2)
            qs.append(q)
        return torch.cat(qs, dim=1)

    def update_command(self):
        # print(rp.time.Time().nanoseconds)
        time = (clock.get_time() - self.init_time).nanoseconds / 1e9
        if (time >= 1.0):
            self.sg_idx = int((time - 1) / 0.1 + 1)
        print(time, self.sg_idx)
        # self.sg_idx.clamp_(0, self.number_of_subgoals + 1)
        self.sg_idx = min(
                self.sg_idx,
                self.eetrack_subgoal.shape[-2] - 1)
        self.next_command_s_left = self.eetrack_subgoal[...,
                                                        self.sg_idx, :]

    def get_command(self, root_state_w):
        self.update_command()

        pos_hand_b_left, quat_hand_b_left = body_pose(
            self.tf_buffer,
            "left_rubber_hand",
            rot_type='quat'
        )

        lerp_command_w_left = self.next_command_s_left

        (lerp_command_b_left_pos,
         lerp_command_b_left_quat) = math_utils.subtract_frame_transforms(
            root_state_w[..., 0:3],
            root_state_w[..., 3:7],
            lerp_command_w_left[:, 0:3],
            lerp_command_w_left[:, 3:7],
        )

        # lerp_command_b_left = lerp_command_w_left

        pos_delta_b_left, rot_delta_b_left = math_utils.compute_pose_error(
            torch.from_numpy(pos_hand_b_left)[None],
            torch.from_numpy(quat_hand_b_left)[None],
            lerp_command_b_left_pos,
            lerp_command_b_left_quat,
        )
        axa_delta_b_left = math_utils.wrap_to_pi(rot_delta_b_left)

        hand_command = torch.cat((pos_delta_b_left, axa_delta_b_left), dim=-1)
        return hand_command
