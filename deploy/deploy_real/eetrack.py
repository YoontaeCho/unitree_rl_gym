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


axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)

def interpolate_position(pos1, pos2, n_segments):
    increments = (pos2 - pos1) / n_segments
    interp_pos = [pos1 + increments * p for p in range(n_segments)]
    interp_pos.append(pos2)
    return interp_pos


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


class eetrack:
    def __init__(self, root_state_w, tf_buffer, clock, height:float):
        self._height = height
        self.clock=clock
        self.tf_buffer = tf_buffer
        # self.eetrack_midpt = root_state_w.clone()
        # self.eetrack_midpt[..., 1] += 0.3
        self.eetrack_midpt = (
            root_state_w[..., :3] +
            quat_rotate(root_state_w[0, 3:7].detach().cpu().numpy(),
                        # np.array([0.3, 0.0, 0.0]))[None]
                        np.array([0.35, 0.0, 0.0]))[None] # squatting
                        # np.array([0.4, 0.0, 0.0]))[None] # squatting
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
        self.init_time = self.clock.get_time()
        self._offset = None
        

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

            # dx = (self.eetrack_line_length) / 2 - 0.1
            # dx = -0.4
            dx=self._height
            # dx = 0.0
            dy = (self.eetrack_line_length) / 2.

            deltas = [
                    # [0, +dy, -dx + 0.2 ], # checkpoint?
                    [0, +dy, +dx ],
                    [0, -dy, +dx ],
                    # [0, +dy, -dx ],
                    # [0, -dy, -dx ],
                    # [0, -dy, +dx ],
                    [0, +dy, +dx ],
                    # [0, +dy, -dx + 0.2 ], # checkpoint?
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
        pitch = torch.ones(1, device=self.device) * 0 # angle_from_xyplane_in_global_frame
        yaw = torch.ones(1, device=self.device) * 0. # angle_from_eetrack_line
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

            if True:
                eetrack_ori = self.create_direction().unsqueeze(
                    1).repeat(1, self.number_of_subgoals + 1, 1)
                if True:
                    eetrack_ori[..., :] = root_state_w[..., None, 3:7]
            else:
                eetrack_ori = self.create_direction().unsqueeze(
                    1).repeat(1, self.number_of_subgoals + 1, 1).clone()
                eetrack_ori[..., :] = math_utils.quat_mul(
                    root_state_w[..., None, 3:7].expand_as(eetrack_ori),
                    eetrack_ori 
                )


            # welidng_subgoals -> Nenv x Npoints x (3 + 4)
            q = torch.cat([eetrack_subgoals, eetrack_ori], dim=2)
            qs.append(q)
        return torch.cat(qs, dim=1)

    def update_command(self):
        # print(rp.time.Time().nanoseconds)
        time = (self.clock.get_time() - self.init_time).nanoseconds / 1e9
        if (time >= 1.0):
            self.sg_idx = int((time - 1) / 0.1 + 1)
        print(time, self.sg_idx)
        # self.sg_idx.clamp_(0, self.number_of_subgoals + 1)
        if self._offset is None:
            self._offset = self.sg_idx
        # self.sg_idx = min(
        #         self.sg_idx - self._offset,
        #         self.eetrack_subgoal.shape[-2] - 1)
        # self.sg_idx %= self.eetrack_subgoal.shape[-2]
        iii = (self.sg_idx - self._offset) % self.eetrack_subgoal.shape[-2]
        self.next_command_s_left = self.eetrack_subgoal[..., iii, :]

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