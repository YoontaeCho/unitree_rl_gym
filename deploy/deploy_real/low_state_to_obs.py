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
from common.xml_helper import extract_link_data


axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)
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

class Observation:
    def __init__(self,
                 urdf_path: str,
                 config,
                 tf_buffer: Buffer):
        self.links = list(URDF.load(urdf_path).link_map.keys())
        self.config = config
        self.num_lab_joint = len(config.lab_joint)
        self.tf_buffer = tf_buffer
        self.lab_from_mot = index_map(config.lab_joint,
                                      config.motor_joint)
        # bring default values
        self.com_data = extract_link_data(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.xml')

    def __call__(self,
                 low_state: LowStateHG,
                 last_action: np.ndarray,
                 hands_command: np.ndarray
                 ):
        config=self.config
        lab_from_mot = self.lab_from_mot
        num_lab_joint = self.num_lab_joint
        # observation terms (order preserved)

        # FIXME(ycho): dummy value
        # base_lin_vel = np.zeros(3)
        ang_vel = np.array([low_state.imu_state.gyroscope],
                           dtype=np.float32)

        if True:
            # NOTE(ycho): requires running `fake_world_tf_pub.py`.
            world_from_pelvis = self.tf_buffer.lookup_transform(
                'world',
                'pelvis',
                rp.time.Time()
                # clock.get_time()
            )
            rxn = world_from_pelvis.transform.rotation
            quat = np.array([rxn.w, rxn.x, rxn.y, rxn.z])
        else:
            quat = low_state.imu_state.quaternion

        if config.imu_type == "torso":
            waist_yaw = low_state.motor_state[config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = low_state.motor_state[config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel)

        # NOTE(ycho): `ang_vel` is _probably_ in the pelvis frame,
        # since otherwise transform_imu_data() would be unnecessary for
        # `ang_vel`.
        base_ang_vel = ang_vel.squeeze(0)

        # TODO(ycho): check if the convention "q_base^{-1} @ g" holds.
        projected_gravity = get_gravity_orientation(quat)

        fp_l = body_pose(self.tf_buffer, 'left_ankle_roll_link')
        fp_r = body_pose(self.tf_buffer, 'right_ankle_roll_link')
        foot_pose = np.concatenate([fp_l[0], fp_r[0], fp_l[1], fp_r[1]])

        hp_l = body_pose(self.tf_buffer, 'left_rubber_hand')
        hp_r = body_pose(self.tf_buffer, 'right_rubber_hand')
        hand_pose = np.concatenate([hp_l[0], hp_r[0], hp_l[1], hp_r[1]])

        # FIXME(ycho): implement com_pos_wrt_pelvis
        projected_com = compute_com(self.tf_buffer, self.com_data, self.links)[..., :2]
        # projected_zmp = _ # IMPOSSIBLE

        # Map `low_state` to index-mapped joint_{pos,vel}
        joint_pos = np.zeros(num_lab_joint,
                             dtype=np.float32)
        joint_vel = np.zeros(num_lab_joint,
                             dtype=np.float32)
        joint_pos[lab_from_mot] = [low_state.motor_state[i_mot].q for i_mot in
                                   range(len(lab_from_mot))]
        joint_pos -= config.lab_joint_offsets
        joint_vel[lab_from_mot] = [low_state.motor_state[i_mot].dq for i_mot in
                                   range(len(lab_from_mot))]
        actions = last_action

        # Given as delta_pos {xyz,axa}; i.e. 6D vector
        # hands_command = self.eetrack.get_command()

        right_arm_com = compute_com(self.tf_buffer, self.com_data, [
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_pitch_link"
            "right_wrist_roll_link",
            "right_wrist_yaw_link"
        ])
        left_arm_com = compute_com(self.tf_buffer, self.com_data, [
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_pitch_link"
            "left_wrist_roll_link",
            "left_wrist_yaw_link"
        ])
        world_from_pelvis = self.tf_buffer.lookup_transform(
            'world',
            'pelvis',
            rp.time.Time()
            # clock.get_time()
        )
        pelvis_height = [world_from_pelvis.transform.translation.z]

        obs = [
            base_ang_vel,
            projected_gravity,
            foot_pose,
            hand_pose,
            projected_com,
            joint_pos,
            # 0.2 * joint_vel,
            joint_vel,
            actions,
            hands_command,
            right_arm_com,
            left_arm_com,
            pelvis_height
        ]
        # print([np.shape(o) for o in obs])
        return np.concatenate(obs, axis=-1)
