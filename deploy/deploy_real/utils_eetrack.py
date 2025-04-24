from typing import Union, List
import numpy as np
import torch
import torch as th

import rclpy as rp


from tf2_ros import TransformException

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


def body_pose(
        tf_buffer,
        frame: str,
        ref_frame: str = 'pelvis',
        stamp=None,
        rot_type: str = 'axa'):
    """ --> tf does not exist """
    if stamp is None:
        stamp = rp.time.Time()
        # stamp = self.clock.get_time()
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


class Range:
    def __init__(self, 
                init_x_b,
                init_y_b,
                init_z_b,
                init_roll_b,
                init_pitch_b,
                init_yaw_b,
                dx_local, 
                dy_local, 
                dz_local
                ):
        self.init_x_b = init_x_b
        self.init_y_b = init_y_b
        self.init_z_b = init_z_b
        self.init_roll_b = init_roll_b
        self.init_pitch_b = init_pitch_b
        self.init_yaw_b = init_yaw_b
        self.dx_local = dx_local
        self.dy_local = dy_local
        self.dz_local = dz_local

class eetrack:
    def __init__(self, root_state_w, tf_buffer, clock, ranges : Range):
        self.clock = clock
        self.tf_buffer = tf_buffer
        
        self.eetrack_line_length = 0.1
        self.eetrack_vel = 0.01

        self.step_dt = 0.02
        self.dt_segment_length = self.eetrack_vel * self.step_dt # 0.0002
        self.non_first_subgoal_sampling_time = self.dt_segment_length / self.eetrack_vel
        self.number_of_subgoals = int(self.eetrack_line_length / self.dt_segment_length) # 0.1 / 0.0002 = 500
        
        self.device = "cpu"
        self.sg_idx = 0
        # first subgoal sampling time = 1.0s
        # self.init_time = rp.time.Time()#.nanoseconds / 1e9 + 1.0
        self.init_time = self.clock.get_time()
        self.init_root_state_w = root_state_w
        self.init_root_pos_w = root_state_w[:, :3]

        self.ranges = ranges
        self.init_eetrack_sampler()

        self.create_eetrack(root_state_w[:, -4:])
        self.eetrack_subgoal = self.create_subgoal()

        self.is_initial_goal = True


    def init_eetrack_sampler(self):
        min_eetrack_init_xyz_b = th.tensor(
            [self.ranges.init_x_b[0], self.ranges.init_y_b[0], self.ranges.init_z_b[0]], device=self.device
        )
        max_eetrack_init_xyz_b = th.tensor(
            [self.ranges.init_x_b[1], self.ranges.init_y_b[1], self.ranges.init_z_b[1]], device=self.device
        )
        ensure_not_same = min_eetrack_init_xyz_b == max_eetrack_init_xyz_b
        max_eetrack_init_xyz_b[ensure_not_same] += th.finfo(max_eetrack_init_xyz_b.dtype).eps
        self.eetrack_init_xyz_b_sampler = th.distributions.Uniform(
            low=min_eetrack_init_xyz_b,
            high=max_eetrack_init_xyz_b,
        )

        min_eetrack_init_rpy_b = (
            th.pi
            / 180
            * th.tensor([self.ranges.init_roll_b[0], self.ranges.init_pitch_b[0], self.ranges.init_yaw_b[0]], device=self.device)
        )
        max_eetrack_init_rpy_b = (
            th.pi
            / 180
            * th.tensor([self.ranges.init_roll_b[1], self.ranges.init_pitch_b[1], self.ranges.init_yaw_b[1]], device=self.device)
        )
        ensure_not_same = min_eetrack_init_rpy_b == max_eetrack_init_rpy_b
        max_eetrack_init_rpy_b[ensure_not_same] += th.finfo(max_eetrack_init_rpy_b.dtype).eps
        self.eetrack_init_rpy_b_sampler = th.distributions.Uniform(
            low=min_eetrack_init_rpy_b,
            high=max_eetrack_init_rpy_b,
        )

        min_eetrack_xyz_dir_local = th.tensor(
            [self.ranges.dx_local[0], self.ranges.dy_local[0], self.ranges.dz_local[0]], device=self.device
        )
        max_eetrack_xyz_dir_local = th.tensor(
            [self.ranges.dx_local[1], self.ranges.dy_local[1], self.ranges.dz_local[1]], device=self.device
        )
        ensure_not_same = min_eetrack_xyz_dir_local == max_eetrack_xyz_dir_local
        max_eetrack_xyz_dir_local[ensure_not_same] += th.finfo(max_eetrack_xyz_dir_local.dtype).eps
        self.eetrack_xyz_dir_local_sampler = th.distributions.Uniform(
            low=min_eetrack_xyz_dir_local,
            high=max_eetrack_xyz_dir_local,
        )

    def create_eetrack(self, root_state_w):

        euler = self.eetrack_init_rpy_b_sampler.sample((1,)).to(self.device)
        eetrack_quat_b = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

        eetrack_xyz_dir_local = self.eetrack_xyz_dir_local_sampler.sample((1,)).to(self.device)
        # Ensure the norm of direction is 1.
        eetrack_xyz_dir_local /= eetrack_xyz_dir_local.norm(dim=-1, keepdim=True)

        eetrack_xyz_dir_b = math_utils.quat_apply(eetrack_quat_b, eetrack_xyz_dir_local)

        eetrack_start_b = self.eetrack_init_xyz_b_sampler.sample((1,)).to(self.device)
        eetrack_end_b = eetrack_start_b + self.eetrack_line_length * eetrack_xyz_dir_b

        eetrack_start_b = eetrack_start_b.double()
        eetrack_end_b = eetrack_end_b.double()

        # Rotate the eetrack line (yaw) and add initial root position.
        self.eetrack_start_w = (
            math_utils.quat_apply_yaw(root_state_w, eetrack_start_b) + self.init_root_pos_w
        )
        self.eetrack_end_w = (
            math_utils.quat_apply_yaw(root_state_w, eetrack_end_b) + self.init_root_pos_w
        )
        self.eetrack_quat_w = math_utils.quat_mul(math_utils.yaw_quat(root_state_w), eetrack_quat_b)


    def create_subgoal(self):
        # initial hand pos
        pos_hand_w_left, quat_hand_w_left = body_pose(
            self.tf_buffer,
            frame="end_effector",
            ref_frame="world",
            rot_type='quat'
        )
        
        # initial hand pos -> eetrack start pos
        # breakpoint()
        to_eeline_subgoals = interpolate_position(
            torch.tensor(pos_hand_w_left).unsqueeze(0),
            self.eetrack_start_w,
            100
        )

        # eetrack start pos -> eetrack end pos
        on_eeline_subgoals = interpolate_position(
            self.eetrack_start_w,
            self.eetrack_end_w,
            self.number_of_subgoals,
        )

        eetrack_subgoals = to_eeline_subgoals + on_eeline_subgoals
        
        eetrack_subgoals = [
            (
                l.clone().to(self.device, dtype=th.float32)
                if isinstance(l, th.Tensor)
                else th.tensor(l, device=self.device, dtype=th.float32)
            )
            for l in eetrack_subgoals
        ]
        eetrack_subgoals = th.stack(eetrack_subgoals, axis=1)
        # eetrack_quat = self.eetrack_quat_w.unsqueeze(1).repeat(1, 101 + self.number_of_subgoals + 1, 1)
        eetrack_quat = torch.tensor(quat_hand_w_left).unsqueeze(0).unsqueeze(1).repeat(1, 101 + self.number_of_subgoals + 1, 1)

        return th.cat([eetrack_subgoals, eetrack_quat], dim=2)

    def update_command(self):
        """
        update command for eetrack
        initial_goal: True if this is the first command.
        initial_goal should be given as True or False by user.
        """
        if self.is_initial_goal:
            self.sg_idx = 0
            self.init_time = self.clock.get_time()
        else:
            # print(rp.time.Time().nanoseconds)
            time = (self.clock.get_time() - self.init_time).nanoseconds / 1e9
            if (time >= 1.0):
                # subgoal is updated on every 0.02s
                update_time = 0.02
                self.sg_idx = int((time - 1) / update_time + 1)
            # self.sg_idx.clamp_(0, self.number_of_subgoals + 1)
        # FIXME
        self.sg_idx = 0
        self.next_command_s_left = self.eetrack_subgoal[..., self.sg_idx, :]

    def get_command(self, root_state_w):
        self.update_command()

        pos_hand_b_left, quat_hand_b_left = body_pose(
            self.tf_buffer,
            "end_effector",
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
