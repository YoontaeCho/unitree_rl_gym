import time
import mujoco
import numpy as np
import random as rd

DEBUG = True
from math_utils import (
    as_np,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate_inverse,
    axis_angle_from_quat,
    wrap_to_pi,
    subtract_frame_transforms,
    quat_rotate,
    quat_apply_yaw,
    quat_from_euler_xyz,
    compute_pose_error
)
quat_from_angle_axis = as_np(quat_from_angle_axis)
quat_mul = as_np(quat_mul)
quat_rotate_inverse = as_np(quat_rotate_inverse)
axis_angle_from_quat = as_np(axis_angle_from_quat)
wrap_to_pi = as_np(wrap_to_pi)
subtract_frame_transforms = as_np(subtract_frame_transforms)
quat_rotate = as_np(quat_rotate)
quat_apply_yaw = as_np(quat_apply_yaw)
quat_from_euler_xyz = as_np(quat_from_euler_xyz)
compute_pose_error = as_np(compute_pose_error)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2

    See https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space

    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-6:  # If vectors are nearly parallel, return identity matrix
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


class EETrack:
    def __init__(self, root_state_w):
        self.eetrack_midpt = root_state_w[0:3]
        self.number_of_subgoals = 30
        self.eetrack_line_length = 0.3
        self.create_eetrack(root_state_w[3:7])
        self.eetrack_subgoal = self.create_subgoals(root_state_w[3:7])
        self.sg_idx = 0
        self.init_time = time.time()

    def interpolate_position(self, pos1, pos2, n_segments):
        increments = (pos2 - pos1) / n_segments
        interp_pos = np.array([pos1 + increments * p for p in range(n_segments)])
        interp_pos = np.vstack([interp_pos, pos2])
        return interp_pos

    def create_eetrack(self, root_quat):
        is_hor = rd.choice([True, False])
        eetrack_offset = rd.random() - 0.5
        eetrack_half_len = self.eetrack_line_length / 2.

        if DEBUG :
            is_hor = True
            eetrack_offset = 0.0

        hor = is_hor
        ver = (not is_hor)

        start_y = hor * eetrack_half_len + ver * eetrack_offset
        start_z = hor * eetrack_offset + ver * eetrack_half_len
        end_y = hor * -eetrack_half_len + ver * eetrack_offset
        end_z = hor * eetrack_offset + ver * -eetrack_half_len

        self.eetrack_start = np.zeros(3)
        self.eetrack_end = np.zeros(3)
        self.eetrack_start[0] = 0.3
        self.eetrack_start[1] = start_y
        self.eetrack_start[2] = start_z
        self.eetrack_end[0] = 0.3
        self.eetrack_end[1] = end_y
        self.eetrack_end[2] = end_z

        # Rotate the eetrack line (yaw)
        self.eetrack_start = quat_apply_yaw(
            root_quat, self.eetrack_start
        ) + self.eetrack_midpt
        self.eetrack_end= quat_apply_yaw(
            root_quat, self.eetrack_end
        ) + self.eetrack_midpt

    def create_subgoals(self, root_quat):
        eetrack_subgoal = self.interpolate_position(
            self.eetrack_start, self.eetrack_end, self.number_of_subgoals
        )
        eetrack_ori = self.create_direction(root_quat)
        eetrack_ori = np.expand_dims(eetrack_ori, axis=0)  # shape: (1, 3)
        eetrack_ori = np.repeat(eetrack_ori, self.number_of_subgoals + 1, axis=0)
        return np.hstack([eetrack_subgoal, eetrack_ori])

    def create_direction(self, root_quat):
        angle_from_eetrack_line = rd.random() * np.pi
        angle_from_xy_plane_in_global_frame = rd.random() * np.pi - np.pi / 2
        angle_from_eetrack_line = np.pi / 2
        if DEBUG:
            angle_from_xy_plane_in_global_frame = 0
        roll = 0
        pitch = angle_from_xy_plane_in_global_frame
        yaw = angle_from_eetrack_line - np.pi / 2
        quat = quat_from_euler_xyz(
            np.asarray(roll),
            np.asarray(pitch),
            np.asarray(yaw)
        )
        return quat_mul(root_quat, quat)

    def update_command(self):
        t = time.time() - self.init_time
        if t >= 1.0:
            self.sg_idx = (t - 1.0)/0.1 + 1
        self.sg_idx = int(min(self.sg_idx, self.number_of_subgoals))
        self.next_command_s_left = self.eetrack_subgoal[self.sg_idx, :]

    def get_command(self, model, data, root_w, hand_w) :
        self.update_command()
        # print("get_command:", self.next_command_s_left)
        pos_hand_b_left, quat_hand_b_left = subtract_frame_transforms(
            root_w[:3],
            root_w[3:7],
            hand_w[:3],
            hand_w[3:7],
        )

        lerp_command_w_left = self.next_command_s_left
        lerp_command_b_left_pos, lerp_command_b_left_quat = subtract_frame_transforms(
            root_w[:3],
            root_w[3:7],
            lerp_command_w_left[0:3],
            lerp_command_w_left[3:7],
        )

        pos_delta_b_left, rot_delta_b_left = compute_pose_error(
            pos_hand_b_left,
            quat_hand_b_left,
            lerp_command_b_left_pos,
            lerp_command_b_left_quat,
        )

        axa_delta_b_left = wrap_to_pi(rot_delta_b_left)

        hand_command = np.concatenate([pos_delta_b_left, axa_delta_b_left])
        return hand_command

    def vis(self, viewer):
        viewer.user_scn.ngeom = 0
        start = self.eetrack_start
        end = self.eetrack_end
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            np.array([0.01, 0, 0]),
            self.eetrack_subgoal[self.sg_idx, :3],
            np.eye(3).flatten(),
            np.array([0, 1, 0, 1])
        )
        if start[1] == end[1]:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                int(mujoco.mjtGeom.mjGEOM_LINE),
                np.array([10, 0, np.linalg.norm(end - start)]),
                end,
                rotation_matrix_from_vectors([0, 0, 1], start-end).flatten(),
                np.array([1, 0, 0, 0.5])
            )
        else :
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                int(mujoco.mjtGeom.mjGEOM_LINE),
                np.array([10, 0, np.linalg.norm(end - start)]),
                start,
                rotation_matrix_from_vectors([0, 0, 1], end - start).flatten(),
                np.array([1, 0, 0, 0.5])
            )
        viewer.user_scn.ngeom = 2
        viewer.sync()
