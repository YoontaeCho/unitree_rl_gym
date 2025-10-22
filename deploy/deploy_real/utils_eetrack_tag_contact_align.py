from typing import Union, List
import numpy as np
import torch
import torch as th

import rclpy as rp


from tf2_ros import TransformException

import math_utils
import random as rd
from scipy.spatial.transform import Rotation as R


axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)
yaw_quat = math_utils.as_np(math_utils.yaw_quat)
matrix_from_quat = math_utils.as_np(math_utils.matrix_from_quat)
subtract_frame_transforms = math_utils.as_np(math_utils.subtract_frame_transforms)
quat_from_euler_xyz = math_utils.as_np(math_utils.quat_from_euler_xyz)

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

def interpolate_quaternion(quat1, quat2, n_segments):
    quat1 = torch.from_numpy(quat1[None, None, ...])
    quat2 = quat2[None, ...]
    t = torch.linspace(0, 1, n_segments + 1).view(1, -1, 1)
    interp_q = math_utils.slerp_vectorized(quat1, quat2, t)
    interp_q = interp_q[0]
    return interp_q

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
    def __init__(
        self,
        root_state_w,
        tf_buffer,
        clock,
        eetrack_vel=0.005,
        start_pos_w=None,
        end_pos_w=None,
        to_start=False,
        inverse_y=False
    ):
        self.clock = clock
        self.tf_buffer = tf_buffer
        
        self.eetrack_vel = eetrack_vel

        # Welding offset from the line
        self.offset_len = 0.015

        self.step_dt = 0.02
        self.dt_segment_length = self.eetrack_vel * self.step_dt # 0.0002
        
        self.device = "cpu"
        self.sg_idx = 0
        # first subgoal sampling time = 1.0s
        # self.init_time = rp.time.Time()#.nanoseconds / 1e9 + 1.0
        self.init_time = self.clock.get_time()
        self.init_root_state_w = root_state_w
        self.init_root_pos_w = root_state_w[:, :3]
        
        # self.eetrack_start_w, self.eetrack_start_quat_w = body_pose(self.tf_buffer, "eetrack_start", "mid_sole_link", rot_type="quat")
        # self.eetrack_end_w, self.eetrack_end_quat_w = body_pose(self.tf_buffer, "eetrack_end", "mid_sole_link", rot_type="quat")

        # Hard-coded due to tf subscription error.
        # self.eetrack_start_w, self.eetrack_start_quat_w = np.array([0.3030, -0.3624,  0.3067]), np.array([-0.8831, -0.1125, -0.3658,  0.2715])
        # self.eetrack_end_w, self.eetrack_end_quat_w = np.array([0.2188, -0.4865,  0.3067]), np.array([-0.8831, -0.1125, -0.3658,  0.2715])
        
        
        ############################ TODO Subscribe Welding line publish (start - end point) ###############################
        data = dict(np.load("test_welding_path.npz")) # pyroki path??
        data["pos"][:,2] += 0.01

        
        if start_pos_w is None or end_pos_w is None:
            target_0_pos, target_0_quat = body_pose(
                    self.tf_buffer,
                    frame="end_effector",
                    ref_frame="mid_sole_link",
                    rot_type='quat'
            )
            target_1_pos, target_1_quat = body_pose(
                    self.tf_buffer,
                    frame="end_effector",
                    ref_frame="mid_sole_link",
                    rot_type='quat'
            )
            eetrack_start_w = target_0_pos
            eetrack_start_quat_w = target_0_quat

            eetrack_end_w = target_1_pos
            eetrack_end_quat_w = target_1_quat

            self.eetrack_start_w, self.eetrack_start_quat_w = eetrack_start_w, eetrack_start_quat_w
            self.eetrack_end_w, self.eetrack_end_quat_w = eetrack_end_w, eetrack_end_quat_w

            # TODO: remove after testing
            self.eetrack_start_w, self.eetrack_start_quat_w = np.array([0.3030, -0.3624,  0.3067]), np.array([-0.8831, -0.1125, -0.3658,  0.2715])
            self.eetrack_end_w, self.eetrack_end_quat_w = np.array([0.2188, -0.4865,  0.3067]), np.array([-0.8831, -0.1125, -0.3658,  0.2715])
        else:
            welding_start_pos_w = start_pos_w.copy()
            welding_end_pos_w = end_pos_w.copy()
            eetrack_start_pos_w, eetrack_start_quat_w, eetrack_end_pos_w, eetrack_end_quat_w = eetrack.get_eetrack_pos_quat(
                welding_start_pos_w,
                welding_end_pos_w,
                offset_len=self.offset_len,
                approach_deg=40.0,
                # approach_deg=35.0,
                inverse_y=inverse_y
            )
            self.eetrack_start_w, self.eetrack_start_quat_w = eetrack_start_pos_w, eetrack_start_quat_w
            self.eetrack_end_w, self.eetrack_end_quat_w = eetrack_end_pos_w, eetrack_end_quat_w

        if to_start:
            self.create_eetrack()
            self.eetrack_subgoal = self.create_subgoal_to_start()
        else:
            self.create_eetrack()
            self.eetrack_subgoal = self.create_subgoal()


    def create_eetrack(self):
        self.eetrack_line_length = np.linalg.norm(self.eetrack_start_w - self.eetrack_end_w)
        self.number_of_subgoals = int(self.eetrack_line_length / self.dt_segment_length)

        self.eetrack_start_th_w = th.as_tensor(self.eetrack_start_w, dtype=th.double, device=self.device)[None]
        self.eetrack_end_th_w = th.as_tensor(self.eetrack_end_w, dtype=th.double, device=self.device)[None]

        self.eetrack_quat_w = th.as_tensor(self.eetrack_start_quat_w, dtype=th.double, device=self.device)[None]


    def create_subgoal(self):
        # initial hand pos
        pos_hand_w_left, quat_hand_w_left = body_pose(
            self.tf_buffer,
            frame="end_effector",
            ref_frame="mid_sole_link",
            rot_type='quat'
        )
        
        self.to_eetrack_sgs_num =  to_eetrack_sgs_num = 50
        # 1. current hand pose -> eetack start
        to_eeline_subgoals = interpolate_position(
            torch.tensor(pos_hand_w_left).unsqueeze(0),
            self.eetrack_start_th_w,
            to_eetrack_sgs_num
        )
        
        # eetrack start pos -> eetrack end pos
        # Welding line
        on_eeline_subgoals = interpolate_position(
            self.eetrack_start_th_w,
            # self.eetrack_start_w,
            self.eetrack_end_th_w,
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

        lerped_quats = []

        # to_z_quat = interpolate_quaternion(
        #     quat_hand_w_left,
        #     self.eetrack_quat_w,
        #     to_z_num
        # ).unsqueeze(0)
        # lerped_quats.append(to_z_quat)

        z_to_eetrack_quat = interpolate_quaternion(
            quat_hand_w_left,
            self.eetrack_quat_w,
            to_eetrack_sgs_num
        ).unsqueeze(0)
        lerped_quats.append(z_to_eetrack_quat)

        to_eetrack_quat = torch.cat(
            lerped_quats,
            dim=1,
        )
        on_eetrack_quat = self.eetrack_quat_w.unsqueeze(1).repeat(1, self.number_of_subgoals + 1, 1)
        
        eetrack_quat = torch.cat([to_eetrack_quat, on_eetrack_quat], dim=1)

        self.number_of_subgoals += to_eetrack_sgs_num

        return th.cat([eetrack_subgoals, eetrack_quat], dim=2)
    
    def create_subgoal_to_start(self):
        # initial hand pos
        pos_hand_w_left, quat_hand_w_left = body_pose(
            self.tf_buffer,
            frame="end_effector",
            ref_frame="mid_sole_link",
            rot_type='quat'
        )

        self.to_start_line_length = np.linalg.norm(pos_hand_w_left - self.eetrack_start_w)
        self.to_eetrack_sgs_num = self.number_of_subgoals = int(self.to_start_line_length / self.dt_segment_length)

        to_start_subgoals = interpolate_position(
            th.tensor(pos_hand_w_left).unsqueeze(0),
            self.eetrack_start_th_w,
            self.number_of_subgoals,
        )
        to_start_subgoals = th.stack(to_start_subgoals, dim=1)

        to_start_quat = interpolate_quaternion(
            quat_hand_w_left,
            self.eetrack_quat_w,
            self.number_of_subgoals,
        ).unsqueeze(0)

        return th.cat([to_start_subgoals, to_start_quat], dim=2)

    def update_command(self):
        """
        update command for eetrack
        initial_goal: True if this is the first command.
        initial_goal should be given as True or False by user.
        """
        self.sg_idx += 1
        self.sg_idx = min(self.sg_idx , self.number_of_subgoals)
        if self.sg_idx == self.to_eetrack_sgs_num:
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")
            print("--------------- On eetrack start ---------------")

        # self.sg_idx = 0
        print("Percent:", self.sg_idx/self.number_of_subgoals)
        self.next_command_s_left = self.eetrack_subgoal[..., self.sg_idx, :]

    def get_command(self, root_state_w):
        self.update_command()

        pos_hand_b_left, quat_hand_b_left = body_pose(
            self.tf_buffer,
            "end_effector",
            rot_type='quat'
        )

        lerp_command_w_left = self.next_command_s_left
        # root_state = pelvis
        (self.lerp_command_b_left_pos,
         self.lerp_command_b_left_quat) = math_utils.subtract_frame_transforms(
            root_state_w[..., 0:3],
            root_state_w[..., 3:7],
            lerp_command_w_left[:, 0:3],
            lerp_command_w_left[:, 3:7],
        )

        # lerp_command_b_left = lerp_command_w_left

        pos_delta_b_left, rot_delta_b_left = math_utils.compute_pose_error(
            torch.from_numpy(pos_hand_b_left)[None],
            torch.from_numpy(quat_hand_b_left)[None],
            self.lerp_command_b_left_pos,
            self.lerp_command_b_left_quat,
        )
        axa_delta_b_left = math_utils.wrap_to_pi(rot_delta_b_left)

        hand_command = torch.cat((pos_delta_b_left, axa_delta_b_left), dim=-1)
        return hand_command
    

    @staticmethod
    def get_eetrack_pos_quat(
        welding_start_pos_w, 
        welding_end_pos_w, 
        offset_len=0.01, 
        approach_deg=45.0,
        inverse_y=False
        ):
        # Computing the quaternion of welder (compute approaching vector of welder)
        # Assume the point is in world frame.
        z_up_axis = np.array([0,0,1])
        # Assume the start point is in left (+y) and the end point is in right (-y)
        if not inverse_y:
            y_axis = welding_start_pos_w - welding_end_pos_w
        else:
            y_axis = welding_end_pos_w - welding_start_pos_w

        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_up_axis)
        z_up_mat = np.stack([x_axis, y_axis, z_up_axis], axis=1)

        # Rotate z_up for approach deg about y-axis
        sciR = R.from_matrix(z_up_mat) * R.from_euler('y', approach_deg, degrees=True)
        # Assume same rotation for start and end
        eetrack_start_quat_w = eetrack_end_quat_w = np.roll(sciR.as_quat(), 1)

        eetrack_mat = sciR.as_matrix()
        eetrack_x_axis = eetrack_mat[:,0]
        eetrack_start_pos_w = welding_start_pos_w - offset_len*eetrack_x_axis
        eetrack_end_pos_w = welding_end_pos_w - offset_len*eetrack_x_axis

        # Manual offset due to calibration error and vision error.
        eetrack_y_axis = eetrack_mat[:,1]
        eetrack_start_pos_w += 0.0 * eetrack_y_axis
        eetrack_end_pos_w += 0.0 * eetrack_y_axis

        eetrack_z_axis = eetrack_mat[:,2]
        eetrack_start_pos_w += 0.0 * eetrack_z_axis
        eetrack_end_pos_w += 0.0 * eetrack_z_axis

        return eetrack_start_pos_w, eetrack_start_quat_w, eetrack_end_pos_w, eetrack_end_quat_w
