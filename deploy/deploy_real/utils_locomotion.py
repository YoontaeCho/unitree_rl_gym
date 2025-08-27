from yourdfpy import URDF
from tf2_ros.buffer import Buffer
from common.xml_helper import extract_link_data
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from common.rotation_helper import get_gravity_orientation, transform_imu_data
import rclpy as rp
import numpy as np
import math_utils
from tf2_ros import TransformException
import time


axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)

last_timestamp = 0

def body_pose(
        tf_buffer,
        frame: str,
        ref_frame: str = 'pelvis',
        stamp=None,
        rot_type: str = 'axa',
        x=False):
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

        # if x:
        #     global last_timestamp
        #     print()
        #     print(t.header.stamp.nanosec - last_timestamp)
        #     last_timestamp = t.header.stamp.nanosec
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


class EETrackObservation:
    def __init__(self,
                 config,
                 tf_buffer: Buffer):
        self.config = config
        self.num_lab_joint = len(config.lab_joint)
        self.tf_buffer = tf_buffer
        self.lab_from_mot = index_map(config.lab_joint,
                                      config.motor_joint)
        # bring default values
        self.prev_pelvis_height = None
        self.curr_joint_pos = None
    
    def _base_ang_vel(self, low_state: LowStateHG):
        ang_vel = np.array([low_state.imu_state.gyroscope],
                           dtype=np.float32)
        base_ang_vel = ang_vel.squeeze(0)
        return base_ang_vel
    
    def _projected_gravity(self, low_state: LowStateHG):
        # TODO(ycho): check if the convention "q_base^{-1} @ g" holds.
        # world_from_pelvis = self.tf_buffer.lookup_transform(
        #     'world',
        #     'pelvis',
        #     rp.time.Time()
        #     # clock.get_time()
        # )
        # rxn = world_from_pelvis.transform.rotation
        # quat = np.array([rxn.w, rxn.x, rxn.y, rxn.z])
        # projected_gravity = get_gravity_orientation(quat)
        # return projected_gravity

        qw, qx, qy, qz = [
            float(x) for x in 
            low_state.imu_state.quaternion
        ]
        quat = np.array([qw, qx, qy, qz])
        projected_gravity = get_gravity_orientation(quat)
        return projected_gravity

    
    def _foot_pose(self):
        fp_l = body_pose(self.tf_buffer, 'left_ankle_roll_link')
        fp_r = body_pose(self.tf_buffer, 'right_ankle_roll_link')
        foot_pose = np.concatenate([fp_l[0], fp_r[0], fp_l[1], fp_r[1]])
        return foot_pose
    
    def _hand_pose(self):   
        hp_l = body_pose(self.tf_buffer, 'left_rubber_hand')
        hp_r = body_pose(self.tf_buffer,  'end_effector')
        hand_pose = np.concatenate([hp_l[0], hp_r[0], hp_l[1], hp_r[1]])
        return hand_pose
    
    def _joint_pos_vel(self, low_state: LowStateHG, offset):
        # Map `low_state` to index-mapped joint_{pos,vel}
        joint_pos = np.zeros(self.num_lab_joint, dtype=np.float32)
        joint_vel = np.zeros(self.num_lab_joint, dtype=np.float32)
        joint_pos[self.lab_from_mot] = [low_state.motor_state[i_mot].q for i_mot in range(self.num_lab_joint)]
        self.curr_joint_pos = joint_pos.copy()
        joint_pos -= offset
        joint_vel[self.lab_from_mot] = [low_state.motor_state[i_mot].dq for i_mot in range(self.num_lab_joint)]
        return joint_pos, joint_vel
    
    def _pelvis_height(self):
        world_from_pelvis = self.tf_buffer.lookup_transform(
            'world',
            'pelvis',
            rp.time.Time()
        )
        # pelvis_height = [world_from_pelvis.transform.translation.z + 0.04]
        pelvis_height = [world_from_pelvis.transform.translation.z + 0.00]
        # print(f'pelvis_height: {pelvis_height}')
        return pelvis_height
    
    def _pelvis_height_prev(self):
        if self.prev_pelvis_height is None:
            prev_pelvis_height = self._pelvis_height()
        else:
            prev_pelvis_height = self.prev_pelvis_height
        return prev_pelvis_height

    def __call__(self,
                 low_state: LowStateHG,
                 hands_command: np.ndarray
                 ):

        base_ang_vel = self._base_ang_vel(low_state)
        projected_gravity = self._projected_gravity()
        foot_pose = self._foot_pose()
        hand_pose = self._hand_pose()
        joint_pos, joint_vel = self._joint_pos_vel(low_state, self.config.eetrack_joint_offsets)
        pelvis_height = self._pelvis_height()

        # hands_command = np.zeros(6)

        obs = [
            base_ang_vel,       # 3
            projected_gravity,  # 3
            foot_pose,          # 12
            hand_pose,          # 12
            joint_pos,          # 29
            joint_vel,          # 29
            hands_command,      # 2
            pelvis_height       # 1
        ]

        return np.concatenate(obs, axis=-1)
    

class SitObservation(EETrackObservation):
    def _hand_pose(self):   
        hp_l = body_pose(self.tf_buffer, 'left_rubber_hand')
        hp_r = body_pose(self.tf_buffer,  'end_effector')
        hand_pose = np.concatenate([hp_l[0], hp_r[0], hp_l[1], hp_r[1]])
        return hand_pose
    
    def __call__(self,
                 low_state: LowStateHG,
                 height_command: np.ndarray
                 ):
        base_ang_vel = self._base_ang_vel(low_state)
        # NOTE(ycho): requires running `fake_world_tf_pub.py`.
        projected_gravity = self._projected_gravity()
        foot_pose = self._foot_pose()
        hand_pose = self._hand_pose()
        joint_pos, joint_vel = self._joint_pos_vel(low_state, self.config.lab_joint_offsets)
        pelvis_height = self._pelvis_height()
        prev_pelvis_height = self._pelvis_height_prev()

        obs = [
            base_ang_vel,       # 3 
            projected_gravity,  # 3 6
            foot_pose,          # 12 18
            hand_pose,          # 12 30
            joint_pos,          # 29 59
            joint_vel,          # 29 88
            height_command,     # 2 90
            pelvis_height,      # 1 91
            prev_pelvis_height  # 1 92
        ]

        self.prev_pelvis_height = pelvis_height
        return np.concatenate(obs, axis=-1)

class LocomotionObservation(EETrackObservation):
    def __call__(self,
                 low_state: LowStateHG,
                 velocity_command: np.ndarray,
                 phase_command: np.ndarray,
                 last_action: np.ndarray,
                 ):
        base_ang_vel = self._base_ang_vel(low_state)
        # NOTE(ycho): requires running `fake_world_tf_pub.py`.
        projected_gravity = self._projected_gravity()
        joint_pos, joint_vel = self._joint_pos_vel(low_state, self.config.lab_joint_offsets)

        obs = [
            base_ang_vel,       # 3 
            projected_gravity,  # 3 6
            velocity_command,   # 3 9
            joint_pos,          # 29 38
            joint_vel,          # 29 67
            last_action,        # 29 96
            phase_command,      # 2 98
        ]

        return np.concatenate(obs, axis=-1)

class LocomotionObservation_14dof(EETrackObservation):
    def __call__(self,
                 low_state: LowStateHG,
                 velocity_command: np.ndarray,
                 phase,
                 last_action: np.ndarray,
                 ):
        base_ang_vel = self._base_ang_vel(low_state)
        # NOTE(ycho): requires running `fake_world_tf_pub.py`.
        offset = np.zeros(29)
        offset[self.lab_from_mot] = self.config.locomotion_motor_joint_offsets
        projected_gravity = self._projected_gravity(low_state)
        joint_pos, joint_vel = self._joint_pos_vel(low_state, offset)

        phase_command = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])

        obs = [
            base_ang_vel,        
            projected_gravity, 
            joint_pos,          
            joint_vel,         
            velocity_command,  
            phase_command,    
            last_action, 
        ]

        return np.concatenate(obs, axis=-1)
    
from utils_robot import Robot

# TODO
class LocomotionVelocityCommand:
    def __init__(self):
        pass
    
    def __call__(self, root_quat_w, stop_locomotion: bool = False, counter: int = 0):

        if stop_locomotion or counter > 200:
            # 4 sec
            return np.zeros(6)
        else:
            # set to default movement
            # - linvel x = 1
            # - linvel y = 0
            # - angvel z = 0
            return quat_rotate_inverse(root_quat_w, np.array([1, 0, 0]))
            

class LocomotionPhaseCommand:
    """
    Phase command class for locomotion.

    This class works independently with the robot's state.
    """
    def __init__(self, period, dt =0.02):

        self.phase = 0
        self.period = period
        self.dt = dt

    def __call__(self, counter : int = 0):
        self.phase = (counter * self.dt) % self.period / self.period

        return np.array([np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)])

class SitActionVer2:
    def __init__(self, config, robot_model: Robot):
        self.robot_model = robot_model
        self.lim_lo_pin = self.robot_model.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.robot_model.robot.model.upperPositionLimit
        self.config = config
        self.mot_from_lab = index_map(
            self.config.motor_joint,
            self.config.lab_joint
            )
        self.pin_from_mot = index_map(
            self.robot_model.joint_names,
            self.config.motor_joint
            )
        
    def __call__(self, action):
        # motor order
        target_dof_pos = np.zeros(29)
        # checked
        target_dof_pos[self.mot_from_lab] = 0.5 * action


        target_dof_pos = np.clip(
                target_dof_pos,
                self.lim_lo_pin[self.pin_from_mot],
                self.lim_hi_pin[self.pin_from_mot]
            )
        return target_dof_pos

#TODO
class LocomotionAction(SitActionVer2):
    def __call__(self, action):
        # motor order
        target_dof_pos = np.zeros(29)
        # checked
        target_dof_pos[self.mot_from_lab] = 0.5 * action + np.asarray(self.config.lab_joint_offsets)

        target_dof_pos = np.clip(
                target_dof_pos,
                self.lim_lo_pin[self.pin_from_mot],
                self.lim_hi_pin[self.pin_from_mot]
            )
        return target_dof_pos


class LocomotionAction_14dof(SitActionVer2):
    def __init__(self, config, robot_model: Robot):
        super().__init__(config, robot_model)

        lower_body_joints = [
        'left_hip_pitch_joint',
        'right_hip_pitch_joint',
        'left_hip_roll_joint',
        'right_hip_roll_joint',
        'waist_roll_joint',
        'left_hip_yaw_joint',
        'right_hip_yaw_joint',
        'waist_pitch_joint',
        'left_knee_joint',
        'right_knee_joint',
        'left_ankle_pitch_joint', 
        'right_ankle_pitch_joint',
        'left_ankle_roll_joint',
        'right_ankle_roll_joint',
        ]

        self.mot_from_lab_lower_joints = index_map(self.config.motor_joint, lower_body_joints)


    def __call__(self, action):
        # motor order
        target_dof_pos = np.zeros(29)
        # checked
        target_dof_pos[self.mot_from_lab_lower_joints] = 0.5 * action

        target_dof_pos = np.clip(
                target_dof_pos,
                self.lim_lo_pin[self.pin_from_mot],
                self.lim_hi_pin[self.pin_from_mot]
            )
        return target_dof_pos


class NavigationAction(SitActionVer2):
    """
    Returns velocity command (linvel x, linvel y, angvel z)
    """
    def __call__(self, action):
        x, y, z = action

        x = np.clip(x, -0.1, 0.2)
        y = np.clip(y, -0.1, 0.1)
        z = np.clip(z, -0.4, 0.4)

        action = np.array([x, y, z])
        
        return action

import math_utils
yaw_quat = math_utils.as_np(math_utils.yaw_quat)
quat_apply = math_utils.as_np(math_utils.quat_apply)

class NavigationCommand:
    def __init__(self, 
                 x: float = 0.0,
                 y: float = 0.0,
                 heading: float = 0.0):
        self.pos_command_w = np.array([x, y, 0.0], dtype=np.float32)
        self.heading_command_w = np.array([heading], dtype=np.float32)
        self.pos_command_b = np.zeros_like(self.pos_command_w)
        self.heading_command_b = np.zeros(1, dtype=np.float32)
        self.FORWARD_VEC_B = np.array([1.0, 0.0, 0.0], dtype=np.double)
        # current pose
        self.current_xyz_w = np.zeros(3, dtype=np.float32)
        self.current_quat_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def heading_w(self, current_quat_w):
        forward_w = quat_apply(current_quat_w, self.FORWARD_VEC_B)
        return np.arctan2(forward_w[1], forward_w[0])
    
    def update(self):

        target_vec = self.pos_command_w - self.current_xyz_w
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.current_quat_w), target_vec)
        # breakpoint()
        self.heading_command_b[:] = wrap_to_pi(
            self.heading_command_w - 
            np.array([self.heading_w(self.current_quat_w)])
            )

    def command(self, current_xyz_w, current_quat_w):
        
        # update the current pose
        self.current_quat_w = current_quat_w
        self.current_xyz_w = current_xyz_w
        self.update()

        """The desired 2D-pose in base frame. Shape is ( 4)."""
        return np.concatenate([self.pos_command_b, self.heading_command_b], axis=0)


class NavigationObservation(EETrackObservation):
    def __call__(self,
                 low_state: LowStateHG,
                 pose_command: np.ndarray,
                 phase : np.ndarray,
                 last_action: np.ndarray,
                 ):
        base_ang_vel = self._base_ang_vel(low_state)
        # NOTE(ycho): requires running `fake_world_tf_pub.py`.
        offset = np.zeros(29)
        offset[self.lab_from_mot] = self.config.locomotion_motor_joint_offsets
        projected_gravity = self._projected_gravity(low_state)
        joint_pos, joint_vel = self._joint_pos_vel(low_state, offset)
        phase_command = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
        obs = [
            base_ang_vel,       # 3 
            projected_gravity,  # 3 6
            pose_command,       # 4 10
            joint_pos,          # 29 39
            joint_vel,          # 29 68
            last_action,        # 29 97
            phase_command,      # 2 99
        ]

        return np.concatenate(obs, axis=-1)