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
    
    def _projected_gravity(self):
        # TODO(ycho): check if the convention "q_base^{-1} @ g" holds.
        world_from_pelvis = self.tf_buffer.lookup_transform(
            'world',
            'pelvis',
            rp.time.Time()
            # clock.get_time()
        )
        rxn = world_from_pelvis.transform.rotation
        quat = np.array([rxn.w, rxn.x, rxn.y, rxn.z])
        projected_gravity = get_gravity_orientation(quat)
        return projected_gravity
    
    def _projected_gravity_from_lowstate(self, low_state: LowStateHG):
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
    
class EETrackObservationWithLastAction:
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
    
    def _projected_gravity(self):
        # TODO(ycho): check if the convention "q_base^{-1} @ g" holds.
        world_from_pelvis = self.tf_buffer.lookup_transform(
            'world',
            'pelvis',
            rp.time.Time()
            # clock.get_time()
        )
        rxn = world_from_pelvis.transform.rotation
        quat = np.array([rxn.w, rxn.x, rxn.y, rxn.z])
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
                 hands_command: np.ndarray,
                 last_actions: np.ndarray,
                 last_last_actions: np.ndarray,
                 clamp: bool = False
                 ):

        base_ang_vel = self._base_ang_vel(low_state)
        projected_gravity = self._projected_gravity()
        foot_pose = self._foot_pose()
        hand_pose = self._hand_pose()
        joint_pos, joint_vel = self._joint_pos_vel(low_state, self.config.eetrack_joint_offsets)
        pelvis_height = self._pelvis_height()

        # hands_command = np.zeros(6)
        if clamp:
            xyz = hands_command[:3]
            axa = hands_command[3:]
            
            # 1
            # xyz = xyz.clip(min=-0.02, max=0.02)
            # axa = axa.clip(min=-0.2, max=0.2)
            # 2
            # xyz = xyz.clip(min=-0.01, max=0.01)
            # axa = axa.clip(min=-0.2, max=0.2)
            # 3
            # xyz = xyz.clip(min=-0.005, max=0.005)
            # axa = axa.clip(min=-0.2, max=0.2)
            # 4
            xyz = xyz.clip(min=-0.005, max=0.005)
            axa = axa.clip(min=-0.1, max=0.1)

            hands_command = np.concatenate([xyz, axa], axis=0)

        obs = [
            base_ang_vel,       # 3
            projected_gravity,  # 3 6
            foot_pose,          # 12 18
            # hand_pose,          # 12 30
            joint_pos,          # 29 47
            joint_vel,          # 29 88
            hands_command,      # 6 94
            pelvis_height,      # 1 95
            # last_actions,
            # last_last_actions
        ]

        return np.concatenate(obs, axis=-1)
    

class SitObservation(EETrackObservation):
    def _hand_pose(self):   
        hp_l = body_pose(self.tf_buffer, 'left_rubber_hand')
        hp_r = body_pose(self.tf_buffer,  'welder')
        hand_pose = np.concatenate([hp_l[0], hp_r[0], hp_l[1], hp_r[1]])
        return hand_pose
    
    def _pelvis_height(self, xyz):
        pelvis_height = xyz[2:3]
        return pelvis_height
    
    def _pelvis_height_prev(self, xyz):
        if self.prev_pelvis_height is None:
            prev_pelvis_height = self._pelvis_height(xyz)
        else:
            prev_pelvis_height = self.prev_pelvis_height
        return prev_pelvis_height
    
    def __call__(self,
                 low_state: LowStateHG,
                 height_command: np.ndarray,
                 xyz
                 ):
        base_ang_vel = self._base_ang_vel(low_state)
        # NOTE(ycho): requires running `fake_world_tf_pub.py`.
        projected_gravity = self._projected_gravity_from_lowstate(low_state)
        foot_pose = self._foot_pose()
        hand_pose = self._hand_pose()
        joint_pos, joint_vel = self._joint_pos_vel(low_state, self.config.lab_joint_offsets)
        pelvis_height = self._pelvis_height(xyz)
        prev_pelvis_height = self._pelvis_height_prev(xyz)

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
    
    
class SitObservation_v2(SitObservation):
    def __call__(self,
                 low_state: LowStateHG,
                 height_command: np.ndarray,
                 xyz,
                 hip_pitch_joint_offset = None
                 ):
        base_ang_vel = self._base_ang_vel(low_state)
        # NOTE(ycho): requires running `fake_world_tf_pub.py`.
        projected_gravity = self._projected_gravity_from_lowstate(low_state)
        foot_pose = self._foot_pose()
        hand_pose = self._hand_pose()
        joint_offset =  self.config.lab_joint_offsets_sit
        if hip_pitch_joint_offset is not None:
            joint_offset[0] = hip_pitch_joint_offset[0]
            joint_offset[1] = hip_pitch_joint_offset[1]
        joint_pos, joint_vel = self._joint_pos_vel(low_state, joint_offset)
        pelvis_height = self._pelvis_height(xyz)
        prev_pelvis_height = self._pelvis_height_prev(xyz)

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



from utils_robot import Robot

class SimpleAction:
    def __init__(self, config, robot_model: Robot):
        self.robot_model = robot_model
        self.lim_lo_pin = self.robot_model.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.robot_model.robot.model.upperPositionLimit
        self.config = config
        self.mot_from_jpa = index_map(
            self.config.motor_joint, 
            self.config.jpa_joint
            )
        self.mot_from_rjpa = index_map(
            self.config.motor_joint,
            self.config.rjpa_joint
            )
        
        self.jpa_from_mot = index_map(
            self.config.jpa_joint,
            self.config.motor_joint
            )
        
        self.rjpa_from_mot = index_map(
            self.config.rjpa_joint,
            self.config.motor_joint
            )
        
        self.lab_from_jpa = index_map(
            self.config.lab_joint,
            self.config.jpa_joint
            )
        self.lab_from_rjpa = index_map(
            self.config.lab_joint,
            self.config.rjpa_joint
            )
        self.lab_from_mot = index_map(
            self.config.lab_joint,
            self.config.motor_joint
            )
        self.rjpa_from_lab = index_map(
            self.config.rjpa_joint,
            self.config.lab_joint
            )   
        

        self.default_offset = np.asarray(self.config.lab_joint_offsets)

        # checked
        self.joint_pos_action_offset = (
            self.default_offset[self.lab_from_jpa]
            # self.default_offset[self.mot_from_jpa]
        )

        self.mot_from_lab = index_map(
            self.config.motor_joint,
            self.config.lab_joint
            )
        self.mot_from_lab_eetrack = index_map(
            self.config.motor_joint,
            self.config.lab_joint_eetrack
            )
        
        self.pin_from_mot = index_map(
            self.robot_model.joint_names,
            self.config.motor_joint
            )
        
        self.counter = 0



    def __call__(self, action, obs):
        """Generate motor-ordered joint position command from action and current joint position"""
        q = obs[..., 30:59] #FIXME: 32:61 is not always correct
        q_mot = np.zeros(29)
        q_mot[self.mot_from_lab] = q

        # NOTE(hh) Why should we add this?
        q_mot[self.mot_from_lab] += np.asarray(self.config.lab_joint_offsets)

        # motor order
        target_dof_pos = np.zeros(29)
        target_dof_pos += q_mot

        # one_hot = np.zeros(19)
        # one_hot[5] = -0.1
        # if self.counter < 50:
        #     target_dof_pos[self.mot_from_jpa] += one_hot
        
        # # print(q[7])
        # self.counter += 1


        # checked
        target_dof_pos[self.mot_from_jpa] = self.joint_pos_action_offset +  0.5 * action[..., :19]

        # checked
        target_dof_pos[self.mot_from_rjpa] += 0.3 * action[..., 19:]

        target_dof_pos = np.clip(
                target_dof_pos,
                self.lim_lo_pin[self.pin_from_mot],
                self.lim_hi_pin[self.pin_from_mot]
            )

        return target_dof_pos

class SimpleEETrackAction(SimpleAction):
    def __call__(self, action, obs):
        """Generate motor-ordered joint position command from action and current joint position"""
        q = obs[..., 30:59]
        q_mot = np.zeros(29)
        q_mot[self.mot_from_lab] = q

        # NOTE(hh) Why should we add this?
        q_mot[self.mot_from_lab] += np.asarray(self.config.lab_joint_offsets)

        # motor order
        target_dof_pos = np.zeros(29)
        target_dof_pos += q_mot


        target_dof_pos[self.mot_from_lab_eetrack] += 0.3 * action

        target_dof_pos = np.clip(
                target_dof_pos,
                self.lim_lo_pin[self.pin_from_mot],
                self.lim_hi_pin[self.pin_from_mot]
            )

        return target_dof_pos
    

class VelocityHeightCommand:
    """Command which generates velocity, and height command based on config"""
    def __init__(self, config):
        self.config = config
        self.pelvis_height_w = self.config.target_height
        self.max_velocity = self.config.max_velocity
        self.slow_bound = self.config.slow_bound

        self.initial_pelvis_height = None
    
    def __call__(self, current_pelvis_height_w :float, sitting :bool = False):
        if sitting:
            target_height = self.pelvis_height_w
        else:
            # if self.initial_pelvis_height is None:
            #     self.initial_pelvis_height = current_pelvis_height_w
            target_height = 0.7
            
        pelvis_height_diff = target_height - current_pelvis_height_w
        pelvis_lin_vel_z_w = np.clip( np.sign(pelvis_height_diff) 
                                    * self.max_velocity
                                    * np.sqrt(np.abs(pelvis_height_diff / self.slow_bound)),
                                    -self.max_velocity,
                                    self.max_velocity
                                    # 0
                                    )
        self.pelvis_lin_vel_z_w = pelvis_lin_vel_z_w
        return np.asarray([pelvis_lin_vel_z_w, target_height])


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

class EETrackActionVer2(SitActionVer2):
    def __call__(self, action):
        # motor order
        target_dof_pos = np.zeros(29)
        # checked
        target_dof_pos[self.mot_from_lab] = 0.5 * action + np.asarray(self.config.eetrack_joint_offsets)

        target_dof_pos = np.clip(
                target_dof_pos,
                self.lim_lo_pin[self.pin_from_mot],
                self.lim_hi_pin[self.pin_from_mot]
            )
        return target_dof_pos

# HC    
class TF2Pose():
    def __init__(self,
                 tf_buffer: Buffer):
        self.tf_buffer = tf_buffer
    def __call__(self,
                 ref_frame:str, 
                 frame: str):
        pose = body_pose(self.tf_buffer, ref_frame=ref_frame, frame=frame)
        return np.concatenate([pose[0], pose[1]])
    
# HC
import asyncio
def set_transfrom(tf_buffer: Buffer, node, timeout:float = 10.0):
    start_time = time.time()

    while True:
        try:
            t = tf_buffer.lookup_transform(
                'world',  
                'camera_init',
                rp.time.Time())
            # print("########## Got it ########## ")
            return
        
        except TransformException as ex:
            print(f'Could not transform world to pelvis: {ex}')

        # Check timeout
        if time.time() - start_time > timeout:
            print("########## ERROR ########## ")
            return
        
        # Sleep a bit to avoid busy loop
        time.sleep(0.1)
    
    

    # tf_future = tf_buffer.wait_for_transform_async('world', 'camera_init', rp.time.Time())
    # rp.spin_until_future_complete(node, tf_future, timeout_sec=10.0)

    # if tf_future.done():
    #     print("########## Got it! ########## ")
    # else:
    #     print("########## ERROR ########## ")

    # return

    # try:
    #     # asyncio.wait_for will raise asyncio.TimeoutError if timeout exceeded
    #     await asyncio.wait_for(
    #         tf_buffer.lookup_transform_async('world', 'pelvis', rp.time.Time()),
    #         timeout=timeout
    #     )

    # except asyncio.TimeoutError:
    #     print("########## TIMEOUT ##########")
    #     return
