from yourdfpy import URDF
from tf2_ros.buffer import Buffer
from common.xml_helper import extract_link_data
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from common.rotation_helper import get_gravity_orientation, transform_imu_data
import rclpy as rp
import numpy as np
import math_utils
from tf2_ros import TransformException


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


class Stage1Observation:
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
        self.prev_pelvis_height = None

    def __call__(self,
                 low_state: LowStateHG,
                 hands_command: np.ndarray
                 ):
        lab_from_mot = self.lab_from_mot
        num_lab_joint = self.num_lab_joint

        ang_vel = np.array([low_state.imu_state.gyroscope],
                           dtype=np.float32)
        base_ang_vel = ang_vel.squeeze(0)

        # NOTE(ycho): requires running `fake_world_tf_pub.py`.
        world_from_pelvis = self.tf_buffer.lookup_transform(
            'world',
            'pelvis',
            rp.time.Time()
        )
        rxn = world_from_pelvis.transform.rotation
        quat = np.array([rxn.w, rxn.x, rxn.y, rxn.z])


        # TODO(ycho): check if the convention "q_base^{-1} @ g" holds.
        projected_gravity = get_gravity_orientation(quat)

        fp_l = body_pose(self.tf_buffer, 'left_ankle_roll_link')
        fp_r = body_pose(self.tf_buffer, 'right_ankle_roll_link')
        foot_pose = np.concatenate([fp_l[0], fp_r[0], fp_l[1], fp_r[1]])

        hp_l = body_pose(self.tf_buffer, 'left_rubber_hand')
        hp_r = body_pose(self.tf_buffer, 'right_rubber_hand')
        hand_pose = np.concatenate([hp_l[0], hp_r[0], hp_l[1], hp_r[1]])

        # Map `low_state` to index-mapped joint_{pos,vel}
        joint_pos = np.zeros(num_lab_joint,
                             dtype=np.float32)
        joint_vel = np.zeros(num_lab_joint,
                             dtype=np.float32)
        joint_pos[lab_from_mot] = [low_state.motor_state[i_mot].q for i_mot in
                                   range(len(lab_from_mot))]
        joint_pos -= self.config.lab_joint_offsets
        joint_vel[lab_from_mot] = [low_state.motor_state[i_mot].dq for i_mot in
                                   range(len(lab_from_mot))]

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
            joint_pos,
            joint_vel,
            hands_command,
            pelvis_height
        ]

        self.prev_pelvis_height = pelvis_height
        # print([np.shape(o) for o in obs])
        return np.concatenate(obs, axis=-1)
    

class Stage2Observation(Stage1Observation):
    def __call__(self,
                 low_state: LowStateHG,
                 height_command: np.ndarray
                 ):
        # array_a[a_from_b] = array_b
        lab_from_mot = self.lab_from_mot
        num_lab_joint = self.num_lab_joint

        ang_vel = np.array([low_state.imu_state.gyroscope],
                           dtype=np.float32)
        base_ang_vel = ang_vel.squeeze(0)

        # NOTE(ycho): requires running `fake_world_tf_pub.py`.
        world_from_pelvis = self.tf_buffer.lookup_transform(
            'world',
            'pelvis',
            rp.time.Time()
        )
        rxn = world_from_pelvis.transform.rotation
        quat = np.array([rxn.w, rxn.x, rxn.y, rxn.z])


        # TODO(ycho): check if the convention "q_base^{-1} @ g" holds.
        projected_gravity = get_gravity_orientation(quat)

        fp_l = body_pose(self.tf_buffer, 'left_ankle_roll_link')
        fp_r = body_pose(self.tf_buffer, 'right_ankle_roll_link')
        foot_pose = np.concatenate([fp_l[0], fp_r[0], fp_l[1], fp_r[1]])

        hp_l = body_pose(self.tf_buffer, 'left_rubber_hand')
        hp_r = body_pose(self.tf_buffer, 'right_rubber_hand')
        hand_pose = np.concatenate([hp_l[0], hp_r[0], hp_l[1], hp_r[1]])

        # Map `low_state` to index-mapped joint_{pos,vel}
        joint_pos = np.zeros(num_lab_joint,
                             dtype=np.float32)
        joint_vel = np.zeros(num_lab_joint,
                             dtype=np.float32)
        joint_pos[lab_from_mot] = [low_state.motor_state[i_mot].q for i_mot in
                                   range(len(lab_from_mot))]
        joint_pos -= self.config.lab_joint_offsets
        joint_vel[lab_from_mot] = [low_state.motor_state[i_mot].dq for i_mot in
                                   range(len(lab_from_mot))]

        world_from_pelvis = self.tf_buffer.lookup_transform(
            'world',
            'pelvis',
            rp.time.Time()
            # clock.get_time()
        )
        pelvis_height = [world_from_pelvis.transform.translation.z]

        if self.prev_pelvis_height is None:
            prev_pelvis_height = pelvis_height
        else:
            prev_pelvis_height = self.prev_pelvis_height

        obs = [
            base_ang_vel,
            projected_gravity,
            foot_pose,
            hand_pose,
            joint_pos,
            joint_vel,
            height_command,
            pelvis_height,
            prev_pelvis_height
        ]

        self.prev_pelvis_height = pelvis_height
        # print([np.shape(o) for o in obs])
        return np.concatenate(obs, axis=-1)
    

class SimpleAction:
    def __init__(self, config):
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

        self.default_offset = np.asarray(self.config.lab_joint_offsets)

        self.joint_pos_action_offset = (
            self.default_offset[self.lab_from_jpa]
            # self.default_offset[self.mot_from_jpa]
        )



    def __call__(self, action, current_joint_pos):
        target_dof_pos = np.zeros(29)

        
        # use default offset for JointPositionAction
        if False:
            joint_pos_action = action[..., :19]
            relative_joint_pos_action = np.zeros(10)
            target_dof_pos[self.mot_from_jpa] = self.joint_pos_action_offset + \
                                                        0.5 * joint_pos_action

            # use current joint pos for RelativeJointPositionAction
            target_dof_pos[self.mot_from_rjpa] = np.array(current_joint_pos) + \
                                                                0.3 * relative_joint_pos_action
        if True:
            joint_pos_action = np.ones(19) * 0.5 # 19
            relative_joint_pos_action = np.zeros(10)

            # target_dof_pos[self.mot_from_jpa] = joint_pos_action
            target_dof_pos[self.mot_from_jpa] = self.joint_pos_action_offset +  0.5 * action[..., :19]
            # target_dof_pos[self.mot_from_rjpa] = relative_joint_pos_action
            target_dof_pos[self.mot_from_rjpa] = np.array(current_joint_pos)[self.lab_from_rjpa] + 0.3 * action[..., 19:]

        return target_dof_pos
    

class VelocityHeightCommand:
    """Command which generates velocity, and height command based on config"""
    def __init__(self, config):
        self.config = config
        self.pelvis_height_w = self.config.target_height
        self.max_velocity = self.config.max_velocity
        self.slow_bound = self.config.slow_bound
    
    def __call__(self, current_pelvis_height_w):
        pelvis_height_diff = self.pelvis_height_w - current_pelvis_height_w
        pelvis_lin_vel_z_w = np.clip( np.sign(pelvis_height_diff) 
                                     * self.max_velocity
                                     * np.sqrt(np.abs(pelvis_height_diff / self.slow_bound)),
                                     -self.max_velocity,
                                     self.max_velocity
                                     )
        return np.asarray([pelvis_lin_vel_z_w, self.pelvis_height_w])