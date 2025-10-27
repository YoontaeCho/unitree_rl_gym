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


class BaseObservation:
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
        pelvis_height = [world_from_pelvis.transform.translation.z]
        return pelvis_height
    
    def _pelvis_height_prev(self):
        if self.prev_pelvis_height is None:
            prev_pelvis_height = self._pelvis_height()
        else:
            prev_pelvis_height = self.prev_pelvis_height
        return prev_pelvis_height

    def __call__(self):
        raise NotImplementedError
    

class TrajCmdObservation(BaseObservation):
    def __call__(self,
                 low_state: LowStateHG,
                 cmd: np.ndarray,
                 last_action: np.ndarray,
                 ):
        """[INFO] Observation Manager: <ObservationManager> contains 5 groups.
        +---------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (113,)) |
        +-----------+---------------------------------+-----------+
        |   Index   | Name                            |   Shape   |
        +-----------+---------------------------------+-----------+
        |     0     | base_ang_vel                    |    (3,)   |
        |     1     | projected_gravity               |    (3,)   |
        |     2     | joint_pos                       |   (29,)   |
        |     3     | joint_vel                       |   (29,)   |
        |     4     | actions                         |   (29,)   |
        |     5     | traj_commands                   |   (20,)   |
        +-----------+---------------------------------+-----------+"""

        base_ang_vel = self._base_ang_vel(low_state)
        offset = np.zeros(29)
        offset[:] = self.config.lab_joint_offsets
        projected_gravity = self._projected_gravity(low_state)
        joint_pos, joint_vel = self._joint_pos_vel(low_state, offset)

        obs = [
            base_ang_vel,        
            projected_gravity, 
            joint_pos,          
            joint_vel,         
            last_action,  
            cmd,
        ]

        return np.concatenate(obs, axis=-1)
    
from utils_robot import Robot


class BaseAction:
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


class TrajCmdAction(BaseAction):
    def __init__(self, config, robot_model: Robot):
        super().__init__(config, robot_model)

        """@configclass
        class ActionsCfg:
            joint_pos = collision_skill_mdp.RFIJointPositionActionCfg(
                asset_name="robot",
                joint_names=[".*"],
                scale=0.5,
                use_default_offset=True,
                randomize_torque_rfi=True,
                rfi_lim_scale=0.1,  # Scale factor for the random torque RFI
            )"""

    def __call__(self, action):
        # motor order
        target_dof_pos = np.zeros(29)
        # checked
        target_dof_pos[self.mot_from_lab] = 0.5 * action + self.config.lab_joint_offsets


        target_dof_pos = np.clip(
                target_dof_pos,
                self.lim_lo_pin[self.pin_from_mot],
                self.lim_hi_pin[self.pin_from_mot]
            )
        return target_dof_pos

