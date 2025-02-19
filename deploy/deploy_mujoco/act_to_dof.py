import numpy as np
import pinocchio as pin
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

def xyzw2wxyz(q_xyzw: np.ndarray, dim: int = -1):
    return np.roll(q_xyzw, 1, axis=dim)

class ActToDof :
    def __init__ (self, config, ikctrl):
        self.config = config
        self.ikctrl = ikctrl
        self.lim_lo_pin = self.ikctrl.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.ikctrl.robot.model.upperPositionLimit
        self.pin_from_lab = index_map(
            self.ikctrl.joint_names,
            self.config.lab_joint
        )
        self.mot_from_lab = index_map(
            self.config.motor_joint,
            self.config.lab_joint
        )
        self.mot_from_arm = index_map(
            self.config.motor_joint,
            self.config.arm_joint
        )
        self.mot_from_nonarm = index_map(
            self.config.motor_joint,
            self.config.non_arm_joint
        )
        self.lab_from_nonarm = index_map(
            self.config.lab_joint,
            self.config.non_arm_joint
        )
        self.default_nonarm = (
            np.asarray(self.config.lab_joint_offsets)[self.lab_from_nonarm]
        )

    def __call__ (self, obs, action):
        hands_command_b = obs[..., 119:125]
        non_arm_joint_pos = action[..., :22]
        left_arm_residual = action[..., 22:29]

        q_lab = obs[..., 32:61] # current : lab joint order
        
        # pin : pin joint order
        q_pin = np.zeros_like (self.ikctrl.cfg.q)
        q_pin[self.pin_from_lab] = q_lab + np.asarray(self.config.lab_joint_offsets)
        
        # mot : lab joint order
        q_mot = np.zeros(29)
        q_mot[self.mot_from_lab] = q_lab + np.asarray(self.config.lab_joint_offsets)

        axa = hands_command_b[..., 3:]
        angle = np.asarray(np.linalg.norm(axa, axis=-1))
        axis = axa / np.maximum(angle, 1e-6)
        d_quat = quat_from_angle_axis(angle, axis)

        source_pose = self.ikctrl.fk(q_pin)
        source_xyz = source_pose.translation
        source_quat = xyzw2wxyz(pin.Quaternion(source_pose.rotation).coeffs())
        print('fk_source', np.concatenate([source_xyz, source_quat]))
        target_xyz = source_xyz + hands_command_b[..., :3]
        target_quat = quat_mul(d_quat, source_quat)
        target = np.concatenate([target_xyz, target_quat])
        print("fk_target", target)
        res_q_ik = self.ikctrl(q_pin, target)

        target_dof_pos = np.zeros(29)
        target_dof_pos += q_mot
        target_dof_pos[self.mot_from_arm] = res_q_ik

        target_dof_pos[self.mot_from_arm] += np.clip(
            0.3 * left_arm_residual, -0.2, 0.2
        )

        target_dof_pos[self.mot_from_nonarm] = (
            self.default_nonarm + 0.5 * non_arm_joint_pos
        )

        return target_dof_pos

