import os
from typing import Tuple
from contextlib import contextmanager
from pathlib import Path
import time
import yaml
import numpy as np
import torch as th
# from yourdfpy import URDF
import pinocchio as pin
import pink
from pink.tasks import FrameTask
from scipy.spatial.transform import Rotation as R
@contextmanager
def with_dir(d):
    d0 = os.getcwd()
    try:
        os.chdir(d)
        yield
    finally:
        os.chdir(d0)
def xyzw2wxyz(q_xyzw: th.Tensor, dim: int = -1):
    if isinstance(q_xyzw, np.ndarray):
        return np.roll(q_xyzw, 1, axis=dim)
    return th.roll(q_xyzw, 1, dims=dim)
def wxyz2xyzw(q_wxyz: th.Tensor, dim: int = -1):
    if isinstance(q_wxyz, np.ndarray):
        return np.roll(q_wxyz, -1, axis=dim)
    return th.roll(q_wxyz, -1, dims=dim)
def SE3_to_xyzwxyz(cfg):
    pos = cfg.translation
    rot_mat = cfg.rotation
    quat = pin.Quaternion(rot_mat)
    return np.concatenate([pos, xyzw2wxyz(quat.coeffs())])
def dls_ik(
        dpose: np.ndarray,
        jac: np.ndarray,
        sqlmda: float
):
    """
    Arg:
        dpose: task-space error (A[..., err])
        jac: jacobian (A[..., err, dof])
        sqlmda: DLS damping factor.
    Return:
        joint residual (A[..., dof])
    """
    if isinstance(dpose, tuple):
        dpose = np.concatenate([dpose[0], dpose[1]], axis=-1)
    J = jac
    A = J @ J.T
    # NOTE(ycho): add to view of diagonal
    a = np.einsum('...ii->...i', A)
    a += sqlmda
    dq = (J.T @ np.linalg.solve(A, dpose[..., None]))[..., 0]
    return dq
class IKCtrl:
    def __init__(self,
                 urdf_path: str,
                 act_joints: Tuple[str, ...],
                 frame: str = 'left_hand_palm_link',
                 sqlmda: float = 0.05**2,
                 fix_base: bool = True):
        path = Path(urdf_path)
        with with_dir(path.parent):
            robot = pin.RobotWrapper.BuildFromURDF(path.name,
                                                   package_dirs=["."],
                                                   root_joint=None if fix_base else pin.JointModelFreeFlyer())
            self.robot = robot
            # NOTE(ycho): we skip joint#0(="universe")
            joint_names = list(self.robot.model.names)
            assert (joint_names[0] == 'universe')
            self.joint_names = joint_names[1:]
        # NOTE(ycho): build index map between pin.q and other set(s) of ordered
        # joints.
        pin_from_act = []
        for j in act_joints:
            pin_from_act.append(robot.index(j) - 1)
        self.frame = frame
        self.pin_from_act = np.asarray(pin_from_act, dtype=np.int32)
        self.task = FrameTask(frame, position_cost=1.0, orientation_cost=0.0)
        self.sqlmda = sqlmda
        self.cfg = pink.Configuration(robot.model, robot.data,
                                      np.zeros_like(robot.q0))
    def fk(self, q: np.ndarray, frame: str = None):
        if frame is None:
            frame = self.frame
        robot = self.robot
        return pink.Configuration(
            robot.model, robot.data, q).get_transform_frame_to_world(
            frame)
    def __call__(self,
                 q0: np.ndarray,
                 target_pose: np.ndarray,
                 rel: bool = False,
                 v0: np.ndarray = None
                 ):
        """
        Arg:
            q0: Current robot joints; A[..., 43?]
            target_pose:
                Policy output. A[..., 7] formatted as (xyz, q_{wxyz})
                Given as world frame absolute pose, for some reason.
        Return:
            joint residual: A[..., 7]
        """
        robot = self.robot
        # source pose
        self.cfg.update(q0)
        T0 = self.cfg.get_transform_frame_to_world(self.frame)
        # target pose
        dst_xyz = target_pose[..., 0:3]
        dst_quat = pin.Quaternion(wxyz2xyzw(target_pose[..., 3:7]))
        T1 = pin.SE3(dst_quat, dst_xyz)
        if rel:
            TL = pin.SE3.Identity()
            TL.translation = dst_xyz
            TR = pin.SE3.Identity()
            TR.rotation = dst_quat.toRotationMatrix()
            T1 = TL * T0 * TR
        # jacobian
        self.task.set_target(T0)
        jac = self.task.compute_jacobian(self.cfg)
        jac = jac[:, self.pin_from_act]
        # error&ik
        dT = T1.actInv(T0)
        dpose = pin.log(dT).vector
        dq = dls_ik(dpose, jac, self.sqlmda)
        # optionally also compute gravity related terms ?
        if v0 is None:
            v0 = np.zeros_like(q0)
        h = pin.nonLinearEffects(robot.model,
                                 robot.data,
                                 q0,
                                 # FIXME(ycho): use true velocity here.
                                 v0)
        tau_arm = h[self.pin_from_act]
        return dq, tau_arm
    
lab_joint = [
  'left_hip_pitch_joint',
  'right_hip_pitch_joint',
  'waist_yaw_joint',
  'left_hip_roll_joint',
  'right_hip_roll_joint',
  'waist_roll_joint',
  'left_hip_yaw_joint',
  'right_hip_yaw_joint',
  'waist_pitch_joint',
  'left_knee_joint',
  'right_knee_joint',
  'left_shoulder_pitch_joint',
  'right_shoulder_pitch_joint',
  'left_ankle_pitch_joint', 
  'right_ankle_pitch_joint',
  'left_shoulder_roll_joint',
  'right_shoulder_roll_joint',
  'left_ankle_roll_joint',
  'right_ankle_roll_joint',
  'left_shoulder_yaw_joint',
  'right_shoulder_yaw_joint',
  'left_elbow_joint',
  'right_elbow_joint',
  'left_wrist_roll_joint',
  'right_wrist_roll_joint',
  'left_wrist_pitch_joint',
  'right_wrist_pitch_joint',
  'left_wrist_yaw_joint',
  'right_wrist_yaw_joint'
]

motor_joint = [
  'left_hip_pitch_joint',
  'left_hip_roll_joint',
  'left_hip_yaw_joint',
  'left_knee_joint',
  'left_ankle_pitch_joint',
  'left_ankle_roll_joint',
  'right_hip_pitch_joint',
  'right_hip_roll_joint',
  'right_hip_yaw_joint',
  'right_knee_joint',
  'right_ankle_pitch_joint',
  'right_ankle_roll_joint',
  'waist_yaw_joint',
  'waist_roll_joint',
  'waist_pitch_joint',
  'left_shoulder_pitch_joint',
  'left_shoulder_roll_joint',
  'left_shoulder_yaw_joint',
  'left_elbow_joint',
  'left_wrist_roll_joint',
  'left_wrist_pitch_joint',
  'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint',
  'right_shoulder_roll_joint',
  'right_shoulder_yaw_joint',
  'right_elbow_joint',
  'right_wrist_roll_joint',
  'right_wrist_pitch_joint',
  'right_wrist_yaw_joint',
]

# Left
# ik_joint = [
#     'waist_yaw_joint',
#     # 'waist_roll_joint', 
#     # 'waist_pitch_joint',
#     "left_shoulder_pitch_joint",
#     "left_shoulder_roll_joint",
#     "left_shoulder_yaw_joint",
#     "left_elbow_joint",
#     "left_wrist_roll_joint",
#     "left_wrist_pitch_joint",
#     "left_wrist_yaw_joint"
# ]

# Right
ik_joint = [
  # 'waist_yaw_joint',
  # 'waist_roll_joint',
  # 'waist_pitch_joint',
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint"
]


default_angles = [
  -0.2,  0.0,  0.0,  0.42, -0.23, 0.0, 
  -0.2,  0.0,  0.0,  0.42, -0.23, 0.0, 
  0., 0., 0.,
  0.0, 0.0, 0., 0.0, 0., 0., 0., 
  0.0, 0.0, 0., 0.0, 0., 0., 0., 
]

ik_to_motor = [motor_joint.index(j) for j in ik_joint]

ikctrl = IKCtrl(urdf_path="../../resources/robots/g1_description/g1_29dof_rev_1_0_ver4_camera_mount_v4.urdf",
       act_joints=ik_joint,
       frame="end_effector",
       fix_base=True,
)
ikctrl.robot.initViewer(loadModel=True)
# ikctrl.robot.display(q[lab_from_pin])
time.sleep(100)