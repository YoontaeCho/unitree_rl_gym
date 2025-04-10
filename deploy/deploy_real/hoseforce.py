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


@contextmanager
def with_dir(d):
    d0 = os.getcwd()
    try:
        os.chdir(d)
        yield
    finally:
        os.chdir(d0)


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


class HoseForceEstimator:
    def __init__(self,
                 urdf_path: str,
                 end_effector: str,
                 config):
        path = Path(urdf_path)
        with with_dir(path.parent):
            self.robot = pin.RobotWrapper.BuildFromURDF(filename=path.name,
                                                        package_dirs=["."],
                                                        root_joint=None)
        joint_names = list(self.robot.model.names)
        assert (joint_names[0] == 'universe')
        self.joint_names = joint_names[1:]
        
        
        self.config = config
        
        self.pin_from_mot = index_map(
            self.joint_names,
            self.config.motor_joint
            )
        
        self.pin_from_arm = index_map(
            self.joint_names,
            self.config.arm_joint
            )
        
        self.mot_from_arm = index_map(
            self.config.motor_joint,
            self.config.arm_joint
            )
        
        self.eef_id = self.robot.model.getFrameId(end_effector)
        if self.eef_id < 0:
            raise ValueError(f"Frame {end_effector} not found in URDF.")

    def compute_force(self, q_mot, tau_mot):
        q_pin = np.zeros(len(self.joint_names))
        q_pin[self.pin_from_mot] = q_mot
        
        data = self.robot.data
        pin.computeGeneralizedGravity(self.robot.model, data, q_pin)
        tau_gravity = data.g[self.pin_from_mot]
        tau_residual = tau_mot - tau_gravity
        tau_residual_arm = tau_residual[self.mot_from_arm]
        
        pin.computeJointJacobians(self.robot.model, data, q_pin)
        pin.updateFramePlacements(self.robot.model, data, self.eef_id)
        J = pin.getFrameJacobian(
            self.robot.model,
            data,
            self.eef_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_arm = J[:, self.pin_from_arm]
        
        F_eef, _, _, _ = np.linalg.lstsq(J.T, tau_residual_arm, rcond=None)
        
        return F_eef