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

@contextmanager
def with_dir(d):
    d0 = os.getcwd()
    try:
        os.chdir(d)
        yield
    finally:
        os.chdir(d0)

class Robot:
    def __init__(self,
                 urdf_path: str):
        path = Path(urdf_path)
        with with_dir(path.parent):
            self.robot = pin.RobotWrapper.BuildFromURDF(filename=path.name,
                                                   package_dirs=["."],
                                                   root_joint=None)
            joint_names = list(self.robot.model.names)
            assert (joint_names[0] == 'universe')
            self.joint_names = joint_names[1:]