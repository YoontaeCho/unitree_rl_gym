#!/usr/bin/env python3

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from unitree_hg.msg import LowState as LowStateHG
from tf2_ros import TransformBroadcaster, TransformStamped

import numpy as np
import pinocchio as pin
import pink
import yaml
from common.np_math import (index_map, with_dir)
from math_utils import (as_np, quat_rotate, yaw_quat, quat_mul, quat_inv)

quat_rotate = as_np(quat_rotate)
yaw_quat = as_np(yaw_quat)
quat_inv = as_np(quat_inv)
quat_mul = as_np(quat_mul)


class FakeWorldPublisher(Node):
    def __init__(self):
        super().__init__('fake_world_publisher')

        #urdf_path = '../../resources/robots/g1_description/g1_29dof_rev_1_0_ver4.urdf'
        urdf_path = '../../resources/robots/g1_description/g1_29dof_rev_1_0_d435_with_welder_v2.urdf'
        #urdf_path = '../../resources/robots/g1_description/g1_29dof_rev_1_0_tag_calibrated_camera_pose_fix_welder.urdf'
        path = Path(urdf_path)
        with with_dir(path.parent):
            robot = pin.RobotWrapper.BuildFromURDF(filename=path.name,
                                                   package_dirs=["."],
                                                   root_joint=None)
            self.robot = robot

        self.low_state = LowStateHG()
        self.low_state_subscriber = self.create_subscription(
            LowStateHG,
            'lowstate',
            self.on_low_state,
            10)
        self.tf_broadcaster = TransformBroadcaster(self)

        pin_joint = self.robot.model.names[1:]
        with open('./configs/ik.yaml', 'r') as fp:
            motor_joint = yaml.safe_load(fp)['motor_joint']

        self.pin_from_mot = index_map(
            pin_joint, motor_joint
        )

    def on_low_state(self,
                     msg: LowStateHG):
        # BK:
        # this function publishes pelvis height and rotation
        # it seems fake_world_tf = pelvis orientation + (0,0,pelvis_height)?
        # Why is this so? I thought this was the tf between the feet?

        self.low_state = msg

        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'fake_world'
        t.child_frame_id = 'pelvis'

        # 
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = self.pelvis_height(self.low_state)

        # Set world_from_pelvis quaternion based on IMU state
        qw, qx, qy, qz = [
            float(x) for x in 
            self.low_state.imu_state.quaternion
        ]
        imu_quat = np.array([qw, qx, qy, qz])
        imu_quat = quat_mul(quat_inv(yaw_quat(imu_quat)), imu_quat)
        t.transform.rotation.x = imu_quat[1]
        t.transform.rotation.y = imu_quat[2]
        t.transform.rotation.z = imu_quat[3]
        t.transform.rotation.w = imu_quat[0]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'fake_world'

        # 
        t.transform.translation.x = 0.5
        t.transform.translation.y = 0.5
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def pelvis_height(self, low_state: LowStateHG):
        robot = self.robot

        # gets joint angles
        q_mot = [self.low_state.motor_state[i_mot].q for i_mot in range(29)]

        # not sure what this does
        q_pin = np.zeros_like(self.robot.q0)
        q_pin[self.pin_from_mot] = q_mot
        cfg = pink.Configuration(robot.model, robot.data, q_pin)

        # get ankle poses in world frame using forward kinematics
        pelvis_from_rf = cfg.get_transform_frame_to_world(
            'right_ankle_roll_link')
        pelvis_from_lf = cfg.get_transform_frame_to_world(
            'left_ankle_roll_link')


        # get ankle translations in world frame
        xyz_rf = np.asarray( pelvis_from_rf.translation,
                            dtype=np.float32)
        xyz_lf = np.asarray( pelvis_from_lf.translation,
                            dtype=np.float32)

        # get orientation of pelvis from IMU in world frame
        world_from_pelvis_quat = np.asarray(low_state.imu_state.quaternion,
                                            dtype=np.float32)

        # rotate ankle world vector by the pelvis orientation
        pelvis_z_rf = -quat_rotate(
            world_from_pelvis_quat, xyz_rf)[2] + 0.02 # (ycho): 0.02 = approx "roll_link" height 
        pelvis_z_lf = -quat_rotate(
            world_from_pelvis_quat, xyz_lf)[2] + 0.02

        # pelvis height = avg of vertical distance from pelvis to both feet
        return 0.5 * pelvis_z_lf + 0.5 * pelvis_z_rf


def main():
    rclpy.init()
    node = FakeWorldPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
