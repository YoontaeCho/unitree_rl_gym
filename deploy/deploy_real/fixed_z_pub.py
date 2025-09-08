#!/usr/bin/env python3

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from unitree_hg.msg import LowState as LowStateHG
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped, StaticTransformBroadcaster

import numpy as np
import yaml
from geometry_msgs.msg import Vector3, Quaternion, Point
from nav_msgs.msg import Odometry
from unitree_go.msg import SportModeState

from scipy.spatial.transform import Rotation as R

import pinocchio as pin
from icecream import ic
from common.np_math import (index_map, with_dir)

import math_utils
axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
import pink

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

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * q_w[..., None] * 2.0
    c = q_vec * np.einsum("...i,...i->...", q_vec, v)[..., None] * 2.0
    return a + b + c

def to_array(v):
    if isinstance(v, Vector3) or isinstance(v, Point):
        return np.array([v.x, v.y, v.z], dtype=np.float32)
    elif isinstance(v, Quaternion):
        return np.array([v.x, v.y, v.z, v.w], dtype=np.float32)

class PelvistoTrack(Node):
    def __init__(self):
        super().__init__('pelvis_track_publisher')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.low_state = LowStateHG()
        self.low_state_subscriber = self.create_subscription(
            LowStateHG,
            'lowstate',
            self.on_low_state,
            10)
        
        urdf_path = '../../resources/robots/g1_description/g1_29dof_rev_1_0_ver4.urdf'
        path = Path(urdf_path)
        with with_dir(path.parent):
            robot = pin.RobotWrapper.BuildFromURDF(filename=path.name,
                                                   package_dirs=["."],
                                                   root_joint=None)
            self.robot = robot

        pin_joint = self.robot.model.names[1:]
        with open('./configs/ik.yaml', 'r') as fp:
            motor_joint = yaml.safe_load(fp)['motor_joint']

        self.pin_from_mot = index_map(
            pin_joint, motor_joint
        )
        
    def on_low_state(self,
                     msg: LowStateHG):
        self.low_state = msg
        try:
            map_from_world = self.tf_buffer.lookup_transform(
                    'world', 
                    'base_link',
                    rclpy.time.Time(),
                )
                
        except Exception as ex:
            print(f'Could not transform map to base_link as world to map is yet published: {ex}')
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link_z_fk'
        # print("X : ", map_from_world.transform.translation.x)
        # print("Y : ", map_from_world.transform.translation.y)

        z_value, rot = self.lidar_height_rot(self.low_state)

        try:
            to_zed_from_midsole = self.tf_buffer.lookup_transform(
                    'mid_sole_link', 
                    'zed2i_base_link',
                    rclpy.time.Time(),
                )
            z_value  = to_zed_from_midsole.transform.translation.z
            print("MID SOLE FOUND")
        except:
            z_value, _ = self.lidar_height_rot(self.low_state)
            
        t.transform.translation.x = map_from_world.transform.translation.x
        t.transform.translation.y = map_from_world.transform.translation.y
        ####################### Use Height from FK #######################
        t.transform.translation.z = z_value

        if True:
            ####################### Use rotation from cam #######################
            t.transform.rotation.x = map_from_world.transform.rotation.x
            t.transform.rotation.y = map_from_world.transform.rotation.y
            t.transform.rotation.z = map_from_world.transform.rotation.z
            t.transform.rotation.w = map_from_world.transform.rotation.w
        elif False:
            ####################### Use rotation from FK #######################
            t.transform.rotation.x = float(rot[0])
            t.transform.rotation.y = float(rot[1])
            t.transform.rotation.z = float(rot[2])
            t.transform.rotation.w = float(rot[3])
        else:
            ####################### Use Y from cam, RP from FK #######################
            # Sit policy explodes!! NO use. just concept.
            rpy_rot = R.from_quat(rot).as_euler("xyz")   # roll, pitch, yaw from FK
            rpy_map = R.from_quat([
                map_from_world.transform.rotation.x,
                map_from_world.transform.rotation.y,
                map_from_world.transform.rotation.z,
                map_from_world.transform.rotation.w,
            ]).as_euler("xyz")
            combined_rpy = [rpy_rot[0],  # roll from rot
                    rpy_rot[1],  # pitch from rot
                    rpy_map[2]]  # yaw from map_from_world
            combined_quat = R.from_euler("xyz", combined_rpy).as_quat()

            t.transform.rotation.x = float(combined_quat[0])
            t.transform.rotation.y = float(combined_quat[1])
            t.transform.rotation.z = float(combined_quat[2])
            t.transform.rotation.w = float(combined_quat[3])



        self.tf_broadcaster.sendTransform(t)

        
    def lidar_height_rot(self, low_state: LowStateHG):
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

        r = R.from_quat(np.roll(world_from_pelvis_quat, -1))
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        r_noYaw = R.from_euler("xyz", [roll + np.pi, pitch, 0.0], degrees=False)
        world_from_pelvis_quat_noYaw = r_noYaw.as_quat()
        # print(xyz_rf, xyz_lf)
        pelvis_z_rf = -quat_rotate(
            world_from_pelvis_quat_noYaw, xyz_rf)[2] + 0.02 #0.028531
        pelvis_z_lf = -quat_rotate(
            world_from_pelvis_quat_noYaw, xyz_lf)[2] + 0.02 #0.028531
        # print(xyz_lf)
        pelvis_from_lidar = self.tf_buffer.lookup_transform('pelvis',
                    'zed2i_base_link', rclpy.time.Time())
        
        lidar_z_pevlis = quat_rotate(world_from_pelvis_quat_noYaw,
            to_array(pelvis_from_lidar.transform.translation))[2]
        lidar_rot = (R.from_quat(np.roll(world_from_pelvis_quat_noYaw, -1)) *
                    R.from_quat(to_array(pelvis_from_lidar.transform.rotation)))
        
        # print(pelvis_from_lidar.transform.translation,
        # 0.5 * pelvis_z_lf + 0.5 * pelvis_z_rf,
        # lidar_z_pevlis
        # )
        return (0.5 * pelvis_z_lf + 0.5 * pelvis_z_rf + lidar_z_pevlis,
                    lidar_rot.as_quat())



def main():
    rclpy.init()
    node = PelvistoTrack()
    # executor = MultiThreadedExecutor(num_threads=8)
    # executor.add_node(node)
    try:
        # executor.spin()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()