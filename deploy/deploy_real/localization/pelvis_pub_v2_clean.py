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
from lpf import ActionFilterButter 
from icecream import ic

import math_utils
axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)

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

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # # Timer for dynamic transform broadcasting (e.g., pelvis tracking)
        # self.timer = self.create_timer(0.01, self.on_timer)
        # # One-shot timer to check & publish the static transform after a short delay
        self.static_tf_timer = self.create_timer(1.0, self.publish_static_tf)

        self._pos_lpf_filter = ActionFilterButter(lowcut=np.zeros(1*3)*2,
                                    highcut=np.ones(1*3) * 6.0, 
                                    sampling_rate=100.0, num_joints=1*3)
        self._rot_lpf_filter = ActionFilterButter(lowcut=np.zeros(1*3)*2,
                                    highcut=np.ones(1*3) * 6.0, 
                                    sampling_rate=100.0, num_joints=1*3)
    

    def on_low_state(self,
                     msg: LowStateHG):
        self.low_state = msg
        try:
            imu_from_pelvis_tf = self.tf_buffer.lookup_transform(
                    'mid360_link_IMU', 
                    'pelvis',
                    # 'zed2_camera_center',
                    rclpy.time.Time(),
                    # rclpy.duration.Duration(seconds=0.05)
                )
        except Exception as ex:
            print(f'Could not transform mid360_link_IMU to pelvis as world to camera_init is yet published: {ex}')
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'body'
        t.child_frame_id = 'pelvis'

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = imu_from_pelvis_tf.transform.translation.x
        t.transform.translation.y = imu_from_pelvis_tf.transform.translation.y
        t.transform.translation.z = imu_from_pelvis_tf.transform.translation.z

        t.transform.rotation.x = imu_from_pelvis_tf.transform.rotation.x
        t.transform.rotation.y = imu_from_pelvis_tf.transform.rotation.y
        t.transform.rotation.z = imu_from_pelvis_tf.transform.rotation.z
        t.transform.rotation.w = imu_from_pelvis_tf.transform.rotation.w

        self.tf_broadcaster.sendTransform(t)

    def publish_static_tf(self):
        """Check if a static transform from 'world' to 'camera_init' exists.
        If not, publish it using the parameter 'camera_init_z' for the z-value.
        This method is designed to run only once.
        """
        # Cancel the timer so this callback runs only one time.
        if self.low_state.crc == 0:
            return
        self.static_tf_timer.cancel()

        try:
            # Try to look up an existing transform from "world" to "camera_init".
            # Here, rclpy.time.Time() (i.e. time=0) means "the latest available".
            self.tf_buffer.lookup_transform(
                "world", "camera_init", rclpy.time.Time()
            )
            self.get_logger().info(
                "Static transform from 'world' to 'camera_init' already exists. Not publishing a new one."
            )
        except Exception as ex:
            # If the transform isn't found, declare (or get) the parameter for z and publish the static transform.
            z_value, rot = self.lidar_height_rot(self.low_state)
            static_tf = TransformStamped()
            static_tf.header.stamp = self.get_clock().now().to_msg()
            static_tf.header.frame_id = "world"
            static_tf.child_frame_id = "camera_init"
            # static_tf.child_frame_id = "pelvis"

            static_tf.transform.translation.x = 0.0
            static_tf.transform.translation.y = 0.0
            static_tf.transform.translation.z = z_value
            static_tf.transform.rotation.x = float(rot[0])
            static_tf.transform.rotation.y = float(rot[1])
            static_tf.transform.rotation.z = float(rot[2])
            static_tf.transform.rotation.w = float(rot[3])

            self.static_tf_broadcaster.sendTransform(static_tf)
            self.get_logger().info(
                f"Published static transform from 'world' to 'camera_init' with z = {z_value} quat = {rot}"
            )

    def lidar_height_rot(self, low_state: LowStateHG):
        print(self.tf_buffer.lookup_transform('pelvis',
                    'left_ankle_roll_link', rclpy.time.Time()))

        world_from_pelvis_quat = np.asarray(low_state.imu_state.quaternion,
                                            dtype=np.float32)
        
        # ic(world_from_pelvis_quat)
        # ic(axis_angle_from_quat(world_from_pelvis_quat))
        r = R.from_quat(np.roll(world_from_pelvis_quat, -1))
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        r_noYaw = R.from_euler("xyz", [roll + np.pi, pitch, 0.0], degrees=False)
        world_from_pelvis_quat_noYaw = r_noYaw.as_quat()


        pelvis_from_rf = self.tf_buffer.lookup_transform('pelvis',
                    'right_ankle_roll_link', rclpy.time.Time())
        pelvis_from_lf = self.tf_buffer.lookup_transform('pelvis',
                    'left_ankle_roll_link', rclpy.time.Time())  
        xyz_rf = to_array(pelvis_from_rf.transform.translation) 
        xyz_lf = to_array(pelvis_from_lf.transform.translation) 

        # print(xyz_rf, xyz_lf)
        pelvis_z_rf = -quat_rotate(
            world_from_pelvis_quat_noYaw, xyz_rf)[2] + 0.028531
        pelvis_z_lf = -quat_rotate(
            world_from_pelvis_quat_noYaw, xyz_lf)[2] + 0.028531
        # print(xyz_lf)
        pelvis_from_lidar = self.tf_buffer.lookup_transform('pelvis',
                    'mid360_link_frame', rclpy.time.Time())
        
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()