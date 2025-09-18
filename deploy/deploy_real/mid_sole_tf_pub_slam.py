#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from unitree_hg.msg import LowState as LowStateHG
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped, TransformException
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import pinocchio as pin
import pink
import yaml
from common.np_math import (index_map, with_dir)
from math_utils import (as_np, quat_rotate, axis_angle_from_quat, wrap_to_pi, yaw_quat)
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

quat_rotate = as_np(quat_rotate)
axis_angle_from_quat = as_np(axis_angle_from_quat)
wrap_to_pi = as_np(wrap_to_pi)
yaw_quat = as_np(yaw_quat)


def body_pose(
        tf_buffer,
        frame: str,
        ref_frame: str = 'pelvis',
        stamp=None,
        rot_type: str = 'axa'):
    """ --> tf does not exist """
    if stamp is None:
        stamp = rclpy.time.Time()
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


class MidSoleTFPublisher(Node):
    def __init__(self):
        super().__init__('mid_sole_tf_publisher')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = TransformBroadcaster(self)

        time.sleep(5.0)

        self.low_state = LowStateHG()
        self.low_state_subscriber = self.create_subscription(
            LowStateHG,
            'lowstate',
            self.on_low_state,
            10)

    def on_low_state(self,
                     msg: LowStateHG):
        self.low_state = msg

        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'pelvis'
        t.child_frame_id = 'mid_sole_link'

        left_sole_pos, left_sole_quat = body_pose(
            self.tf_buffer,
            'left_sole_link',
            'pelvis',
            rot_type='quat'
        )
        right_sole_pos, right_sole_quat = body_pose(
            self.tf_buffer,
            'right_sole_link',
            'pelvis',
            rot_type='quat'
        )

        mid_sole_pos = (left_sole_pos + right_sole_pos) / 2.0
        mid_sole_quat = np.roll(Slerp([0,1], R.from_quat([np.roll(left_sole_quat,-1), np.roll(right_sole_quat,-1)]))(0.5).as_quat(),1)
        # mid_sole_quat = yaw_quat(mid_sole_quat)

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = mid_sole_pos[0]
        t.transform.translation.y = mid_sole_pos[1]
        t.transform.translation.z = mid_sole_pos[2]

        t.transform.rotation.w = mid_sole_quat[0]
        t.transform.rotation.x = mid_sole_quat[1]
        t.transform.rotation.y = mid_sole_quat[2]
        t.transform.rotation.z = mid_sole_quat[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

        try:
            left_sole_pos, left_sole_quat = body_pose(
                self.tf_buffer,
                'mid360_link_IMU',
                'left_sole_link',
                rot_type='quat'
            )
            right_sole_pos, right_sole_quat = body_pose(
                self.tf_buffer,
                'mid360_link_IMU',
                'right_sole_link',
                rot_type='quat'
            )

            mid_sole_pos = (left_sole_pos + right_sole_pos) / 2.0

            body_pos_from_world, body_quat_from_world = body_pose(
                self.tf_buffer,
                'body',
                'world',
                rot_type='quat'
            )

            body_tf_from_world = TransformStamped()
           
            body_tf_from_world.header.stamp = self.get_clock().now().to_msg()
            body_tf_from_world.header.frame_id = "body"
            body_tf_from_world.child_frame_id = "body_z_from_mid_sole_link"


            body_tf_from_world.transform.translation.z = body_pos_from_world[2] - mid_sole_pos[2]


            self.tf_broadcaster.sendTransform(body_tf_from_world)

        except Exception as e:
            self.get_logger().info(
                "Transform connection betwwen body <> world is not found yet."
            )

        # if True:

        #     t = TransformStamped()

        #     # Read message content and assign it to
        #     # corresponding tf variables
        #     t.header.stamp = self.get_clock().now().to_msg()
        #     t.header.frame_id = 'world'
        #     t.child_frame_id = 'fake_world'

        #     try:
        #         pelvis_tf = self.tf_buffer.lookup_transform(
        #             'world', 
        #             'pelvis',
        #             rclpy.time.Time(),
        #         )
        #     except:
        #         return

        #     # 
        #     t.transform.translation.x = pelvis_tf.transform.translation.x
        #     t.transform.translation.y = pelvis_tf.transform.translation.y
        #     t.transform.translation.z = 0.

        #     qx = pelvis_tf.transform.rotation.x
        #     qy = pelvis_tf.transform.rotation.y
        #     qz = pelvis_tf.transform.rotation.z
        #     qw = pelvis_tf.transform.rotation.w

        #     # # Normalize (defensive) then invert
        #     # r_pelvis = R.from_quat([qx, qy, qz, qw])
        #     # r_fake = r_pelvis.inv()                # exact inverse (no yaw removal)

        #     # qx, qy, qz, qw = r_fake.as_quat()

        #     qw, qx, qy, qz = yaw_quat(np.array([qw, qx, qy, qz]))
        #     t.transform.rotation.x = qx
        #     t.transform.rotation.y = qy
        #     t.transform.rotation.z = qz
        #     t.transform.rotation.w = qw

        #     # Send the transformation
        #     self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = MidSoleTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()