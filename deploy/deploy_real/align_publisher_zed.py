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
from scipy.spatial.transform import Slerp
import torch
from typing import Literal, Mapping, Iterable
from functools import wraps

@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.clone().view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

def th2np(x):
    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, torch.Tensor):
        return x.numpy()

    if isinstance(x, Mapping):
        return {k: th2np(v) for (k,v) in x.items()}

    if isinstance(x, Iterable):
        return [th2np(e) for e in x]
    return x

def np2th(x):
    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)

    if isinstance(x, Mapping):
        return {k: np2th(v) for (k,v) in x.items()}

    if isinstance(x, Iterable):
        return [np2th(e) for e in x]

    return x

def as_np(func):
    @wraps(func)
    def np_func(*args, **kwds):
        th_args = np2th(args)
        th_kwds = np2th(kwds)
        th_out = func(*th_args, **th_kwds)
        # TODO(ycho): consider interoperable functions,
        # i.e., numpy in -> numpy out; torch in -> torch out
        np_out = th2np(th_out)
        return np_out
    return np_func

def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )

def quat_norm(q):
    """Normalizes a quaternion."""
    norm = np.linalg.norm(q)
    if norm == 0:
        return q
    return q / norm

yaw_quat = as_np(yaw_quat)
quat_apply = as_np(quat_apply)

import math_utils, math

axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_mul = as_np(math_utils.quat_mul)



# tag position
# 0   1
# 2   3
#
#   R
#

class NavAlignPublisher(Node):
    def __init__(self):
        super().__init__('gtworld_publisher')
        self.camera_frame = ''
        self.tags = ["tag_0", "tag_1", "tag_2", "tag_3"]
        self.mid_point = "tag_mid"
        self.world_frame = "fake_world"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.w2t = [None, None, None, None]
        self.l2t = [None, None, None, None]
        self.w2target = None
        self.w2n = None
        self.tf_broadcaster_w2m = StaticTransformBroadcaster(self)
        # self.tf_broadcaster = TransformBroadcaster(self)
        # self.tf_timer = self.camera_timer(0.1, self.tf_broadcaster)
        self.tf_timer_w2m = self.create_timer(1.0, self.publish_tf_w2m)

    def publish_tf_w2m(self):
        if self.w2n is None:
            try:
                use_zed = True
                if use_zed:
                    for i, tag in enumerate(self.tags):
                        cam_to_tag = self.tf_buffer.lookup_transform(
                            "zed_left_camera_optical_frame",
                            tag,
                            rclpy.time.Time()
                        )
                    
                        world_to_cam = self.tf_buffer.lookup_transform(
                            "fake_world",
                            "zed2i_left_camera_optical_frame",
                            rclpy.time.Time()
                        )

                        # Convert to numpy for multiplication
                        def tf_to_mat(tf):
                            t = tf.transform.translation
                            q = tf.transform.rotation
                            trans = np.array([t.x, t.y, t.z])
                            quat = np.array([q.w, q.x, q.y, q.z])  # wxyz
                            rot = R.from_quat(np.roll(quat, -1)).as_matrix()  # xyzw to wxyz
                            mat = np.eye(4)
                            mat[:3, :3] = rot
                            mat[:3, 3] = trans
                            return mat

                        def mat_to_tf(mat, parent, child):
                            t = TransformStamped()
                            t.header.stamp = self.get_clock().now().to_msg()
                            t.header.frame_id = parent
                            t.child_frame_id = child
                            t.transform.translation.x = mat[0, 3]
                            t.transform.translation.y = mat[1, 3]
                            t.transform.translation.z = mat[2, 3]
                            quat = R.from_matrix(mat[:3, :3]).as_quat()  # xyzw
                            quat = np.roll(quat, 1)  # wxyz
                            t.transform.rotation.w = quat[0]
                            t.transform.rotation.x = quat[1]
                            t.transform.rotation.y = quat[2]
                            t.transform.rotation.z = quat[3]
                            return t

                        w2c_mat = tf_to_mat(world_to_cam)
                        c2t_mat = tf_to_mat(cam_to_tag)
                        w2t_mat = w2c_mat @ c2t_mat
                        adjusted_tf = mat_to_tf(w2t_mat, self.world_frame, f"tag_adjusted_{i}")

                        # self.tf_broadcaster.publish(adjusted_tf)
                        self.w2t[i] = adjusted_tf
                        # Optionally, store or use w2t_mat as needed                
                else:
                    for i, tag in enumerate(self.tags):
                        self.w2t[i] = self.tf_buffer.lookup_transform(
                            self.world_frame,
                            tag,
                            rclpy.time.Time()
                        )
            except Exception as e:
                self.get_logger().error(f"Failed to publish static transform: {e}")
                return

            x = []
            y = []
            z = []
            quats = []
            for idx, tag in enumerate(self.w2t):
                x.append(self.w2t[idx].transform.translation.x)
                y.append(self.w2t[idx].transform.translation.y)
                z.append(self.w2t[idx].transform.translation.z)
                quats.append(
                    np.array([
                        self.w2t[idx].transform.rotation.w,
                        self.w2t[idx].transform.rotation.x,
                        self.w2t[idx].transform.rotation.y,
                        self.w2t[idx].transform.rotation.z,
                    ])
                )
            
            quat_0_1 = np.roll(Slerp([0,1], R.from_quat([np.roll(quats[0],-1), np.roll(quats[1],-1)]))(0.5).as_quat(),1)
            quat_2_3 = np.roll(Slerp([0,1], R.from_quat([np.roll(quats[2],-1), np.roll(quats[3],-1)]))(0.5).as_quat(),1)
            mid_quat = np.roll(Slerp([0,1], R.from_quat([np.roll(quat_0_1,-1), np.roll(quat_2_3,-1)]))(0.5).as_quat(),1)
            mid_quat = yaw_quat(mid_quat) # wxyz

            t_n = TransformStamped()
            t_n.header.stamp = self.get_clock().now().to_msg()
            t_n.header.frame_id = self.world_frame
            t_n.child_frame_id = "welding_line_center"
            t_n.transform.translation.x = np.mean(x) - 0.05
            t_n.transform.translation.y = np.mean(y)
            t_n.transform.translation.z = 0.3 #np.mean(z) - 0.5
            mid_quat = quat_mul(
                np.array([np.cos(math.pi/4), 0.0, 0.0, np.sin(math.pi/4)]).astype(np.float32),
                mid_quat.astype(np.float32)
            )

            t_n.transform.rotation.x = float(mid_quat[1])
            t_n.transform.rotation.y = float(mid_quat[2])
            t_n.transform.rotation.z = float(mid_quat[3])
            t_n.transform.rotation.w = float(mid_quat[0])

            self.w2n = t_n
            ####### TODO : Should add TF from welding obj <> nav target #######################################
            ### Applying relative TF might be easy
            t_target = TransformStamped()
            t_target.header.stamp = self.get_clock().now().to_msg()
            t_target.header.frame_id = "welding_line_center"
            t_target.child_frame_id = "nav_target"
            # t_target.transform.translation.x = -0.49950311
            # t_target.transform.translation.y = 0.16916784
            t_target.transform.translation.x = -1.3
            t_target.transform.translation.y = -0.3
            t_target.transform.translation.z = -t_n.transform.translation.z


            # Compute quaternion from yaw angle (in radians)
            def quaternion_from_yaw(yaw):
                # Returns quaternion in (x, y, z, w) order
                qx = 0.0
                qy = 0.0
                qz = np.sin(yaw / 2.0)
                qw = np.cos(yaw / 2.0)
                return np.array([qx, qy, qz, qw], dtype=np.float32)

            # Example: set yaw to 0 (or replace with your desired yaw angle)
            yaw_angle = 0.7085
            quat = quaternion_from_yaw(yaw_angle)

            # t_target.transform.rotation.x = float(quat[0])
            # t_target.transform.rotation.y = float(quat[1])
            # t_target.transform.rotation.z = float(quat[2])
            # t_target.transform.rotation.w = float(quat[3])

            t_target.transform.rotation.x = 0.
            t_target.transform.rotation.y = 0.
            t_target.transform.rotation.z = 0.
            t_target.transform.rotation.w = 1.

            self.w2target = t_target


        self.tf_broadcaster_w2m.sendTransform(self.w2n)
        self.tf_broadcaster_w2m.sendTransform(self.w2target)
        try:
            midsole_tf = self.tf_buffer.lookup_transform(
                            "nav_target",
                            "mid_sole_link",
                            rclpy.time.Time()
                        )
            
            translation_err = np.linalg.norm(np.array([
                midsole_tf.transform.translation.x,
                midsole_tf.transform.translation.y,
                0.0
                ]))
            

            
            rotation_err = axis_angle_from_quat(
                np.array(
                    [
                        midsole_tf.transform.rotation.w,
                        midsole_tf.transform.rotation.x,
                        midsole_tf.transform.rotation.y,
                        midsole_tf.transform.rotation.z,
                    ]
                )
            )
            print()
            
            print("translation error : ", translation_err)
            print("x : ", midsole_tf.transform.translation.x)
            print("y : ", midsole_tf.transform.translation.y)
            print("rotation error : ", rotation_err)
        except Exception as e:
            print(f"Error : {e}")
            
        
def main():


    rclpy.init()
    node = NavAlignPublisher()
    print("python launched")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()