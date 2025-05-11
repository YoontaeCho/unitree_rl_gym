import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy as rp
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped

"""
Before running this script,
1. Ensure realsense camera node is running.
ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=1280x720x30
ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=1280x720x30 pointcloud.enable:=true

2. Ensure apriltag node is running.
ros2 launch tag_realsense.launch.py camera:=/camera/camera/color
"""

class EEtrackGoal(Node):
    def __init__(self):
        super().__init__('eetrack_goal')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # xyzrpy [m, rad]
        # TODO: Read from a config file
        self.tag_pose_to_eetrack_start = {
            6: np.array([0.0, 0.01, 0.0, 0.0, -np.pi/4, np.pi])
        }
        self.tag_pose_to_eetrack_end = {
            7: np.array([0.0, -0.01, 0.0, 0.0, -np.pi/4, np.pi])
        }

        self.tf_timer = self.create_timer(0.01, self.publish_hand_target)

    def publish_hand_target(self):
        cur_time = self.get_clock().now().to_msg()
        # Publish the eetrack start goal
        # TODO: If there is multiple tag2goal transforms, average the position and orientation
        for tag_id, transform in self.tag_pose_to_eetrack_start.items():
            t = TransformStamped()

            # Format header
            t.header.stamp = cur_time
            t.header.frame_id = f'tag{tag_id}'
            t.child_frame_id = 'eetrack_start'

            t.transform.translation.x = transform[0]
            t.transform.translation.y = transform[1]
            t.transform.translation.z = transform[2]

            quat = R.from_euler('xyz', transform[3:6]).as_quat()
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(t)

        # Publish the eetrack end goal
        for tag_id, transform in self.tag_pose_to_eetrack_end.items():
            t = TransformStamped()

            # Format header
            t.header.stamp = cur_time
            t.header.frame_id = f'tag{tag_id}'
            t.child_frame_id = 'eetrack_end'

            t.transform.translation.x = transform[0]
            t.transform.translation.y = transform[1]
            t.transform.translation.z = transform[2]

            quat = R.from_euler('xyz', transform[3:6]).as_quat()
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(t)

if __name__ == '__main__':
    rp.init()
    eetrack_goal = EEtrackGoal()
    try:
        rp.spin(eetrack_goal)
    except KeyboardInterrupt:
        pass
    finally:
        eetrack_goal.destroy_node()
        rp.shutdown()
