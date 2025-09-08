import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point32, PoseStamped
from sensor_msgs.msg import PointCloud

# import jaxlie

class SitBasePosePublisher(Node):
    def __init__(self):
        super().__init__("sit_base_pose_publisher")

        self.target_sit_base_pose_publisher = self.create_publisher(
            PoseStamped,
            "target_sit_base_pose",
            10,
        )

        # Temporally hard-coded x, y, yaw.
        self.sit_base_xyyaw_wrt_object_pose = np.array([-0.49950311, 0.16916784, 0.7085])
        # Welding line subscription
        self.create_subscription(
            PointCloud,
            "welding_line",
            self.pub_target_sit_base_pose,
            10,
        )

        # Temporal welding line publisher
        self.pusblish_welding_line_tmp()


    def pub_target_sit_base_pose(self, msg: PointCloud):
        points = msg.points

        len_points = len(points)
        start_point = points[0]
        mid_point = points[len_points//2]
        end_point = points[-1]

        start_point_xyz = np.array([start_point.x, start_point.y, start_point.z])
        mid_point_xyz = np.array([mid_point.x, mid_point.y, mid_point.z])
        end_point_xyz = np.array([end_point.x, end_point.y, end_point.z])

        # Assume the point is in world frame.
        object_z_axis = np.array([0,0,1])
        # Assume the start point is in left (+y) and the end point is in right (-y)
        object_y_axis = start_point_xyz - end_point_xyz
        object_y_axis = object_y_axis / np.linalg.norm(object_y_axis)
        object_x_axis = np.cross(object_y_axis, object_z_axis)
        object_T = np.stack([object_x_axis, object_y_axis, object_z_axis], axis=1)

        object_R = R.from_matrix(object_T)
        object_pos = mid_point_xyz + 0.0025*object_x_axis - 0.005*object_z_axis

        x_mean, y_mean, yaw_mean = self.sit_base_xyyaw_wrt_object_pose
        sit_base_pos = object_pos + x_mean*object_x_axis + y_mean*object_y_axis
        sit_base_pos[2] = 0.0
        sit_base_quat = (object_R * R.from_euler('z', yaw_mean)).as_quat()

        pub_msg = PoseStamped()

        pub_msg.header.stamp = self.get_clock().now().to_msg()
        pub_msg.header.frame_id = msg.header.frame_id

        pub_msg.pose.position.x = sit_base_pos[0]
        pub_msg.pose.position.y = sit_base_pos[1]
        pub_msg.pose.position.z = sit_base_pos[2]

        pub_msg.pose.orientation.x = sit_base_quat[0]
        pub_msg.pose.orientation.y = sit_base_quat[1]
        pub_msg.pose.orientation.z = sit_base_quat[2]
        pub_msg.pose.orientation.w = sit_base_quat[3]

        self.target_sit_base_pose_publisher.publish(pub_msg)


    def pusblish_welding_line_tmp(self):
        welding_path = np.load("eetrack/welding_path.npy")
        points = welding_path[:,:3]

        self.tmp_welding_line_publisher = self.create_publisher(
            PointCloud,
            "welding_line",
            10,
        )

        msg = PointCloud()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        for point in points:
            point_msg = Point32()
            point_msg.x = point[0]
            point_msg.y = point[1]
            point_msg.z = point[2]
            msg.points.append(point_msg)

        self.tmp_welding_line_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    sit_base_pose_pub = SitBasePosePublisher()
    try:
        rclpy.spin(sit_base_pose_pub)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()