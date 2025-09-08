import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys

G1_INPUT_IMAGE_LEFT = "input/left/color/000012.jpg"
G1_INPUT_IMAGE_RIGHT = "input/right/color/000012.jpg"
G1_INPUT_IMAGE_INTRINSIC = "input/left/intrinsic/000012.txt" # intrinsic from left

G1_TOPIC_LEFT_COLOR = "/zed/zed_node/left/image_rect_color"
G1_TOPIC_RIGHT_COLOR = "/zed/zed_node/right/image_rect_color"
G1_TOPIC_CAMERA_INFO = "/zed/zed_node/left/camera_info" # intrinsic from left



class PublishWeldpointsNode(Node):
    # get 2 x 3 numpy array for endpoints in the welding line obtained.
    def __init__(self):
        super().__init__("weldpoint_publish_node")
        self.TOPIC_NAME = "eetrack_vision/weldpoints"
        # assert points.shape == (2, 3), f"Array shape mismatch. Got {points.shape}. Expected (2, 3) shape array for welding line endpoints"
        self.points = None

        self.subscription = self.create_subscription(
            MultiArrayLayout,
            "eetrack_vision/weldpoints",
            self.zed_callback,
            10)

    def publish_weldpoints(self):
        try:
            self.points = np.load("/app/weldpoints.npy")
            assert self.points.shape == (2, 3), f"Array shape mismatch. Got {self.points.shape}. Expected (2, 3) shape array for welding line endpoints"
        except:
            self.get_logger().info(f"Points are not saved yet.") # just log msg.data for conciseness
            return
        # prepare msg data
        msg = Float64MultiArray()
        msg.data = list(self.points.flatten())

        row_dim = MultiArrayDimension()
        row_dim.label = "rows"
        row_dim.size = 2
        row_dim.stride = 3

        col_dim = MultiArrayDimension()
        col_dim.label = "columns"
        col_dim.size = 3
        col_dim.stride = 1

        msg.layout = MultiArrayLayout()
        msg.layout.dim = [row_dim, col_dim]
        msg.layout.data_offset = 0

        self.publisher.publish(msg)
        self.get_logger().info(f"Publishing {self.TOPIC_NAME}: {msg}") # just log msg.data for conciseness

def get_zed_images(args=None):
    rclpy.init(args=args)
    for topic, file, in zip([G1_TOPIC_LEFT_COLOR, G1_TOPIC_RIGHT_COLOR, G1_TOPIC_CAMERA_INFO], [G1_INPUT_IMAGE_LEFT, G1_INPUT_IMAGE_RIGHT, G1_INPUT_IMAGE_INTRINSIC]):
        node = ZedCaptureNode(topic, file)
        for _ in range(10):
            rclpy.spin_once(node)
        node.destroy_node()
    rclpy.shutdown()

def publish_weldpoints(args=None):
    rclpy.init(args=args)
    node = PublishWeldpointsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # default: get zed images and info (left, right, and intrinsic K)
        get_zed_images()
    else:
        if sys.argv[1] == "pub_weld":
            publish_weldpoints()
