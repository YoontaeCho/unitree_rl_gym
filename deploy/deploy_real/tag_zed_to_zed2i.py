import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer,StaticTransformBroadcaster, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import math
import time


class FramePublisher(Node):
    def __init__(self):
        super().__init__('frame_publisher')

        # Create TransformBroadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        

        self.tag_names = ["tag_0", "tag_1", "tag_2", "tag_3", "tag_arm", "tag_10", "tag_11", "tag_12"]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to call broadcast function
        self.timer = self.create_timer(0.01, self.broadcast_timer_callback)

    def broadcast_timer_callback(self):
        for tag_name in self.tag_names:
            try:
                t = self.tf_buffer.lookup_transform(
                    "zed_left_camera_optical_frame",
                    tag_name,
                    rclpy.time.Time(),
                )
            except:
                continue

            # Header
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = t.header.frame_id.replace("zed", "zed2i")       # parent frame
            t.child_frame_id = tag_name + "_from_opt_frame"   # child frame

            # Publish transform
            self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = FramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()