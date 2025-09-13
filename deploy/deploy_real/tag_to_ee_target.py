import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import math
import time


class FramePublisher(Node):
    def __init__(self):
        super().__init__('frame_publisher')

        # Create TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        self.tag_names = ["tag_0", "tag_1", "tag_2", "tag_3", "tag_arm"]

        # Timer to call broadcast function
        self.timer = self.create_timer(0.01, self.broadcast_timer_callback)

    def broadcast_timer_callback(self):
        for tag_name in self.tag_names:
            t = TransformStamped()

            # Header
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = tag_name       # parent frame
            t.child_frame_id = tag_name.replace("tag", "target")   # child frame

            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0

            if "arm" not in tag_name:
                t.transform.rotation.x = -0.5
                t.transform.rotation.y = 0.5
                t.transform.rotation.z = 0.5
                t.transform.rotation.w = 0.5
            else:
                #0, 0, 0.7071068, 0.7071068
                t.transform.rotation.x = 0.
                t.transform.rotation.y = 0.
                t.transform.rotation.z = 0.7071068
                t.transform.rotation.w = 0.7071068

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