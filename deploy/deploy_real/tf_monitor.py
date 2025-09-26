import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
import time

class TFHzMonitor(Node):
    def __init__(self, parent_frame, child_frame):
        super().__init__('tf_hz_monitor')
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.callback,
            10
        )
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.last_time = None
        self.count = 0

    def callback(self, msg):
        for transform in msg.transforms:
            # print(transform.header.frame_id)
            # print(transform.child_frame_id)
            # print()
            if (transform.header.frame_id == self.parent_frame and
                transform.child_frame_id == self.child_frame):
                now = time.time()
                if self.last_time is not None:
                    dt = now - self.last_time
                    hz = 1.0 / dt
                    self.get_logger().info(f"{self.parent_frame}->{self.child_frame} : {hz:.2f} Hz")
                self.last_time = now

def main():
    rclpy.init()
    node = TFHzMonitor('right_shoulder_pitch_link', 'right_shoulder_roll_link')  # change frames here
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()