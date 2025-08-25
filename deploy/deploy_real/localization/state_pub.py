from math import sin, cos, pi
import os
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class StatePublisher(Node):

    def __init__(self):
        super().__init__('state_publisher')
        qos_profile = QoSProfile(depth=10)
        # Preferred: provide a path to YAML; fallback to direct 'joint_names' parameter
        self.declare_parameter('config_yaml_path', '')
        self.declare_parameter(
            'joint_names',
            value=['__default__'],
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY),
        )
        # yaml_path: str = self.get_parameter('config_yaml_path').get_parameter_value().string_value
        joint_names_param = self.get_parameter('joint_names').get_parameter_value().string_array_value
        self.joint_names = []
        yaml_path = '/root/elevation_mapping_cupy/elevation_mapping_cupy/config/setups/g1/g1_state_pub.yaml'
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, 'r') as fp:
                cfg = yaml.safe_load(fp) 
                self.joint_names = cfg['state_pub']['ros__parameters']['joint_names']
        #     try:
        #         with open(yaml_path, 'r') as fp:
        #             cfg = yaml.safe_load(fp) or {}
        #         params = cfg.get('state_pub', {}).get('ros__parameters', cfg)
        #         if isinstance(params, dict) and 'joint_names' in params:
        #             names = params['joint_names']
        #             if isinstance(names, list):
        #                 self.joint_names = [str(n) for n in names]
        #     except Exception as exc:
        #         self.get_logger().warn(f"Failed to parse YAML at '{yaml_path}': {exc}")
        # if not self.joint_names and joint_names_param and joint_names_param[0] != '__default__':
        #     self.joint_names = list(joint_names_param)

        self.low_state = LowStateHG()
        self.low_state_subscriber = self.create_subscription(LowStateHG,
                    'lowstate', self.on_low_state, 10)
        self.joint_pub = self.create_publisher(JointState,
                                               'joint_states', qos_profile)
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))
        self.joint_state = JointState()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

    def on_low_state(self, msg: LowStateHG):
        self.low_state = msg
        joint_state = self.joint_state
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            joint_state.header.stamp = msg.header.stamp
        else:
            joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names
        joint_state.position = [0.0 for _ in self.joint_names]
        joint_state.velocity = [0.0 for _ in self.joint_names]
        
        n:int = min(len(self.joint_names), len(self.low_state.motor_state))
        for i in range(n):
            joint_state.position[i] = self.low_state.motor_state[i].q
            joint_state.velocity[i] = self.low_state.motor_state[i].dq
        # print(joint_state)
        self.joint_pub.publish(joint_state)
        try:
            imu_to_pelvis = self.tf_buffer.lookup_transform('mid360_link', 
                    'pelvis',
                    rclpy.time.Time())
            t = TransformStamped()
            t.header.stamp = imu_to_pelvis.header.stamp
            t.header.frame_id = 'body'
            t.child_frame_id = 'pelvis'
            t.transform.translation.x = imu_to_pelvis.transform.translation.x
            t.transform.translation.y = imu_to_pelvis.transform.translation.y
            t.transform.translation.z = imu_to_pelvis.transform.translation.z

            t.transform.rotation.x = imu_to_pelvis.transform.rotation.x
            t.transform.rotation.y = imu_to_pelvis.transform.rotation.y
            t.transform.rotation.z = imu_to_pelvis.transform.rotation.z
            t.transform.rotation.w = imu_to_pelvis.transform.rotation.w
            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().info(f"Error looking up transform from mid360_link_IMU to pelvis: {e}")
    
    def run(self):
        # loop_rate = self.create_rate(10)
        # try:
        #     # rclpy.spin()
        #     while rclpy.ok():
        #         rclpy.spin_once(self)
        #         loop_rate.sleep()
        rclpy.spin()
        # except KeyboardInterrupt:
        #     pass



def main():
    rclpy.init()
    node = StatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()