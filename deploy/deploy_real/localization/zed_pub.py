#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Empty
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
from PIL import Image as PILImage
import numpy as np
import os

import socket
def send_file(filename, host='receiver_ip_here', port=5001):
    s = socket.socket()
    s.connect((host, port))

    with open(filename, "rb") as f:
        data = f.read(1024)
        while data:
            s.send(data)
            data = f.read(1024)

    s.close()
    print("File sent successfully")


class CaptureOnTrigger(Node):
    """
    Subscribes to ZED image + camera_info, caches latest, and on trigger:
      - republishes the cached Image on /captured/image
      - republishes the cached CameraInfo on /captured/camera_info
    Triggers:
      - topic:  /capture_trigger  (std_msgs/Empty)
      - service: /capture         (std_srvs/Trigger)  -> returns OK + count
    """

    def __init__(self):
        super().__init__('capture_on_trigger')

        # Params (override in launch or CLI)
        left_cam_image_topic = "/zed/zed_node/right/image_rect_color"
        right_cam_image_topic = "/zed/zed_node/left/image_rect_color"
        left_cam_info = "/zed/zed_node/left/camera_info"
        right_cam_info = "/zed/zed_node/right/camera_info"

        captured_left_image_topic = '/captured/left/image'
        captured_right_image_topic = '/captured/right/image'
        captured_left_camera_info_topic = '/captured/left/camera_info'
        captured_right_camera_info_topic = '/captured/right/camera_info'
        trigger_topic = '/capture_trigger'
        latch_captured = True

        # Cache
        self._last_left_image = None
        self._last_right_image = None
        self._last_left_info = None
        self._last_right_info = None
        self._capture_count = 0

        # Subscriptions: use SensorData QoS for ZED streams
        self._left_img_sub = self.create_subscription(
            Image, left_cam_image_topic, self._on_left_image, qos_profile_sensor_data
        )
        self._right_img_sub = self.create_subscription(
            Image, right_cam_image_topic, self._on_right_image, qos_profile_sensor_data
        )
        self._left_info_sub = self.create_subscription(
            CameraInfo, left_cam_info, self._on_left_info, qos_profile_sensor_data
        )
        self._right_info_sub = self.create_subscription(
            CameraInfo, right_cam_info, self._on_right_info, qos_profile_sensor_data
        )

        # Trigger via topic
        self._trigger_sub = self.create_subscription(
            Empty, trigger_topic, self._on_trigger_msg, 10
        )

        # Trigger via service
        self._srv = self.create_service(Trigger, 'capture', self._on_trigger_srv)

        # Publishers for captured outputs
        if latch_captured:
            captured_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            )
            captured_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        else:
            captured_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            )

        self._pub_left_img = self.create_publisher(Image, captured_left_image_topic, captured_qos)
        self._pub_right_img = self.create_publisher(Image, captured_right_image_topic, captured_qos)
        self._pub_left_info = self.create_publisher(CameraInfo, captured_left_camera_info_topic, captured_qos)
        self._pub_right_info = self.create_publisher(CameraInfo, captured_right_camera_info_topic, captured_qos)


    def _on_left_image(self, msg: Image):
        self._last_left_image = msg
    
    def _on_right_image(self, msg: Image):
        self._last_right_image = msg

    def _on_left_info(self, msg: CameraInfo):
        self._last_left_info = msg

    def _on_right_info(self, msg: CameraInfo):
        self._last_right_info = msg

    def _publish_captured(self) -> bool:
        if self._last_left_image is None or self._last_left_info is None:
            self.get_logger().warn('No image/camera_info cached yet—cannot capture.')
            return False

        # Optionally update timestamp to "now" if you prefer:

        def msg_converter(msg):
            msg.header.stamp = self.get_clock().now().to_msg()
            return msg

        # Publish as-is to preserve original frame timing
        self._pub_left_img.publish(msg_converter(self._last_left_image))
        self._pub_right_img.publish(msg_converter(self._last_right_image))
        self._pub_left_info.publish(msg_converter(self._last_left_info))
        self._pub_right_info.publish(msg_converter(self._last_right_info))


        # save
        if True:
            left_rgb = self._image_to_rgb_numpy(self._last_left_image)
            right_rgb = self._image_to_rgb_numpy(self._last_right_image)
            pil_left_img = PILImage.fromarray(left_rgb)
            pil_right_img = PILImage.fromarray(right_rgb)
            l_filename = os.path.join("/root/unitree_rl_gym/deploy/deploy_real/localization/zed_images", f'capture_left_{self._capture_count:04d}.png')
            r_filename = os.path.join("/root/unitree_rl_gym/deploy/deploy_real/localization/zed_images", f'capture_right_{self._capture_count:04d}.png')
            pil_left_img.save(l_filename)
            pil_right_img.save(r_filename)

            send_file(l_filename, "137.68.192.153")
            send_file(r_filename, "137.68.192.153")
            self._capture_count += 1
            print("CAPTURED!!!")
        return True

    def _on_trigger_msg(self, _msg: Empty):
        self._publish_captured()

    def _on_trigger_srv(self, _req: Trigger.Request, _resp: Trigger.Response):
        ok = self._publish_captured()
        resp = Trigger.Response()
        resp.success = ok
        resp.message = f'capture_count={self._capture_count}' if ok else 'no cached frame yet'
        return resp

    def _image_to_rgb_numpy(self, msg: Image):
        import numpy as np
        enc = msg.encoding.lower()
        h, w = msg.height, msg.width
        step = msg.step  # 바이트/행
        buf = np.frombuffer(msg.data, dtype=np.uint8)

        # 행 보폭(step)이 가끔 w*BPP와 다를 수 있어 안전하게 잘라서 reshape
        if enc in ['rgb8', 'bgr8']:
            bpp = 3
            expected = step * h
            if buf.size < expected:
                raise ValueError(f'buffer too small: {buf.size} < {expected}')
            img = buf[:h*step].reshape(h, step)[:, :w*bpp].reshape(h, w, bpp)
            if enc == 'bgr8':
                img = img[:, :, ::-1]  # BGR -> RGB
            return img

        elif enc in ['rgba8', 'bgra8']:
            bpp = 4
            expected = step * h
            if buf.size < expected:
                raise ValueError(f'buffer too small: {buf.size} < {expected}')
            img = buf[:h*step].reshape(h, step)[:, :w*bpp].reshape(h, w, bpp)
            if enc == 'bgra8':
                # BGRA -> RGB (A는 버림)
                img = img[:, :, [2, 1, 0]]  # B,G,R,A 중 R,G,B만
            else:
                # RGBA -> RGB
                img = img[:, :, [0, 1, 2]]
            return img

        elif enc in ['mono8']:
            bpp = 1
            expected = step * h
            if buf.size < expected:
                raise ValueError(f'buffer too small: {buf.size} < {expected}')
            img = buf[:h*step].reshape(h, step)[:, :w*bpp].reshape(h, w)
            # 회색을 RGB로
            return np.stack([img, img, img], axis=-1)

        else:
            raise ValueError(f'Unsupported encoding: {msg.encoding}')
def main():
    rclpy.init()
    node = CaptureOnTrigger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()