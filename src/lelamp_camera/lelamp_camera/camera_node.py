"""Publishes the laptop webcam as sensor_msgs/Image on /camera/image_raw.

If the camera can't be opened (e.g. WSL2 without usbipd passthrough) and
`fallback_test_pattern` is true, a moving test pattern is published instead so
the rest of the pipeline can still run.
"""
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('fallback_test_pattern', True)
        self.declare_parameter('reopen_interval_s', 5.0)

        self.device = self.get_parameter('device').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.fallback = bool(self.get_parameter('fallback_test_pattern').value)
        self.reopen_interval = float(self.get_parameter('reopen_interval_s').value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.cap = None
        self.last_open_attempt = 0.0
        self.frame_count = 0
        self._open_camera()
        self.timer = self.create_timer(1.0 / self.fps, self._tick)

    def _open_camera(self):
        self.last_open_attempt = time.monotonic()
        # Accept either a path (/dev/video0) or an integer index ("0").
        dev = int(self.device) if str(self.device).isdigit() else self.device
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap = cap
            self.get_logger().info(f'Opened camera {self.device}')
        else:
            cap.release()
            self.cap = None
            msg = f'Could not open camera {self.device}'
            if self.fallback:
                msg += '; publishing test pattern instead'
            self.get_logger().warn(msg)

    def _test_pattern(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        x = int((self.frame_count * 4) % self.width)
        cv2.circle(img, (x, self.height // 2), 40, (0, 200, 255), -1)
        cv2.putText(img, 'NO CAMERA - TEST PATTERN', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img

    def _tick(self):
        frame = None
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().warn('Camera read failed; will try to reopen')
                self.cap.release()
                self.cap = None
                frame = None
        elif time.monotonic() - self.last_open_attempt > self.reopen_interval:
            self._open_camera()

        if frame is None:
            if not self.fallback:
                return
            frame = self._test_pattern()

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)
        self.frame_count += 1

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
