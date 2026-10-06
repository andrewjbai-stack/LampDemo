"""Object detection for the LeLamp: YOLO-World on the CPU, behind a service.

Keeps the newest head-camera frame. Each detect_objects call runs YOLO-World
(detector.py) on that frame, places every object in base_link using the camera
pose at the frame's time (tf), and returns them. It remembers nothing itself;
logic_node keeps the memory.

    ros2 service call /lelamp/detect_objects lelamp_interfaces/srv/DetectObjects

Inputs:  /lelamp/sim_camera/image_raw (sensor_msgs/Image, param camera_topic)
         tf base_link -> camera_link  (robot_state_publisher)
Service: /lelamp/detect_objects       (lelamp_interfaces/DetectObjects)
Outputs: /lelamp/objects/debug_image  (sensor_msgs/Image, bgr8) each detected
             frame with its boxes, while something subscribes (RViz)
"""
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener

from lelamp_interfaces.msg import DetectedObject
from lelamp_interfaces.srv import DetectObjects
from lelamp_logic.detector import locate, pixel_ray


def _quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class ObjectNode(Node):
    def __init__(self):
        super().__init__('object_node')
        self.declare_parameter('camera_topic', 'lelamp/sim_camera/image_raw')
        self.declare_parameter('camera_fovy', 60.0)  # degrees; the sim head camera's
        self.declare_parameter('model_path', '~/.lelamp/models/yolov8s-worldv2.pt')
        self.declare_parameter('detect_conf', 0.15)
        self.declare_parameter('publish_debug_image', True)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.fovy = float(gp('camera_fovy'))
        self.debug = bool(gp('publish_debug_image'))
        self.detector = None
        try:
            from lelamp_logic.detector import Detector
            self.detector = Detector(gp('model_path'), conf=float(gp('detect_conf')))
        except Exception as e:  # missing torch/ultralytics or model
            self.get_logger().error(f'detector failed to load, every call will fail: {e}')
        self.frame = None
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)

        self.create_subscription(Image, gp('camera_topic'), self._on_image,
                                 qos_profile_sensor_data)
        self.create_service(DetectObjects, 'lelamp/detect_objects', self._on_detect)
        self.debug_pub = self.create_publisher(Image, 'lelamp/objects/debug_image', 1)
        self.get_logger().info('object detection ready')

    def _on_image(self, msg):
        self.frame = msg  # only the newest is kept

    def _camera_pose(self, msg):
        """(origin, rotation) of the camera in base_link when msg was taken."""
        frame_id = msg.header.frame_id or 'camera_link'
        try:
            tf = self.tf.lookup_transform('base_link', frame_id, Time.from_msg(msg.header.stamp),
                                          timeout=Duration(seconds=0.05))
        except TransformException:
            # frame newer than the last /joint_states: use the latest pose
            tf = self.tf.lookup_transform('base_link', frame_id, Time())
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z]), _quat_matrix(tf.transform.rotation)

    def _on_detect(self, request, response):
        msg = self.frame
        if self.detector is None or msg is None:
            response.success = False
            response.message = 'no detector' if self.detector is None else 'no camera frame yet'
            return response
        try:
            origin, rot = self._camera_pose(msg)
        except TransformException as e:
            response.success = False
            response.message = f'no camera pose: {e}'
            return response

        img = _to_bgr(msg)
        h, w = img.shape[:2]
        found = self.detector.detect(img)
        for name, conf, box in found:
            x1, y1, x2, y2 = box
            centre = rot @ pixel_ray((x1 + x2) / 2, (y1 + y2) / 2, w, h, self.fovy)
            bottom = rot @ pixel_ray((x1 + x2) / 2, y2, w, h, self.fovy)
            p = locate(origin, centre, bottom)
            response.objects.append(DetectedObject(
                name=name, confidence=float(conf),
                position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                direction=Vector3(x=float(centre[0]), y=float(centre[1]), z=float(centre[2])),
                box=[float(v) for v in box]))

        response.success = True
        response.message = ', '.join(f'{n} {c:.2f}' for n, c, _ in found) or 'nothing found'
        response.header.stamp = msg.header.stamp
        response.header.frame_id = 'base_link'
        response.camera_origin = Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
        if self.debug and self.debug_pub.get_subscription_count() > 0:
            self._publish_debug(img, found, msg.header)
        return response

    def _publish_debug(self, img, found, header):
        """The frame with each detection boxed and labelled 'name conf'."""
        img = img.copy()
        for name, conf, box in found:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            color = (0, 220, 0) if conf >= 0.4 else (0, 200, 255)  # green: sure, amber: maybe
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            label_y = y1 - 3 if y1 > 12 else y1 + 11  # inside the box at the top edge
            cv2.putText(img, f'{name} {conf:.2f}', (x1 + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        out = Image(header=header, height=img.shape[0], width=img.shape[1],
                    encoding='bgr8', step=img.shape[1] * 3, data=img.tobytes())
        self.debug_pub.publish(out)


def _to_bgr(msg):
    channels = 1 if msg.encoding == 'mono8' else 3
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = img[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == 'rgb8':
        img = img[:, :, ::-1]
    elif channels == 1:
        img = np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(img)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
