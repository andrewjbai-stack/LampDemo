"""Face detection + head direction, behind a service.

Keeps the newest camera frame. Each detect_faces call runs OpenCV's YuNet
detector (5 landmarks per face) on that frame and estimates which way each head
is pointing with solvePnP against a generic 3D face. "looking" is decided from
the largest face, with hysteresis between calls.

    ros2 service call /lelamp/detect_faces lelamp_interfaces/srv/DetectFaces

Inputs:  /lelamp/sim_camera/image_raw  (sensor_msgs/Image, param camera_topic)
Service: /lelamp/detect_faces          (lelamp_interfaces/DetectFaces)
             faces, largest first; per face (degrees):
             yaw   (+ = head turned toward image right)
             pitch (+ = head tilted up)
             off_axis = angle between head direction and the line to the camera
             center x, y = normalized image coords [-1, 1] (+x right, +y down),
             center z = face width as a fraction of image width (rough distance cue)
Outputs: /lelamp/face/debug_image      (sensor_msgs/Image) each checked frame,
             annotated, while something subscribes (RViz)
"""
import math
import os

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from lelamp_interfaces.msg import DetectedFace
from lelamp_interfaces.srv import DetectFaces

# Generic face in mm, OpenCV camera axes (x right, y down, z away from camera),
# nose tip at the origin. Order matches YuNet's landmarks: right eye, left eye
# (the person's), nose tip, right mouth corner, left mouth corner.
FACE_MODEL = np.array([
    [-32.0, -35.0, 30.0],
    [32.0, -35.0, 30.0],
    [0.0, 0.0, 0.0],
    [-26.0, 32.0, 22.0],
    [26.0, 32.0, 22.0],
], dtype=np.float64)


class FaceNode(Node):
    def __init__(self):
        super().__init__('face_node')
        default_model = os.path.join(get_package_share_directory('lelamp_logic'),
                                     'models', 'face_detection_yunet_2022mar.onnx')
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('camera_topic', 'lelamp/sim_camera/image_raw')
        self.declare_parameter('score_threshold', 0.8)
        self.declare_parameter('detect_width', 320)  # downscale before detecting
        self.declare_parameter('look_on_deg', 20.0)  # start "looking" below this
        self.declare_parameter('look_off_deg', 28.0)  # stop "looking" above this
        self.declare_parameter('publish_debug_image', True)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.detector = cv2.FaceDetectorYN.create(
            gp('model_path'), '', (320, 240), float(gp('score_threshold')), 0.3, 5000)
        self.detect_width = int(gp('detect_width'))
        self.look_on = float(gp('look_on_deg'))
        self.look_off = float(gp('look_off_deg'))
        self.debug = bool(gp('publish_debug_image'))
        self.looking = False
        self.frame = None
        self.bridge = CvBridge()

        self.debug_pub = self.create_publisher(Image, 'lelamp/face/debug_image', 1)
        self.create_subscription(Image, gp('camera_topic'), self._on_image,
                                 qos_profile_sensor_data)
        self.create_service(DetectFaces, 'lelamp/detect_faces', self._on_detect)

    def _on_image(self, msg):
        self.frame = msg  # only the newest is kept

    def _on_detect(self, request, response):
        msg = self.frame
        if msg is None:
            response.success = False
            response.message = 'no camera frame yet'
            return response
        try:
            return self._detect(msg, response)
        except Exception as e:  # a bad frame or detector error must not kill the node
            self.get_logger().error(f'detect_faces failed: {e}')
            response = DetectFaces.Response()  # drop anything half filled in
            response.success = False
            response.message = f'detection failed: {e}'
            return response

    def _detect(self, msg, response):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        scale = self.detect_width / w
        small = cv2.resize(frame, (self.detect_width, int(round(h * scale))))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        faces = [] if faces is None else sorted(
            (np.append(f[:14] / scale, f[14]) for f in faces),  # full-res coords, same score
            key=lambda f: -f[2] * f[3])  # largest first

        poses = []
        for face in faces:
            yaw, pitch, off_axis, pose = self._head_pose(face, w, h)
            poses.append(pose)
            x, y, fw, fh = (float(v) for v in face[:4])
            response.faces.append(DetectedFace(
                box=[x, y, fw, fh], score=float(face[14]),
                landmarks=[float(v) for v in face[4:14]],
                yaw=yaw, pitch=pitch, off_axis=off_axis,
                center=Point(x=(x + fw / 2) / w * 2 - 1, y=(y + fh / 2) / h * 2 - 1, z=fw / w)))

        if faces:
            off_axis = response.faces[0].off_axis
            self.looking = off_axis < (self.look_off if self.looking else self.look_on)
        else:
            self.looking = False

        response.success = True
        response.looking = self.looking
        response.header = msg.header
        response.message = (f'{len(faces)} face(s), looking' if self.looking else
                            f'{len(faces)} face(s)')
        if self.debug and self.debug_pub.get_subscription_count() > 0:
            self._publish_debug(frame, faces[0] if faces else None,
                                poses[0] if poses else None, msg.header)
        return response

    def _head_pose(self, face, w, h):
        """(yaw, pitch, off_axis) in degrees and the raw PnP pose (or None)."""
        pts = face[4:14].reshape(5, 2).astype(np.float64)
        f = float(w)  # rough focal length for a laptop webcam (~53 deg HFOV)
        cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(FACE_MODEL, pts, cam, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return 0.0, 0.0, 180.0, None
        rot, _ = cv2.Rodrigues(rvec)
        fwd = rot @ np.array([0.0, 0.0, -1.0])  # direction the face points
        yaw = math.degrees(math.atan2(fwd[0], -fwd[2]))
        pitch = math.degrees(math.atan2(-fwd[1], math.hypot(fwd[0], fwd[2])))
        to_cam = -tvec.ravel() / np.linalg.norm(tvec)
        off_axis = math.degrees(math.acos(float(np.clip(fwd @ to_cam, -1.0, 1.0))))
        return yaw, pitch, off_axis, (rvec, tvec, cam)

    def _publish_debug(self, frame, face, pose, header):
        img = frame.copy()
        if face is not None:
            x, y, fw, fh = face[:4].astype(int)
            color = (0, 220, 0) if self.looking else (0, 140, 255)
            cv2.rectangle(img, (x, y), (x + fw, y + fh), color, 2)
            for i in range(5):
                cv2.circle(img, (int(face[4 + 2 * i]), int(face[5 + 2 * i])), 3, (255, 0, 0), -1)
            if pose is not None:
                rvec, tvec, cam = pose
                tip, _ = cv2.projectPoints(np.array([[0.0, 0.0, -80.0]]), rvec, tvec, cam, None)
                nose = (int(face[8]), int(face[9]))
                cv2.line(img, nose, tuple(tip.ravel().astype(int)), color, 3)
            label = 'LOOKING' if self.looking else 'looking away'
            cv2.putText(img, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out.header = header
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FaceNode()
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
