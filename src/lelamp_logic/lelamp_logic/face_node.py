"""Face detection + head direction from the webcam.

Uses OpenCV's YuNet detector (5 landmarks per face) and solvePnP against a
generic 3D face to estimate which way the head is pointing. Tracks the largest
face only.

Inputs:  /camera/image_raw               (sensor_msgs/Image)
Outputs: /lelamp/face/looking            (std_msgs/Bool) head pointed at the camera
         /lelamp/face/head_pose          (geometry_msgs/Vector3Stamped, degrees)
             x = yaw   (+ = head turned toward image right)
             y = pitch (+ = head tilted up)
             z = angle between head direction and the line to the camera
         /lelamp/face/center             (geometry_msgs/PointStamped)
             x, y = face center in normalized image coords [-1, 1] (+x right, +y down)
             z = face width as a fraction of image width (rough distance cue)
         /lelamp/face/debug_image        (sensor_msgs/Image) annotated frame
head_pose and center are only published while a face is visible.
"""
import math
import os

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

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
        self._last_pose = None
        self.bridge = CvBridge()

        self.looking_pub = self.create_publisher(Bool, 'lelamp/face/looking', 10)
        self.pose_pub = self.create_publisher(Vector3Stamped, 'lelamp/face/head_pose', 10)
        self.center_pub = self.create_publisher(PointStamped, 'lelamp/face/center', 10)
        self.debug_pub = self.create_publisher(Image, 'lelamp/face/debug_image', 1)
        self.create_subscription(Image, 'lelamp/sim_camera/image_raw', self._on_image,
                                 qos_profile_sensor_data)

    def _on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        scale = self.detect_width / w
        small = cv2.resize(frame, (self.detect_width, int(round(h * scale))))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)

        face = None
        if faces is not None and len(faces):
            face = max(faces, key=lambda f: f[2] * f[3]) / scale  # largest, full-res coords

        if face is None:
            self.looking = False
        else:
            yaw, pitch, off_axis = self._head_pose(face, w, h)
            self.looking = off_axis < (self.look_off if self.looking else self.look_on)

            pose = Vector3Stamped()
            pose.header = msg.header
            pose.vector.x, pose.vector.y, pose.vector.z = yaw, pitch, off_axis
            self.pose_pub.publish(pose)

            x, y, fw, fh = face[:4]
            center = PointStamped()
            center.header = msg.header
            center.point.x = float((x + fw / 2) / w * 2 - 1)
            center.point.y = float((y + fh / 2) / h * 2 - 1)
            center.point.z = float(fw / w)
            self.center_pub.publish(center)

        self.looking_pub.publish(Bool(data=self.looking))

        if self.debug and self.debug_pub.get_subscription_count() > 0:
            self._publish_debug(frame, face, msg.header)

    def _head_pose(self, face, w, h):
        pts = face[4:14].reshape(5, 2).astype(np.float64)
        f = float(w)  # rough focal length for a laptop webcam (~53 deg HFOV)
        cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(FACE_MODEL, pts, cam, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            self._last_pose = None
            return 0.0, 0.0, 180.0
        rot, _ = cv2.Rodrigues(rvec)
        fwd = rot @ np.array([0.0, 0.0, -1.0])  # direction the face points
        self._last_pose = (rvec, tvec, cam)
        yaw = math.degrees(math.atan2(fwd[0], -fwd[2]))
        pitch = math.degrees(math.atan2(-fwd[1], math.hypot(fwd[0], fwd[2])))
        to_cam = -tvec.ravel() / np.linalg.norm(tvec)
        off_axis = math.degrees(math.acos(float(np.clip(fwd @ to_cam, -1.0, 1.0))))
        return yaw, pitch, off_axis

    def _publish_debug(self, frame, face, header):
        img = frame.copy()
        if face is not None:
            x, y, fw, fh = face[:4].astype(int)
            color = (0, 220, 0) if self.looking else (0, 140, 255)
            cv2.rectangle(img, (x, y), (x + fw, y + fh), color, 2)
            for i in range(5):
                cv2.circle(img, (int(face[4 + 2 * i]), int(face[5 + 2 * i])), 3, (255, 0, 0), -1)
            if self._last_pose is not None:
                rvec, tvec, cam = self._last_pose
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
