"""Shows the camera feed as a "TV screen" inside the RViz 3D scene.

RViz has no built-in textured-plane display, so the image is downsampled to
a coarse grid and drawn as a CUBE_LIST marker with one thin colored tile per
pixel. Pixelated, but it needs no RViz plugins.
"""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


class ImageTvNode(Node):
    def __init__(self):
        super().__init__('image_tv')
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('cols', 64)
        self.declare_parameter('rows', 48)
        self.declare_parameter('width_m', 0.64)
        self.declare_parameter('x', -0.7)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.45)
        self.declare_parameter('yaw', 0.0)  # 0 = screen faces +x
        self.declare_parameter('rate_hz', 10.0)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cols, self.rows = int(gp('cols')), int(gp('rows'))
        self.bridge = CvBridge()
        self.latest = None

        m = Marker()
        m.header.frame_id = gp('frame_id')
        m.ns, m.id = 'image_tv', 0
        m.type, m.action = Marker.CUBE_LIST, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = gp('x'), gp('y'), gp('z')
        yaw = float(gp('yaw'))
        m.pose.orientation.z, m.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
        width = float(gp('width_m'))
        m.scale.x = 0.005
        m.scale.y = m.scale.z = width / self.cols
        m.color.a = 1.0
        m.points = self._grid_points(width)
        self.marker = m

        self.pub = self.create_publisher(Marker, 'lelamp/tv', 1)
        self.create_subscription(Image, 'camera/image_raw', self._on_image,
                                 qos_profile_sensor_data)
        self.create_timer(1.0 / float(gp('rate_hz')), self._publish)

    def _grid_points(self, width):
        # Screen lies in the marker's local y-z plane, facing +x. Seen from the
        # front (+x), image column 0 is on the viewer's left (-y), row 0 on top.
        d = width / self.cols
        height = d * self.rows
        return [Point(x=0.0, y=-width / 2 + (c + 0.5) * d, z=height / 2 - (r + 0.5) * d)
                for r in range(self.rows) for c in range(self.cols)]

    def _on_image(self, msg):
        self.latest = msg

    def _publish(self):
        if self.latest is None:
            return
        img = self.bridge.imgmsg_to_cv2(self.latest, desired_encoding='rgb8')
        small = cv2.resize(img, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        rgb = small.reshape(-1, 3) / 255.0
        self.marker.colors = [ColorRGBA(r=r, g=g, b=b, a=1.0) for r, g, b in rgb.tolist()]
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.marker)


def main(args=None):
    rclpy.init(args=args)
    node = ImageTvNode()
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
