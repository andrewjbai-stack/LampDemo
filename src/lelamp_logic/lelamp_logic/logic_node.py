"""Stub for the LeLamp's "brain".

Today it only proves the wiring: it watches the camera stream and sends the
body a gentle idle "breathing" motion. Perception, speech, memory, and
goal-directed behavior will replace `_think()`.

Inputs:  /camera/image_raw        (sensor_msgs/Image)
Outputs: /lelamp/joint_commands   (sensor_msgs/JointState, target positions)
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, ColorRGBA


class LogicNode(Node):
    def __init__(self):
        super().__init__('logic_node')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('idle_motion', True)

        self.idle_motion = bool(self.get_parameter('idle_motion').value)
        self.t = 0.0
        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)

        self.attentive = False # When user looks at lamp/webcam, lamp becomes attentive
        self.last_logged_attentive = None  # so the first light state gets logged

        #self.create_subscription(Image, 'camera/image_raw', self._on_image,
        #                         qos_profile_sensor_data)
        self.create_subscription(Bool, 'lelamp/face/looking', self._looking,
                                 qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(JointState, 'lelamp/joint_commands', 10)
        self.color_pub = self.create_publisher(ColorRGBA, 'lelamp/light', 10)
        self.create_timer(self.dt, self._think)

    def _looking(self, msg):
        if(msg.data):
            self.attentive = True
        else:
            self.attentive = False


    def _think(self):
        # TODO: perception -> decision -> action. For now, idle breathing.
        self.t += self.dt
        if self.idle_motion:
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = ['base_yaw_joint', 'shoulder_pitch_joint', 'elbow_pitch_joint',
                        'neck_yaw_joint', 'head_pitch_joint']
            cmd.position = [
                0.4 * math.sin(0.2 * self.t),
                0.25 + 0.08 * math.sin(0.5 * self.t),
                -0.9 + 0.08 * math.sin(0.5 * self.t + 0.6),
                0.3 * math.sin(0.35 * self.t),
                0.3 + 0.1 * math.sin(0.5 * self.t + 1.2),
            ]
            self.cmd_pub.publish(cmd)

        color_msg = ColorRGBA()
        if(self.attentive):
            color_msg.r = 1.0
            color_msg.g = 0.0
            color_msg.b = 0.0
            color_msg.a = 1.0
        else:
            color_msg.r = 1.0
            color_msg.g = 1.0
            color_msg.b = 1.0
            color_msg.a = 1.0

        # Log only when the light changes, not every tick
        if self.attentive != self.last_logged_attentive:
            self.get_logger().info(
                f'light -> {"attentive" if self.attentive else "idle"} '
                f'(r={color_msg.r:.2f} g={color_msg.g:.2f} b={color_msg.b:.2f} a={color_msg.a:.2f})')
            self.last_logged_attentive = self.attentive

        self.color_pub.publish(color_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LogicNode()
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
