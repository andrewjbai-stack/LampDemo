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


class LogicNode(Node):
    def __init__(self):
        super().__init__('logic_node')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('idle_motion', True)

        self.idle_motion = bool(self.get_parameter('idle_motion').value)
        self.latest_image = None
        self.frames_seen = 0
        self.t = 0.0
        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)

        self.create_subscription(Image, 'camera/image_raw', self._on_image,
                                 qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(JointState, 'lelamp/joint_commands', 10)
        self.create_timer(self.dt, self._think)
        self.create_timer(5.0, self._report)

    def _on_image(self, msg):
        self.latest_image = msg
        self.frames_seen += 1

    def _report(self):
        self.get_logger().info(f'camera frames seen: {self.frames_seen}')

    def _think(self):
        # TODO: perception -> decision -> action. For now, idle breathing.
        self.t += self.dt
        if not self.idle_motion:
            return
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
