"""Placeholder simulated lamp body.

Subscribes to target joint positions on /lelamp/joint_commands
(sensor_msgs/JointState, positions only) and publishes /joint_states, moving
each joint toward its target no faster than its URDF velocity limit and
clamped to its URDF position limits. robot_state_publisher turns /joint_states
into TF for RViz.
"""
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def parse_revolute_joints(urdf_xml):
    joints = {}
    for j in ET.fromstring(urdf_xml).findall('joint'):
        if j.get('type') not in ('revolute', 'continuous', 'prismatic'):
            continue
        lim = j.find('limit')
        lower = float(lim.get('lower', '-inf')) if lim is not None else float('-inf')
        upper = float(lim.get('upper', 'inf')) if lim is not None else float('inf')
        vel = float(lim.get('velocity', '1.0')) if lim is not None else 1.0
        joints[j.get('name')] = {'lower': lower, 'upper': upper, 'velocity': vel}
    return joints


class JointSimNode(Node):
    def __init__(self):
        super().__init__('joint_sim')
        self.declare_parameter('robot_description', '')
        self.declare_parameter('rate_hz', 50.0)

        urdf = self.get_parameter('robot_description').value
        if not urdf:
            raise RuntimeError('robot_description parameter is empty')
        self.limits = parse_revolute_joints(urdf)
        self.names = list(self.limits)
        self.pos = {n: 0.0 for n in self.names}
        self.target = dict(self.pos)
        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)

        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_subscription(JointState, 'lelamp/joint_commands', self._on_cmd, 10)
        self.create_timer(self.dt, self._step)
        self.get_logger().info(f'Simulating joints: {", ".join(self.names)}')

    def _on_cmd(self, msg):
        for name, p in zip(msg.name, msg.position):
            if name in self.limits:
                lim = self.limits[name]
                self.target[name] = min(max(p, lim['lower']), lim['upper'])

    def _step(self):
        for n in self.names:
            max_step = self.limits[n]['velocity'] * self.dt
            err = self.target[n] - self.pos[n]
            self.pos[n] += max(-max_step, min(max_step, err))
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = [self.pos[n] for n in self.names]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointSimNode()
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
