"""Stub for the LeLamp's "brain".

Runs a small state machine (behaviors.py: idle, attentive, look_around). The
current state sets goals on LampMotion (motion.py), which eases the joints there
every tick. The light color comes from the current state.

Inputs:  /lelamp/face/looking     (std_msgs/Bool)
Outputs: /lelamp/joint_commands   (sensor_msgs/JointState, target positions)
         /lelamp/light            (std_msgs/ColorRGBA)
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, ColorRGBA

from lelamp_logic.behaviors import Attentive, Idle, LookAround, StateMachine
from lelamp_logic.motion import LampMotion


class LogicNode(Node):
    def __init__(self):
        super().__init__('logic_node')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('idle_motion', True)  # breathing on top of every pose

        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)
        self.motion = LampMotion(breathing=bool(self.get_parameter('idle_motion').value))

        self.attentive = False # When user looks at lamp/webcam, lamp becomes attentive

        self.brain = StateMachine(
            self, {'idle': Idle(), 'attentive': Attentive(), 'look_around': LookAround()},
            start='idle',
            on_change=lambda old, new: self.get_logger().info(f'state: {old} -> {new}'))

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
        # The current state decides what to do; behaviors live in behaviors.py.
        self.brain.update(self.dt)

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name, cmd.position = self.motion.step(self.dt)
        self.cmd_pub.publish(cmd)

        r, g, b, a = self.brain.current.color
        self.color_pub.publish(ColorRGBA(r=r, g=g, b=b, a=a))


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
