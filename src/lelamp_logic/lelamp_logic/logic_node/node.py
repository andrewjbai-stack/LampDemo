"""Stub for the LeLamp's "brain".

Runs a small state machine (behaviors.py: idle, attentive, look_around, focus, obey).
The current state sets look and move goals on motion_node (through
motion_client.py), which eases the joints on its own timer. The light color
comes from the current state.

The node is split by input, one mixin per file:
  faces.py   - asks face_node if someone is looking (sets attentive)
  objects.py - asks object_node what's in view, remembers it, looks back at it
  voice.py   - sends what voice_node heard to llm_node and runs its commands

Uses:    /lelamp/look, /lelamp/move, /lelamp/motion_state (motion_node)
Outputs: /lelamp/light (std_msgs/ColorRGBA)
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import ColorRGBA

from lelamp_logic.logic_node.behaviors import Attentive, Focus, Idle, LookAround, Obey, Bored, StateMachine, Thinking
from lelamp_logic.logic_node.faces import FacesMixin
from lelamp_logic.logic_node.motion_client import MotionClient
from lelamp_logic.logic_node.objects import ObjectsMixin
from lelamp_logic.logic_node.voice import VoiceMixin


class LogicNode(FacesMixin, ObjectsMixin, VoiceMixin, Node):
    def __init__(self):
        super().__init__('logic_node')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('face_rate_hz', 10.0)  # how often to check for faces

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.dt = 1.0 / float(gp('rate_hz'))
        self.motion = MotionClient(self)  # motion_node

        self._init_faces(float(gp('face_rate_hz')))
        self._init_objects()
        self._init_voice()

        self.brain = StateMachine(
            self, {'idle': Idle(), 'attentive': Attentive(), 'look_around': LookAround(),
                   'focus': Focus(), 'obey': Obey(), 'bored': Bored(), 'thinking': Thinking()},
            start='idle',
            on_change=lambda old, new: self.get_logger().info(f'state: {old} -> {new}'))

        self.color_pub = self.create_publisher(ColorRGBA, 'lelamp/light', 10)
        self.create_timer(self.dt, self._think)

    def _think(self):
        # The current state decides what to do; behaviors live in behaviors.py.
        self.brain.update(self.dt)

        # The current behavior picks the light (behaviors.py, Behavior.light).
        r, g, b, a = self.brain.current.light(self, self.brain.t)
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
