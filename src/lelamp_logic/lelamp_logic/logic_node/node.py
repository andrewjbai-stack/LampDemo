"""Stub for the LeLamp's "brain".

Runs a small state machine (behaviors.py: idle, attentive, engaged, look_around, search, focus,
obey, nod, shake, bored, thinking).
The current state sets look and move goals on motion_node (through
motion_client.py), which eases the joints on its own timer. The light color
comes from the current state.

The node is split by input, one mixin per file:
  faces.py   - asks face_node if someone is looking (sets attentive)
  objects.py - asks object_node what's in view, remembers it, looks back at it
  voice.py   - sends what voice_node heard to llm_node and runs its commands

Uses:    /lelamp/look, /lelamp/move, /lelamp/motion_state (motion_node)
         /lelamp/play_sound (sound_node)
Outputs: /lelamp/light (std_msgs/ColorRGBA)
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import ColorRGBA

from lelamp_interfaces.srv import PlaySound

from lelamp_logic.logic_node.behaviors import Attentive, Engaged, Focus, Idle, LookAround, Nod, Obey, Bored, Search, Shake, StateMachine, Thinking
from lelamp_logic.logic_node.faces import FacesMixin
from lelamp_logic.logic_node.motion_client import MotionClient
from lelamp_logic.logic_node.objects import ObjectsMixin
from lelamp_logic.logic_node.voice import VoiceMixin


class LogicNode(FacesMixin, ObjectsMixin, VoiceMixin, Node):
    def __init__(self):
        super().__init__('logic_node')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('face_rate_hz', 10.0)  # how often to check for faces
        self.declare_parameter('engage_after', 3.0)  # s of being looked at before engaged

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.dt = 1.0 / float(gp('rate_hz'))
        self.motion = MotionClient(self)  # motion_node
        self.sound_client = self.create_client(PlaySound, 'lelamp/play_sound')  # sound_node

        self._init_faces(float(gp('face_rate_hz')))
        self._init_objects()
        self._init_voice()

        self.brain = StateMachine(
            self, {'idle': Idle(), 'attentive': Attentive(engage_after=float(gp('engage_after'))),
                   'engaged': Engaged(), 'look_around': LookAround(),
                   'search': Search(), 'focus': Focus(), 'obey': Obey(),
                   'nod': Nod(), 'shake': Shake(),
                   'bored': Bored(), 'thinking': Thinking()},
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

    def play_sound(self, name):
        """Ask sound_node to play a sound (e.g. 'thinking'); doesn't wait. Skipped if it isn't running."""
        if not self.sound_client.service_is_ready():
            self.get_logger().warn('sound_node is not running', throttle_duration_sec=10.0)
            return
        self.sound_client.call_async(PlaySound.Request(sound=name)).add_done_callback(self._on_sound)

    def _on_sound(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'play_sound failed: {e}')
            return
        if not res.success:
            self.get_logger().warn(f'play_sound: {res.message}')


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
