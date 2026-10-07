"""Stub for the LeLamp's "brain".

Runs a small state machine (behaviors.py: idle, attentive, look_around, focus, obey).
The current state sets goals on LampMotion (motion.py), which eases the joints
there every tick. The light color comes from the current state.

face_rate_hz times a second it asks face_node whether someone is looking at
the lamp (the detect_faces service); that sets attentive.

While looking around, the lamp asks object_node what it can see (the
detect_objects service) and remembers where each object is in base_link
(object_memory.py). Asking for an object by name makes it look back at it.

Whatever voice_node hears arrives on /lelamp/speech and is sent to
llm_node (the parse_command service), which answers with commands to
run: look at an object, look around, look a way, change posture, set the light.
While it thinks, face and object checks are paused and new phrases are ignored.
A light color asked for by voice stays until another one is asked for; the
state still sets the brightness.

Inputs:  /lelamp/look_at_object       (std_msgs/String, e.g. "clock")
         /lelamp/speech               (std_msgs/String, voice_node, one per phrase)
Uses:    /lelamp/detect_faces         (lelamp_interfaces/DetectFaces, face_node)
         /lelamp/detect_objects       (lelamp_interfaces/DetectObjects, object_node)
         /lelamp/parse_command        (lelamp_interfaces/ParseCommand, llm_node)
Outputs: /lelamp/joint_commands       (sensor_msgs/JointState, target positions)
         /lelamp/light                (std_msgs/ColorRGBA)
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, String

from lelamp_interfaces.srv import DetectFaces, DetectObjects, ParseCommand
from lelamp_logic.behaviors import Attentive, Focus, Idle, LookAround, Obey, Bored, StateMachine, Thinking
from lelamp_logic.motion import LampMotion
from lelamp_logic.object_memory import ObjectMemory

# set_light colors (r, g, b); 'off' turns the light off
COLORS = {'red': (1.0, 0.0, 0.0), 'orange': (1.0, 0.45, 0.0), 'yellow': (1.0, 0.9, 0.1),
          'green': (0.0, 1.0, 0.2), 'blue': (0.1, 0.3, 1.0), 'purple': (0.6, 0.1, 1.0),
          'pink': (1.0, 0.4, 0.7), 'white': (1.0, 1.0, 1.0), 'warm': (1.0, 0.75, 0.45),
          'off': None}
# look directions as (yaw, pitch) degrees; + yaw = lamp's left, + pitch = up
DIRECTIONS = {'left': (60.0, -10.0), 'right': (-60.0, -10.0), 'up': (0.0, 35.0),
              'down': (0.0, -40.0), 'forward': (0.0, -10.0)}


class LogicNode(Node):
    def __init__(self):
        super().__init__('logic_node')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('idle_motion', True)  # breathing on top of every pose
        self.declare_parameter('face_rate_hz', 10.0)  # how often to check for faces

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.dt = 1.0 / float(gp('rate_hz'))
        self.motion = LampMotion(breathing=bool(gp('idle_motion')))

        self.attentive = False # When user looks at lamp/webcam, lamp becomes attentive
        self.faces = []  # latest DetectedFace list from face_node, largest first
        self.focus_target = None  # (x, y, z) the focus behavior looks at
        self.face_client = self.create_client(DetectFaces, 'lelamp/detect_faces')
        self.checking_faces = False  # a detect_faces call is in flight

        self.memory = ObjectMemory()  # fresh every run
        self.detect_client = self.create_client(DetectObjects, 'lelamp/detect_objects')
        self.detecting = False  # a detect_objects call is in flight

        self.parse_client = self.create_client(ParseCommand, 'lelamp/parse_command')
        self.thinking = False  # a parse_command call is in flight
        self.thinking_color_flag = False 
        self.thinking_color = (0.0, 0.0, 1.0, 1.0)
        self.light = None  # (r, g, b) asked for by voice, or 'off'; None = state's own

        self.brain = StateMachine(
            self, {'idle': Idle(), 'attentive': Attentive(), 'look_around': LookAround(),
                   'focus': Focus(), 'obey': Obey(), 'bored': Bored(), 'thinking': Thinking()},
            start='idle',
            on_change=lambda old, new: self.get_logger().info(f'state: {old} -> {new}'))

        self.create_subscription(String, 'lelamp/look_at_object', self._on_look_at_object, 10)
        self.create_subscription(String, 'lelamp/speech', self._on_speech, 10)
        self.cmd_pub = self.create_publisher(JointState, 'lelamp/joint_commands', 10)
        self.color_pub = self.create_publisher(ColorRGBA, 'lelamp/light', 10)
        self.create_timer(self.dt, self._think)
        self.create_timer(1.0 / float(gp('face_rate_hz')), self.check_faces)

    # --- faces ---

    def check_faces(self):
        """Ask face_node about the newest frame; the answer sets attentive."""
        if self.checking_faces or self.thinking or not self.face_client.service_is_ready():
            return
        self.checking_faces = True
        self.face_client.call_async(DetectFaces.Request()).add_done_callback(self._on_faces)

    def _on_faces(self, future):
        self.checking_faces = False
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'detect_faces failed: {e}', throttle_duration_sec=5.0)
            return
        if not res.success:
            return
        self.faces = res.faces
        self.attentive = res.looking

    def _on_look_at_object(self, msg):
        self.look_at_object(msg.data)

    # --- voice ---

    def _on_speech(self, msg):
        """A phrase from voice_node: ask llm_node what to do."""
        if self.thinking:
            self.get_logger().info(f'heard: "{msg.data}" (still thinking, ignored)')
            return
        self.get_logger().info(f'heard: "{msg.data}"')
        if not self.parse_client.service_is_ready():
            self.get_logger().warn('llm_node is not running', throttle_duration_sec=10.0)
            return
        self.thinking = True
        req = ParseCommand.Request(text=msg.data, known_objects=self.memory.names())
        self.parse_client.call_async(req).add_done_callback(self._on_commands)

        self.brain.go('thinking')
        
        # When false the lamp light dims, when true is brightens which allows for the thinking light pattern
        self.thinking_color_flag = False 
        self.thinking_color = (0.9, 0.9, 1.0, 1.0)

    def _on_commands(self, future):
        self.thinking = False
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'parse_command failed: {e}')
            self.brain.go('idle')
            return
        if not res.success:
            self.get_logger().warn(f'parse_command: {res.message}')
            self.brain.go('idle')
            return
        for c in res.commands:
            self.run_command(c.tool, c.arg)
        # No command picked a new state (e.g. only set_light): stop thinking.
        if self.brain.name == 'thinking':
            self.brain.go('idle')

    def run_command(self, tool, arg):
        """Do one command from llm_node. False if it can't."""
        self.get_logger().info(f'command: {tool}({arg})')
        if tool == 'look_at_object':
            return self.look_at_object(arg)
        if tool == 'scan_room':
            self.brain.go('look_around')
        elif tool == 'look' and arg in DIRECTIONS:
            self.motion.look(*DIRECTIONS[arg])
            self.brain.go('obey')
        elif tool == 'move_to' and arg in ('rest', 'lean_in', 'sit_back', 'tall', 'shy'):
            self.motion.move_to(arg)
            self.brain.go('obey')
        elif tool == 'set_light' and arg in COLORS:
            self.light = COLORS[arg] or 'off'
        else:
            self.get_logger().warn(f'unknown command {tool}({arg})')
            return False
        return True

    # --- object memory ---

    def snapshot(self):
        """Ask object_node what's in view now and remember it. Returns at once
        (the answer arrives in _on_detected); False if skipped."""
        if self.detecting or self.thinking or not self.detect_client.service_is_ready():
            return False
        self.detecting = True
        self.detect_client.call_async(DetectObjects.Request()).add_done_callback(self._on_detected)
        return True

    def _on_detected(self, future):
        self.detecting = False
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'detect_objects failed: {e}')
            return
        if not res.success:
            self.get_logger().warn(f'detect_objects: {res.message}', throttle_duration_sec=5.0)
            return
        for obj in res.objects:
            p = obj.position
            self.memory.add(obj.name, (p.x, p.y, p.z), obj.confidence)
        if res.objects:
            self.get_logger().info(f'saw: {res.message}')

    def look_at_object(self, name):
        """Look at a remembered object for a while. False if it's never been seen."""
        point = self.memory.find(name)
        if point is None:
            self.get_logger().info(f"haven't seen '{name}' (known: {', '.join(self.memory.names())})")
            return False
        self.focus_target = point
        self.brain.go('focus')
        return True

    def _think(self):
        # The current state decides what to do; behaviors live in behaviors.py.
        self.brain.update(self.dt)

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name, cmd.position = self.motion.step(self.dt)
        self.cmd_pub.publish(cmd)
        # The state sets the brightness; a voice color replaces the state's color.
        r, g, b, a = self.brain.current.color
        if self.light == 'off':
            a = 0.0
        elif self.light:
            r, g, b = self.light

        if self.brain.name == 'thinking':
            r, g, b, a = self.thinking_color
            if self.thinking_color_flag == False:
                if(self.thinking_color[0] < 0.4):
                    self.thinking_color_flag = True
                rg_value = max(0.0, min(r - 0.05, 1.0))

            if self.thinking_color_flag == True:
                if(self.thinking_color[0] > 0.9):
                    self.thinking_color_flag = False
                rg_value = max(0.0, min(r + 0.05, 1.0))
            self.thinking_color = (rg_value, rg_value, b, a)
            r, g, b, a = self.thinking_color

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
