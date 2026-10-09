"""Voice commands for logic_node.

Whatever voice_node hears arrives on /lelamp/speech. The lamp only listens
while engaged: someone kept looking at it, or said the wake word ("friend"),
which makes it engaged for wake_window seconds. Phrases heard while engaged are
sent to llm_node (the parse_command service), which answers with commands to
run: look at an object, look around, look a way, change posture, set the light,
or answer a yes/no question with a nod or a shake. Questions about what the
lamp has seen ("have you seen a plant?") are answered from the object memory,
not by the LLM. Only a phrase that sounds like a question can be answered
(QUESTION_RE), so chatter the LLM mistakes for one doesn't get a nod or shake,
and only the first answer in a phrase counts.
Anything said after the wake word in the same phrase ("friend, look left") is
sent right away.
While it thinks, face and object checks are paused and new phrases are ignored.
If llm_node takes longer than parse_timeout seconds, the lamp gives up (back to
idle) and the late answer is ignored.
A light color asked for by voice stays until another one is asked for; the
state still sets the brightness.

Input: /lelamp/speech        (std_msgs/String, voice_node, one per phrase)
Uses:  /lelamp/parse_command (lelamp_interfaces/ParseCommand, llm_node)
"""
import re
import time

from std_msgs.msg import String

from lelamp_interfaces.srv import ParseCommand

# set_light colors (r, g, b); 'off' turns the light off
COLORS = {'red': (1.0, 0.0, 0.0), 'orange': (1.0, 0.45, 0.0), 'yellow': (1.0, 0.9, 0.1),
          'green': (0.0, 1.0, 0.2), 'blue': (0.1, 0.3, 1.0), 'purple': (0.6, 0.1, 1.0),
          'pink': (1.0, 0.4, 0.7), 'white': (1.0, 1.0, 1.0), 'warm': (1.0, 0.75, 0.45),
          'off': None}
# look directions as (yaw, pitch) degrees; + yaw = lamp's left, + pitch = up
DIRECTIONS = {'left': (60.0, -10.0), 'right': (-60.0, -10.0), 'up': (0.0, 35.0),
              'down': (0.0, -40.0), 'forward': (0.0, -10.0)}
# a question: a '?' anywhere, or it starts with a yes/no question word (after "hey", "lamp" ...)
QUESTION_RE = re.compile(r"\?|^\W*((lamp|hey|so|and|ok|okay|um|well)\W+)?"
                         r"(is|are|am|do|does|did|can|could|have|has|"
                         r"had|was|were|will|would|should|isn't|aren't|don't|didn't)\b",
                         re.IGNORECASE)


class VoiceMixin:
    def _init_voice(self):
        self.parse_client = self.create_client(ParseCommand, 'lelamp/parse_command')
        self.thinking = False  # a parse_command call is in flight
        self.parse_id = 0  # id of the current call; answers with another id are late
        self.parse_sent = 0.0  # time.monotonic() the current call was sent
        self.parse_timeout = float(self.declare_parameter('parse_timeout', 15.0).value)
        self.create_timer(0.25, self._check_parse_timeout)
        self.voice_light = None  # (r, g, b) asked for by voice, or 'off'; None = state's own
        self.heard = ''  # the phrase sent to llm_node last
        wake_word = self.declare_parameter('wake_word', 'friend').value
        self.wake_re = re.compile(rf'\b{re.escape(wake_word)}\b', re.IGNORECASE)
        self.wake_window = float(self.declare_parameter('wake_window', 8.0).value)
        self.awake_until = 0.0  # time.monotonic() the wake word stops keeping it engaged
        self.create_subscription(String, 'lelamp/speech', self._on_speech, 10)

    def _on_speech(self, msg):
        """A phrase from voice_node: the wake word makes the lamp engaged; while
        engaged, ask llm_node what to do."""
        text = msg.data
        if self.thinking:
            self.get_logger().info(f'heard: "{text}" (still thinking, ignored)')
            return
        wake = self.wake_re.search(text)
        if wake:
            self.awake_until = time.monotonic() + self.wake_window
            if self.brain.name != 'engaged':
                self.brain.go('engaged')
            # a command said in the same breath (keeps a trailing '?': it marks a question)
            text = text[wake.end():].lstrip(' ,.!?').rstrip(' ,.!')
            if not re.search(r'\w', text):
                self.get_logger().info(f'heard the wake word, listening for {self.wake_window:.0f} s')
                return
        elif self.brain.name != 'engaged':
            self.get_logger().info(f'heard: "{text}" (not engaged, ignored)')
            return
        self.get_logger().info(f'heard: "{text}"')
        if not self.parse_client.service_is_ready():
            self.get_logger().warn('llm_node is not running', throttle_duration_sec=10.0)
            return
        self.thinking = True
        self.heard = text
        self.parse_id += 1
        self.parse_sent = time.monotonic()
        self.awake_until = 0.0  # the wake word is used up
        req = ParseCommand.Request(text=text, known_objects=self.memory.names())
        self.parse_client.call_async(req).add_done_callback(
            lambda f, rid=self.parse_id: self._on_commands(f, rid))

        self.brain.go('thinking')

    def _check_parse_timeout(self):
        """Give up on a parse_command call that took too long (Thinking then goes idle)."""
        if self.thinking and time.monotonic() - self.parse_sent > self.parse_timeout:
            self.get_logger().warn(f'parse_command: no answer after {self.parse_timeout:.0f} s, giving up')
            self.thinking = False
            self.parse_id += 1  # its answer is now late

    def _on_commands(self, future, rid):
        if rid != self.parse_id:
            self.get_logger().warn(f'parse_command: late answer ignored ({self._describe(future)})')
            return
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
        # Undo the thinking pose first; each command then sets only what it changes.
        if self.brain.name == 'thinking' and res.commands:
            self.brain.current.restore(self)
        for c in res.commands:
            self.run_command(c.tool, c.arg)
        # No command picked a new state (e.g. only set_light): stop thinking.
        if self.brain.name == 'thinking':
            self.brain.go('idle')

    @staticmethod
    def _describe(future):
        """Short text for a parse_command answer, for logging."""
        try:
            res = future.result()
        except Exception as e:
            return f'failed: {e}'
        return ', '.join(f'{c.tool}({c.arg})' for c in res.commands) or res.message or 'no commands'

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
            self.voice_light = COLORS[arg] or 'off'
        elif tool in ('answer', 'has_seen') and not QUESTION_RE.search(self.heard):
            self.get_logger().info(f'{tool}({arg}) skipped: "{self.heard}" is not a question')
            return False
        elif tool in ('answer', 'has_seen') and self.brain.name in ('nod', 'shake'):
            self.get_logger().info(f'{tool}({arg}) skipped: already answered')
            return False
        elif tool == 'answer' and arg in ('yes', 'no'):
            self.brain.go('nod' if arg == 'yes' else 'shake')
        elif tool == 'has_seen' and arg:
            seen = self.memory.find(arg) is not None
            self.get_logger().info(f"has_seen({arg}): {'yes' if seen else 'no'} "
                                   f"(remembered: {self.memory.summary() or 'nothing'})")
            self.brain.go('nod' if seen else 'shake')
        else:
            self.get_logger().warn(f'unknown command {tool}({arg})')
            return False
        return True
