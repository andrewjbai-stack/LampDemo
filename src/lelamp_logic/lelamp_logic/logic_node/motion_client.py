"""logic_node's handle on motion_node, with the same calls behaviors used on
LampMotion: look_at, look, move_to, reach, settled, heading, look_target, posture.

Goals go out as Look/Move service calls and return at once. Reads come from
the latest /lelamp/motion_state, except look_target and posture, which are kept
here as soon as they're asked for so a script can read back what it just set.

settled() only counts once motion_node has taken every goal sent so far (by
goal_id), so it never reports the previous goal as done. Goals asked for before
motion_node is up are held (newest per service) and sent when it appears.
"""
import math

from lelamp_interfaces.msg import MotionState
from lelamp_interfaces.srv import Look, Move
from lelamp_logic.motion_node import POSTURES, reach_posture


class MotionClient:
    def __init__(self, node):
        self.node = node
        self.look_client = node.create_client(Look, 'lelamp/look')
        self.move_client = node.create_client(Move, 'lelamp/move')
        node.create_subscription(MotionState, 'lelamp/motion_state', self._on_state, 10)
        self.state = None  # latest MotionState
        self.pending = 0   # goals sent with no answer yet
        self.goal_id = 0   # newest goal_id motion_node has given us
        self.waiting = {}  # client -> newest request, while motion_node isn't up

        self.look_target = ('angle', 0.0, math.radians(-20.0))  # LampMotion's start
        self.posture = POSTURES['rest']

    # --- goals ---

    def look_at(self, x, y, z):
        """Keep the head aimed at a point in base_link (metres)."""
        self.look_target = ('point', float(x), float(y), float(z))
        self._send(self.look_client, Look.Request(mode='point', x=float(x), y=float(y), z=float(z)))

    def look(self, yaw_deg, pitch_deg):
        """Aim by angle. yaw: + = lamp's left; pitch: + = up, 0 = level."""
        self.look_target = ('angle', math.radians(yaw_deg), math.radians(pitch_deg))
        self._send(self.look_client, Look.Request(mode='angle', yaw=float(yaw_deg), pitch=float(pitch_deg)))

    def move_to(self, posture, elbow=None):
        """Body posture by name (see POSTURES) or as (shoulder, elbow) radians."""
        if elbow is None:
            self.posture = POSTURES[posture]
            req = Move.Request(mode='posture', posture=posture)
        else:
            self.posture = (float(posture), float(elbow))
            req = Move.Request(mode='angles', shoulder=float(posture), elbow=float(elbow))
        self._send(self.move_client, req)

    def reach(self, forward, height):
        """Move the head pivot to (forward of the base, height above desk). False if out of reach."""
        posture = reach_posture(forward, height)
        if posture is None:
            return False
        self.posture = posture
        self._send(self.move_client, Move.Request(mode='reach', forward=float(forward), height=float(height)))
        return True

    # --- reads ---

    def settled(self):
        """True once motion_node has every goal sent so far and has stopped moving."""
        s = self.state
        return s is not None and self.pending == 0 and s.goal_id >= self.goal_id and s.settled

    def heading(self):
        """Where the head is aimed, as (yaw, pitch) degrees for look()."""
        if self.look_target[0] == 'angle':
            return math.degrees(self.look_target[1]), math.degrees(self.look_target[2])
        if self.state is None:
            return 0.0, 0.0
        return self.state.look_yaw, self.state.look_pitch

    # --- plumbing ---

    def _send(self, client, req):
        if not client.service_is_ready():
            # Keep only the newest goal per service; sent once motion_node is up.
            self.waiting[client] = req
            self.node.get_logger().warn('motion_node is not running', throttle_duration_sec=5.0)
            return
        self.pending += 1
        client.call_async(req).add_done_callback(self._on_answer)

    def _on_answer(self, future):
        self.pending -= 1
        try:
            res = future.result()
        except Exception as e:
            self.node.get_logger().error(f'motion call failed: {e}')
            return
        if not res.success:
            self.node.get_logger().warn(f'motion: {res.message}')
            return
        self.goal_id = max(self.goal_id, res.goal_id)

    def _on_state(self, msg):
        self.state = msg
        for client in [c for c in self.waiting if c.service_is_ready()]:
            self._send(client, self.waiting.pop(client))
