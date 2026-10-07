"""The LeLamp's body: eases the joints toward look and move goals on its own
timer, so motion stays smooth whatever the other nodes are doing.

Other nodes set goals through two services and read back what it's doing on
/lelamp/motion_state (logic_node wraps all of this in motion_client.py).

Services: /lelamp/look  (lelamp_interfaces/Look: aim at a point or by angle)
          /lelamp/move  (lelamp_interfaces/Move: posture by name, angles or reach)
Outputs:  /lelamp/joint_commands (sensor_msgs/JointState, target positions)
          /lelamp/motion_state   (lelamp_interfaces/MotionState, every tick)

LampMotion does the easing, and works without ROS too:

    motion = LampMotion()
    motion.look_at(0.4, 0.2, 0.1)   # aim the head at a point (base_link, metres)
    motion.look(30, -10)            # or aim by angle: 30 deg left, 10 deg down
    motion.move_to('lean_in')       # body posture, by name or (shoulder, elbow)
    motion.reach(0.35, 0.55)        # put the head pivot at (forward, height)
    names, positions = motion.step(dt)   # call every tick, publish the result

Look and move are independent. Moves set the shoulder and elbow; looks set base
yaw and head pitch, and are re-solved every tick for the current posture, so the
head keeps aiming at its target while the body moves.

Geometry from lelamp.urdf (see lelamp_arm_and_ik.md). The lamp's "forward" is
base -X, positive base yaw turns it to its left (counter-clockwise from above),
and positive head pitch tilts the head down.
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lelamp_interfaces.msg import MotionState
from lelamp_interfaces.srv import Look, Move

JOINTS = ['base_yaw_joint', 'shoulder_pitch_joint', 'elbow_pitch_joint',
          'neck_yaw_joint', 'head_pitch_joint']

# name: (lower, upper, max speed rad/s) from the URDF
LIMITS = {
    'base_yaw_joint': (-2.60, 2.60, 1.50),
    'shoulder_pitch_joint': (-0.75, 1.05, 0.95),
    'elbow_pitch_joint': (-1.85, 0.40, 1.15),
    'neck_yaw_joint': (-1.35, 1.35, 1.60),
    'head_pitch_joint': (-0.90, 0.70, 1.45),
}

SHOULDER_Z = 0.115   # shoulder pivot height above the desk
L1 = 0.290           # shoulder -> elbow
L2 = 0.320           # elbow -> head pivot (upper arm 0.250 + neck 0.070)

# (shoulder_pitch, elbow_pitch). Negative shoulder leans forward.
POSTURES = {
    'rest': (0.25, -0.90),
    'lean_in': (-0.45, -0.70),
    'sit_back': (0.60, -1.20),
    'tall': (0.00, -0.15),
    'shy': (0.90, -1.80),
}


def _clamp(name, value):
    lo, hi, _ = LIMITS[name]
    return max(lo, min(hi, value))


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def head_pivot(shoulder, elbow):
    """Head pivot (forward, height above the shoulder) for a posture."""
    phi = shoulder + elbow
    u = -(L1 * math.sin(shoulder) + L2 * math.sin(phi))
    z = L1 * math.cos(shoulder) + L2 * math.cos(phi)
    return u, z


def aim(shoulder, elbow, x, y, z):
    """(base_yaw, head_pitch) that point the head at (x, y, z) in base_link."""
    u_p, z_p = head_pivot(shoulder, elbow)
    base_yaw = _wrap(math.atan2(y, x) - math.pi)
    down = math.atan2(z_p - (z - SHOULDER_Z), math.hypot(x, y) - u_p)
    return base_yaw, down + shoulder + elbow


def reach_posture(forward, height):
    """(shoulder, elbow) that put the head pivot at (forward, height above desk).

    Returns None when the point is out of reach.
    """
    x, z = -forward, height - SHOULDER_Z
    c2 = (x * x + z * z - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1.0:
        return None
    elbow = -math.acos(c2)  # the negative branch fits the elbow limits
    shoulder = math.atan2(x, z) - math.atan2(L2 * math.sin(elbow), L1 + L2 * math.cos(elbow))
    return shoulder, elbow


class _Spring:
    """Critically damped spring: eases in and out, never overshoots."""

    def __init__(self, x, stiffness, max_speed):
        self.x, self.v = x, 0.0
        self.w, self.max_speed = stiffness, max_speed

    def step(self, target, dt):
        self.v += (self.w * self.w * (target - self.x) - 2.0 * self.w * self.v) * dt
        self.v = max(-self.max_speed, min(self.max_speed, self.v))
        self.x += self.v * dt
        return self.x


class LampMotion:
    def __init__(self, posture='rest', gaze_stiffness=5.0, posture_stiffness=2.5,
                 breathing=True):
        self.posture = POSTURES[posture]
        self.look_target = None          # ('point', x, y, z) or ('angle', yaw, up)
        self.look(0.0, -20.0)
        self.breathing = breathing
        self.t = 0.0

        stiff = {'shoulder_pitch_joint': posture_stiffness,
                 'elbow_pitch_joint': posture_stiffness}
        start = {'shoulder_pitch_joint': self.posture[0], 'elbow_pitch_joint': self.posture[1]}
        # 90% of the URDF speed, so the sim's own rate limit never clips the easing
        self.springs = {n: _Spring(start.get(n, 0.0), stiff.get(n, gaze_stiffness),
                                   0.9 * LIMITS[n][2]) for n in JOINTS}

    # --- the API ---

    def look_at(self, x, y, z):
        """Keep the head aimed at a point in base_link (metres)."""
        self.look_target = ('point', float(x), float(y), float(z))

    def look(self, yaw_deg, pitch_deg):
        """Aim by angle. yaw: + = lamp's left; pitch: + = up, 0 = level."""
        self.look_target = ('angle', math.radians(yaw_deg), math.radians(pitch_deg))

    def move_to(self, posture, elbow=None):
        """Body posture by name (see POSTURES) or as (shoulder, elbow) radians."""
        if elbow is None:
            posture = POSTURES[posture]
        else:
            posture = (posture, elbow)
        self.posture = (_clamp('shoulder_pitch_joint', posture[0]),
                        _clamp('elbow_pitch_joint', posture[1]))

    def reach(self, forward, height):
        """Move the head pivot to (forward of the base, height above desk). False if out of reach."""
        posture = reach_posture(forward, height)
        if posture is None:
            return False
        self.move_to(*posture)
        return True

    def settled(self, tol=0.02, pos_tol=0.05):
        """True once every joint is nearly still (under tol rad/s) and within
        pos_tol rad (about 3 deg, enough to cover breathing) of its target.
        Targets are clamped to the joint limits, so an out-of-reach look still
        settles at the limit."""
        goal = self.targets()
        return all(abs(s.v) < tol and abs(s.x - _clamp(n, goal[n])) < pos_tol
                   for n, s in self.springs.items())

    def heading(self):
        """Where the head is aimed now, as (yaw, pitch) degrees for look()."""
        t = self.look_target
        if t[0] == 'point':
            shoulder = self.springs['shoulder_pitch_joint'].x
            elbow = self.springs['elbow_pitch_joint'].x
            yaw, pitch = aim(shoulder, elbow, *t[1:])
            up = shoulder + elbow - pitch
        else:
            _, yaw, up = t
        return math.degrees(yaw), math.degrees(up)

    # --- per tick ---

    def targets(self):
        """Joint targets this tick, before easing and breathing."""
        shoulder, elbow = self.posture
        # Aim with the posture the body actually has right now, so the gaze holds
        # while the body is still moving.
        now_s = self.springs['shoulder_pitch_joint'].x
        now_e = self.springs['elbow_pitch_joint'].x
        if self.look_target[0] == 'point':
            yaw, pitch = aim(now_s, now_e, *self.look_target[1:])
        else:
            _, yaw, up = self.look_target
            pitch = now_s + now_e - up
        return {'base_yaw_joint': yaw, 'shoulder_pitch_joint': shoulder,
                'elbow_pitch_joint': elbow, 'neck_yaw_joint': 0.0,
                'head_pitch_joint': pitch}

    def step(self, dt):
        """Advance the easing by dt seconds. Returns (names, positions) to publish."""
        self.t += dt
        goal = self.targets()
        if self.breathing:
            goal['shoulder_pitch_joint'] += 0.03 * math.sin(0.5 * self.t)
            goal['elbow_pitch_joint'] += 0.03 * math.sin(0.5 * self.t + 0.6)
        positions = [_clamp(n, self.springs[n].step(_clamp(n, goal[n]), dt)) for n in JOINTS]
        return list(JOINTS), positions


class MotionNode(Node):
    def __init__(self):
        super().__init__('motion_node')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('idle_motion', True)  # breathing on top of every pose

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.dt = 1.0 / float(gp('rate_hz'))
        self.motion = LampMotion(breathing=bool(gp('idle_motion')))
        self.goal_id = 0  # counts accepted Look and Move goals

        self.create_service(Look, 'lelamp/look', self._on_look)
        self.create_service(Move, 'lelamp/move', self._on_move)
        self.cmd_pub = self.create_publisher(JointState, 'lelamp/joint_commands', 10)
        self.state_pub = self.create_publisher(MotionState, 'lelamp/motion_state', 10)
        self.create_timer(self.dt, self._tick)

    def _accept(self, res, message):
        self.goal_id += 1
        res.success, res.message, res.goal_id = True, message, self.goal_id
        return res

    def _on_look(self, req, res):
        if req.mode == 'point':
            self.motion.look_at(req.x, req.y, req.z)
            return self._accept(res, f'look at ({req.x:.2f}, {req.y:.2f}, {req.z:.2f})')
        if req.mode == 'angle':
            self.motion.look(req.yaw, req.pitch)
            return self._accept(res, f'look {req.yaw:.0f}, {req.pitch:.0f} deg')
        res.success, res.message = False, f'unknown look mode "{req.mode}"'
        return res

    def _on_move(self, req, res):
        if req.mode == 'posture':
            if req.posture not in POSTURES:
                res.success, res.message = False, f'unknown posture "{req.posture}"'
                return res
            self.motion.move_to(req.posture)
            return self._accept(res, req.posture)
        if req.mode == 'angles':
            self.motion.move_to(req.shoulder, req.elbow)
            return self._accept(res, f'shoulder {req.shoulder:.2f}, elbow {req.elbow:.2f}')
        if req.mode == 'reach':
            if not self.motion.reach(req.forward, req.height):
                res.success, res.message = False, 'out of reach'
                return res
            return self._accept(res, f'reach {req.forward:.2f}, {req.height:.2f}')
        res.success, res.message = False, f'unknown move mode "{req.mode}"'
        return res

    def _tick(self):
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name, cmd.position = self.motion.step(self.dt)
        self.cmd_pub.publish(cmd)

        state = MotionState(goal_id=self.goal_id, settled=self.motion.settled())
        state.look_yaw, state.look_pitch = self.motion.heading()
        target = self.motion.look_target
        state.look_mode = target[0]
        if target[0] == 'point':
            state.look_point.x, state.look_point.y, state.look_point.z = target[1:]
        state.shoulder, state.elbow = self.motion.posture
        self.state_pub.publish(state)


def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
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
