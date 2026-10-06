"""The LeLamp's behaviors, as a small state machine.

Plain Python (no ROS). Each behavior is one class: `enter()` runs once when the
lamp switches to it, and `update()` runs every tick and returns the name of the
next behavior, or None to stay. All the rules for leaving a behavior live in
that behavior, so adding a new one doesn't touch the others.

`lamp` is whatever owns the state (logic_node): it needs `.motion` (LampMotion),
`.attentive` (bool), `.snapshot()` (remember the objects in view) and
`.focus_target` (x, y, z point for the focus behavior).

Scripted animations subclass Animation and write `script()` as a generator that
yields how many seconds to wait before its next step:

    class Nod(Animation):
        def script(self, lamp):
            lamp.motion.look(0, -30)
            yield 0.4
            lamp.motion.look(0, 0)
            yield 0.4
"""
import random

CENTER_TV_POS = (-2.0, 0.0, 0.35)

PASSIVE_LIGHT = (1.0, 1.0, 1.0, 0.5)
ATTENTIVE_LIGHT = (1.0, 1.0, 1.0, 1.0)


class Behavior:
    color = PASSIVE_LIGHT  # light colour while this behavior runs

    def enter(self, lamp):
        pass

    def update(self, lamp, dt):
        return None


class Animation(Behavior):
    """A behavior driven by a generator script. Goes to `then` when it finishes."""
    then = 'idle'

    def interrupt(self, lamp):
        """Name of a behavior to cut to mid-animation, or None. Checked every tick."""
        return 'attentive' if lamp.attentive else None

    def script(self, lamp):
        yield 0.0

    def enter(self, lamp):
        self._steps = self.script(lamp)
        self._wait = 0.0

    def update(self, lamp, dt):
        nxt = self.interrupt(lamp)
        if nxt:
            return nxt
        self._wait -= dt
        while self._wait <= 0.0:
            try:
                self._wait += next(self._steps)
            except StopIteration:
                return self.then
        return None


class Attentive(Behavior):
    """Someone is looking at the lamp: stand tall and look back at them."""
    color = ATTENTIVE_LIGHT

    def __init__(self, grace=1.0):
        self.grace = grace  # seconds of no face before giving up (detector flickers)

    def enter(self, lamp):
        self.away = 0.0
        lamp.motion.move_to('tall')
        lamp.motion.look_at(*CENTER_TV_POS)

    def update(self, lamp, dt):
        self.away = 0.0 if lamp.attentive else self.away + dt
        return 'idle' if self.away > self.grace else None


class Idle(Behavior):
    """Resting. Gets bored and looks around after a while."""

    def __init__(self, bored_after=5.0):
        self.bored_after = bored_after

    def enter(self, lamp):
        self.elapsed = 0.0
        lamp.motion.move_to('rest')
        lamp.motion.look_at(*CENTER_TV_POS)

    def update(self, lamp, dt):
        if lamp.attentive:
            return 'attentive'
        self.elapsed += dt
        return 'look_around' if self.elapsed > self.bored_after else None


class LookAround(Animation):
    """Sweep the room, stopping at each heading to take a snapshot (the object
    memory), then settle back to idle. Starts from whichever end is nearer."""
    HEADINGS = (-140.0, -95.0, -50.0, -5.0, 40.0, 85.0, 130.0)  # deg, + = lamp's left

    def script(self, lamp):
        headings = list(self.HEADINGS)
        if random.random() < 0.5:
            headings.reverse()
        for yaw in headings:
            lamp.motion.look(yaw + random.uniform(-5.0, 5.0), random.uniform(-15.0, 0.0))
            yield 0.5
            waited = 0.0
            while not lamp.motion.settled() and waited < 3.0:
                yield 0.1
                waited += 0.1
            yield 0.3  # let a camera frame from the new pose arrive
            lamp.snapshot()
            yield random.uniform(0.5, 1.0)


class Focus(Behavior):
    """Look at lamp.focus_target (a remembered object) for a while."""

    def __init__(self, hold=4.0):
        self.hold = hold

    def enter(self, lamp):
        self.elapsed = 0.0
        lamp.motion.move_to('lean_in')
        lamp.motion.look_at(*lamp.focus_target)

    def update(self, lamp, dt):
        self.elapsed += dt
        return 'idle' if self.elapsed > self.hold else None


class StateMachine:
    def __init__(self, lamp, behaviors, start, on_change=None):
        self.lamp = lamp
        self.behaviors = behaviors
        self.on_change = on_change
        self.go(start)

    def go(self, name):
        if self.on_change:
            self.on_change(getattr(self, 'name', None), name)
        self.name = name
        self.current = self.behaviors[name]
        self.current.enter(self.lamp)

    def update(self, dt):
        nxt = self.current.update(self.lamp, dt)
        if nxt:
            self.go(nxt)
