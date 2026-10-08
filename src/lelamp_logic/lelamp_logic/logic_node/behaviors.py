"""The LeLamp's behaviors, as a small state machine.

Plain Python (no ROS). Each behavior is one class: `enter()` runs once when the
lamp switches to it, and `update()` runs every tick and returns the name of the
next behavior, or None to stay. All the rules for leaving a behavior live in
that behavior, so adding a new one doesn't touch the others.

`lamp` is whatever owns the state (logic_node): it needs `.motion` (MotionClient,
same calls as LampMotion),
`.attentive` (bool), `.snapshot()` (remember the objects in view),
`.focus_target` (x, y, z point for the focus behavior), `.voice_light`
((r, g, b) asked for by voice, 'off', or None), `.thinking` (waiting for llm_node),
`.awake_until` (time.monotonic() the wake word stops keeping it engaged)
and `.play_sound(name)` (an emotion like 'thinking', or one sound like 'happy_trill';
see assets/sounds/README.md; doesn't wait).

The current behavior also decides the light: `light(lamp, t)` returns RGBA. By
default that's the voice color (if any) at the behavior's brightness; a behavior
that must override the voice color (like Thinking) overrides `light()`.

Scripted animations subclass Animation and write `script()` as a generator that
yields how many seconds to wait before its next step:

    class Nod(Animation):
        def script(self, lamp):
            lamp.motion.look(0, -30)
            yield 0.4
            lamp.motion.look(0, 0)
            yield 0.4
"""
import math
import random
import time


CENTER_TV_POS = (-2.0, 0.0, 0.35)

PASSIVE_LIGHT = (1.0, 1.0, 1.0, 0.5)
ATTENTIVE_LIGHT = (1.0, 1.0, 1.0, 0.7)  # someone looked: a bit brighter than passive
BRIGHT_LIGHT = (1.0, 1.0, 1.0, 1.0)  # engaged, and anything the lamp is busy doing


class Behavior:
    color = PASSIVE_LIGHT  # light colour while this behavior runs

    def light(self, lamp, t):
        """RGBA to show; t = seconds in this behavior. Default: the voice color
        if one was asked for, at this behavior's brightness."""
        r, g, b, a = self.color
        if lamp.voice_light == 'off':
            return r, g, b, 0.0
        if lamp.voice_light:
            r, g, b = lamp.voice_light
        return r, g, b, a

    def enter(self, lamp):
        pass

    def update(self, lamp, dt):
        return None


class Animation(Behavior):
    """A behavior driven by a generator script. Goes to `then` when it finishes."""
    then = 'idle'
    look_for_attentive = False

    def interrupt(self, lamp):
        """Name of a behavior to cut to mid-animation, or None. Checked every tick."""
        return 'attentive' if lamp.attentive and self.look_for_attentive else None

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
    """Someone is looking at the lamp: stand tall and look back at them, a bit
    brighter than passive. Kept looking for engage_after seconds: engaged."""
    color = ATTENTIVE_LIGHT

    def __init__(self, grace=1.0, engage_after=3.0):
        self.grace = grace  # seconds of no face before giving up (detector flickers)
        self.engage_after = engage_after  # seconds of looking before engaged

    def enter(self, lamp):
        lamp.play_sound('notice')
        self.away = 0.0
        self.looked = 0.0
        lamp.motion.move_to('tall')
        lamp.motion.look_at(*CENTER_TV_POS)

    def update(self, lamp, dt):
        if lamp.attentive:
            self.away = 0.0
            self.looked += dt
        else:
            self.away += dt
        if self.away > self.grace:
            return 'idle'
        return 'engaged' if self.looked >= self.engage_after else None


class Engaged(Behavior):
    """Second layer of attention, full brightness: they kept looking (from
    Attentive) or said the wake word (until lamp.awake_until). Only while engaged
    are heard phrases sent to llm_node. Goes back to idle once nobody is looking
    and the wake word's time is up."""
    color = BRIGHT_LIGHT

    def __init__(self, grace=1.0):
        self.grace = grace

    def enter(self, lamp):
        lamp.play_sound('wake')
        self.away = 0.0
        lamp.motion.move_to('tall')  # same pose as Attentive (the wake word skips it)
        lamp.motion.look_at(*CENTER_TV_POS)

    def update(self, lamp, dt):
        held = lamp.attentive or time.monotonic() < lamp.awake_until
        self.away = 0.0 if held else self.away + dt
        return 'idle' if self.away > self.grace else None


class Idle(Behavior):
    """Resting. Gets bored and looks around after a while."""
    color = PASSIVE_LIGHT

    def __init__(self, bored_after=15.0):
        self.bored_after = bored_after

    def enter(self, lamp):
        lamp.play_sound('sleep')

        self.elapsed = 0.0
        lamp.motion.move_to('rest')
        lamp.motion.look_at(*CENTER_TV_POS)

    def update(self, lamp, dt):
        if lamp.attentive:
            return 'attentive'
        self.elapsed += dt
        return 'bored' if self.elapsed > self.bored_after else None


class LookAround(Animation):
    """Sweep the room, stopping at each heading to take a snapshot (the object
    memory), then settle back to idle. Starts from whichever end is nearer."""
    HEADINGS = (-140.0, -95.0, -50.0, -5.0, 40.0, 85.0, 130.0)  # deg, + = lamp's left

    color = BRIGHT_LIGHT

    def script(self, lamp):
        headings = list(self.HEADINGS)
        if random.random() < 0.5:
            headings.reverse()
        for yaw in headings:
            lamp.play_sound('ack')

            lamp.motion.look(yaw + random.uniform(-5.0, 5.0), random.uniform(-15.0, 0.0))
            yield 0.5
            waited = 0.0
            while not lamp.motion.settled() and waited < 3.0:
                yield 0.1
                waited += 0.1
            yield 0.3  # let a camera frame from the new pose arrive
            lamp.snapshot()
            yield random.uniform(0.5, 1.0)

class Bored(Animation):
    """Something to do while nobody's around. Each time, plays one of a few
    bored animations at random, then goes back to idle:

    - glance: two quick looks near wherever the lamp was already looking
    - stretch: reach the arm out and swing all the way to one side, then all
      the way to the other, then come back
    """
    color = PASSIVE_LIGHT

    look_for_attentive = True

    STRETCH_POSTURE = (-0.6, 0.0)  # (shoulder, elbow) arm out straight
    STRETCH_YAW = 90.0          # deg each side (base yaw limit is ~149)

    def heading(self, lamp):
        """Where the lamp is looking now, as (yaw, pitch) degrees for look()."""
        return lamp.motion.heading()

    def wait_settled(self, lamp, timeout=8.0):
        """Yield until the lamp has reached its target and stopped, or timeout."""
        yield 0.4
        waited = 0.4
        while not lamp.motion.settled() and waited < timeout:
            yield 0.1
            waited += 0.1

    def script(self, lamp):
        lamp.play_sound('sigh')
        yield from random.choice((self.glance, self.stretch))(lamp)

    def glance(self, lamp):
        yaw, pitch = self.heading(lamp)
        side = random.choice((-1.0, 1.0))
        for _ in range(2):
            lamp.motion.look(yaw + side * random.uniform(15.0, 30.0),
                             pitch + random.uniform(-10.0, 10.0))
            side = -side if random.random() < 0.7 else side  # usually the other way
            yield random.uniform(1.0, 2.0)
        lamp.motion.look(yaw, pitch)
        yield 0.6

    def stretch(self, lamp):
        yaw, pitch = self.heading(lamp)
        posture = lamp.motion.posture
        side = random.choice((-1.0, 1.0))
        lamp.motion.move_to(*self.STRETCH_POSTURE)
        for s in (side, -side):
            lamp.motion.look(s * self.STRETCH_YAW, 10.0)
            yield from self.wait_settled(lamp)
            yield random.uniform(0.8, 1.2)  # hold the stretch
        lamp.motion.move_to(*posture)
        lamp.motion.look(yaw, pitch)
        lamp.play_sound('happy')

        yield from self.wait_settled(lamp)


class Focus(Behavior):
    """Look at lamp.focus_target (a remembered object) for a while."""

    color = BRIGHT_LIGHT

    def __init__(self, hold=4.0):
        self.hold = hold

    def enter(self, lamp):
        self.elapsed = 0.0
        lamp.motion.move_to('lean_in')
        lamp.motion.look_at(*lamp.focus_target)

    def update(self, lamp, dt):
        self.elapsed += dt
        return 'idle' if self.elapsed > self.hold else None


class Obey(Behavior):
    """Doing what it was told (logic_node sets the motion before switching here).
    Holds that pose for a while, without getting bored or attentive."""
    
    color = BRIGHT_LIGHT

    def __init__(self, hold=8.0):
        self.hold = hold

    def enter(self, lamp):
        lamp.play_sound('ack')

        self.elapsed = 0.0

    def update(self, lamp, dt):
        self.elapsed += dt
        return 'idle' if self.elapsed > self.hold else None

class Thinking(Behavior):
    """The lamp is waiting for the llm to respond, and assumes a thinking pose.
    The light pulses bluish white, whatever color was asked for by voice.
    The answer picks the next state; if lamp.thinking clears without one
    (logic_node gave up waiting), go back to idle."""

    def light(self, lamp, t):
        v = 0.65 + 0.25 * math.sin(2.0 * math.pi * 0.75 * t)
        return v, v, 1.0, 1.0

    DIP = 12.0  # deg the head drops from where it was looking

    def enter(self, lamp):
        # The pose from before thinking, for restore().
        self.look_target = lamp.motion.look_target
        self.posture = lamp.motion.posture
        yaw, pitch = lamp.motion.heading()
        lamp.motion.move_to('sit_back')
        lamp.motion.look(yaw, pitch - self.DIP)
        lamp.play_sound('thinking')

    def restore(self, lamp):
        """Go back to the pose from before thinking, so a command that sets
        only the look or only the posture doesn't keep half the thinking pose."""
        t = self.look_target
        if t[0] == 'point':
            lamp.motion.look_at(*t[1:])
        else:
            lamp.motion.look(math.degrees(t[1]), math.degrees(t[2]))
        lamp.motion.move_to(*self.posture)

    def update(self, lamp, dt):
        return None if lamp.thinking else 'idle'

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
        self.t = 0.0  # seconds in the current behavior
        self.current = self.behaviors[name]
        self.current.enter(self.lamp)

    def update(self, dt):
        self.t += dt
        nxt = self.current.update(self.lamp, dt)
        if nxt:
            self.go(nxt)
