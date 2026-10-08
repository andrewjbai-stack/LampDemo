"""Plays the lamp's sounds on request.

assets/sounds (installed to share/lelamp_logic) has a folder per emotion with a
few different sounds in it, three variations of each:
<emotion>/<emotion>_<sound>_<1..3>.wav (see assets/sounds/README.md). A request
names an emotion ("happy": a random variation of a random happy sound) or one
sound ("happy_trill": a random variation of it). It starts right away; the call
returns without waiting for it to finish, and sounds asked for close together
overlap.

Plays through the default PulseAudio sink (in WSL2 that's the Windows
speakers, through WSLg) with libpulse-simple, like voice_node's mic. Every file
is read into memory at startup so nothing touches the disk when one plays.

    ros2 service call /lelamp/play_sound lelamp_interfaces/srv/PlaySound "{sound: thinking}"
    ros2 service call /lelamp/play_sound lelamp_interfaces/srv/PlaySound "{sound: happy_trill}"

Service: /lelamp/play_sound (lelamp_interfaces/PlaySound)
"""
import ctypes
import os
import random
import re
import threading
import wave
from glob import glob

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from lelamp_interfaces.srv import PlaySound

PA_STREAM_PLAYBACK = 1
PA_SAMPLE_S16LE = 3


class Clip:
    """One WAV file, read into memory as 16-bit samples."""

    def __init__(self, path, volume=1.0):
        self.name = os.path.basename(path)
        with wave.open(path, 'rb') as w:
            if w.getsampwidth() != 2:
                raise ValueError(f'{self.name}: only 16-bit WAV is supported')
            self.rate = w.getframerate()
            self.channels = w.getnchannels()
            data = w.readframes(w.getnframes())
        if volume != 1.0:
            s = np.frombuffer(data, dtype='<i2').astype(np.float32) * volume
            data = np.clip(s, -32768, 32767).astype('<i2').tobytes()
        self.data = data


class Speaker:
    """Plays clips through PulseAudio, each on its own short-lived stream (so they can overlap)."""

    class _Spec(ctypes.Structure):
        _fields_ = [('format', ctypes.c_int), ('rate', ctypes.c_uint32),
                    ('channels', ctypes.c_uint8)]

    class _Attr(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint32)
                    for n in ('maxlength', 'tlength', 'prebuf', 'minreq', 'fragsize')]

    def __init__(self, device=None):
        self.pa = ctypes.CDLL('libpulse-simple.so.0')
        self.pa.pa_simple_new.restype = ctypes.c_void_p
        self.device = device.encode() if device else None

    def play(self, clip):
        """Open a stream for clip (raises if the speaker can't be opened), then
        play it on a background thread."""
        spec = self._Spec(PA_SAMPLE_S16LE, clip.rate, clip.channels)
        # Buffer the whole clip in the server: it's written in one go, so playback
        # still starts at once, and the server isn't woken every few ms to refill.
        size = len(clip.data)
        attr = self._Attr(size, size, size, 0xFFFFFFFF, 0xFFFFFFFF)
        err = ctypes.c_int()
        handle = self.pa.pa_simple_new(
            None, b'lelamp', PA_STREAM_PLAYBACK, self.device, clip.name.encode(),
            ctypes.byref(spec), None, ctypes.byref(attr), ctypes.byref(err))
        if not handle:
            raise RuntimeError(f'could not open the speaker (pulse error {err.value})')
        threading.Thread(target=self._write, args=(ctypes.c_void_p(handle), clip.data),
                         daemon=True).start()

    def _write(self, handle, data):
        err = ctypes.c_int()
        try:
            self.pa.pa_simple_write(handle, data, len(data), ctypes.byref(err))
            self.pa.pa_simple_drain(handle, ctypes.byref(err))
        finally:
            self.pa.pa_simple_free(handle)


class SoundNode(Node):
    def __init__(self):
        super().__init__('sound_node')
        self.declare_parameter('sounds_dir', '')  # '' = share/lelamp_logic/assets/sounds
        self.declare_parameter('volume', 1.0)     # gain applied to every sound
        self.declare_parameter('device', '')      # PulseAudio sink; '' = default

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        sounds_dir = gp('sounds_dir') or os.path.join(
            get_package_share_directory('lelamp_logic'), 'assets', 'sounds')
        self.emotions = self._load(sounds_dir, float(gp('volume')))
        # every sound by its own name too, e.g. 'happy_trill' -> its variations
        self.sounds = {s: v for sounds in self.emotions.values() for s, v in sounds.items()}
        self.speaker = Speaker(gp('device') or None)

        self.create_service(PlaySound, 'lelamp/play_sound', self._on_play)
        self.get_logger().info(f'{len(self.emotions)} emotions, {len(self.sounds)} sounds from {sounds_dir}')
        for emotion, sounds in sorted(self.emotions.items()):
            names = ', '.join(s[len(emotion) + 1:] if s.startswith(emotion + '_') else s for s in sorted(sounds))
            self.get_logger().info(f'  {emotion}: {names}')

    def _load(self, sounds_dir, volume):
        """{emotion: {sound: [Clip, ...]}} from sounds_dir/<emotion>/<sound>_<n>.wav."""
        emotions = {}
        for folder in sorted(glob(os.path.join(sounds_dir, '*', ''))):
            emotion = os.path.basename(os.path.dirname(folder))
            for path in sorted(glob(os.path.join(folder, '*.wav'))):
                sound = re.sub(r'_\d+$', '', os.path.splitext(os.path.basename(path))[0])
                try:
                    emotions.setdefault(emotion, {}).setdefault(sound, []).append(Clip(path, volume))
                except Exception as e:
                    self.get_logger().warn(f'skipping {path}: {e}')
        if not emotions:
            self.get_logger().error(f'no sounds found in {sounds_dir}')
        return emotions

    def _on_play(self, req, res):
        name = req.sound.strip().lower()
        if name in self.emotions:
            clips = random.choice(list(self.emotions[name].values()))
        else:
            clips = self.sounds.get(name)
        if not clips:
            res.success = False
            res.message = (f'unknown sound "{req.sound}"; emotions: {", ".join(sorted(self.emotions))}; '
                           f'sounds: {", ".join(sorted(self.sounds))}')
            self.get_logger().warn(res.message)
            return res
        clip = random.choice(clips)
        try:
            self.speaker.play(clip)
        except Exception as e:
            res.success = False
            res.message = str(e)
            self.get_logger().error(f'{clip.name}: {e}', throttle_duration_sec=5.0)
            return res
        res.success = True
        res.file = clip.name
        res.message = f'playing {clip.name}'
        self.get_logger().info(res.message)
        return res


def main(args=None):
    rclpy.init(args=args)
    node = SoundNode()
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
