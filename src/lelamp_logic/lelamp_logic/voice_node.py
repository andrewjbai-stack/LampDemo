"""Speech to text from the microphone, published as soon as a phrase ends.

Reads the default PulseAudio source (in WSL2 that's the Windows mic, through
WSLg) at 16 kHz. Silero VAD (the model that ships with faster-whisper) marks
where speech starts and stops; when a phrase ends it's transcribed with
faster-whisper on the CPU and published.

    ros2 topic echo /lelamp/speech

Outputs: /lelamp/speech  (std_msgs/String) one message per phrase
"""
import ctypes
import os
import queue
import threading
import time
from collections import deque

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

import numpy as np
import onnxruntime
import rclpy
from faster_whisper import WhisperModel
from faster_whisper.utils import get_assets_path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

RATE = 16000
CHUNK = 512  # samples per VAD step (32 ms), what Silero expects at 16 kHz
CONTEXT = 64  # samples of the previous chunk Silero wants in front of each one


class Mic:
    """Blocking reads from PulseAudio through libpulse-simple (no extra packages)."""

    class _Spec(ctypes.Structure):
        _fields_ = [('format', ctypes.c_int), ('rate', ctypes.c_uint32),
                    ('channels', ctypes.c_uint8)]

    class _Attr(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint32)
                    for n in ('maxlength', 'tlength', 'prebuf', 'minreq', 'fragsize')]

    def __init__(self, device=None):
        self.pa = ctypes.CDLL('libpulse-simple.so.0')
        self.pa.pa_simple_new.restype = ctypes.c_void_p
        spec = self._Spec(5, RATE, 1)  # PA_SAMPLE_FLOAT32LE, mono
        # Small fragments so audio arrives every chunk, not in big lumps.
        attr = self._Attr(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, CHUNK * 4)
        err = ctypes.c_int()
        self.handle = self.pa.pa_simple_new(
            None, b'lelamp', 2, device.encode() if device else None, b'voice',
            ctypes.byref(spec), None, ctypes.byref(attr), ctypes.byref(err))
        if not self.handle:
            raise RuntimeError(f'could not open the microphone (pulse error {err.value})')
        self.buf = ctypes.create_string_buffer(CHUNK * 4)

    def read(self):
        if not self.handle:
            raise RuntimeError('microphone is closed')
        err = ctypes.c_int()
        if self.pa.pa_simple_read(ctypes.c_void_p(self.handle), self.buf,
                                  len(self.buf), ctypes.byref(err)) < 0:
            raise RuntimeError(f'microphone read failed (pulse error {err.value})')
        return np.frombuffer(self.buf.raw, dtype=np.float32).copy()

    def close(self):
        if self.handle:
            self.pa.pa_simple_free(ctypes.c_void_p(self.handle))
            self.handle = None


class Vad:
    """Silero VAD run one chunk at a time, keeping its state between chunks."""

    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            os.path.join(get_assets_path(), 'silero_vad_v6.onnx'),
            providers=['CPUExecutionProvider'], sess_options=opts)
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, chunk):
        x = np.concatenate([self.context, chunk])[None, :]
        self.context = chunk[-CONTEXT:]
        out, self.h, self.c = self.session.run(None, {'input': x, 'h': self.h, 'c': self.c})
        return float(out.ravel()[0])


class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')
        self.declare_parameter('device', '')  # PulseAudio source; '' = default
        self.declare_parameter('model', 'base.en')
        self.declare_parameter('cpu_threads', 4)
        self.declare_parameter('start_prob', 0.5)  # VAD prob that starts a phrase
        self.declare_parameter('stop_prob', 0.35)  # below this counts as silence
        self.declare_parameter('end_silence_s', 0.5)  # this much silence ends a phrase
        self.declare_parameter('pre_roll_s', 0.3)  # audio kept from before the start
        self.declare_parameter('min_speech_s', 0.25)  # shorter blips are dropped
        self.declare_parameter('max_phrase_s', 15.0)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.p = {n: gp(n) for n in ('start_prob', 'stop_prob', 'end_silence_s', 'pre_roll_s',
                                     'min_speech_s', 'max_phrase_s')}

        self.pub = self.create_publisher(String, 'lelamp/speech', 10)

        model_dir = os.path.expanduser('~/.lelamp/models/whisper')
        self.get_logger().info(f"loading whisper '{gp('model')}' (cpu, int8)...")
        self.whisper = WhisperModel(gp('model'), device='cpu', compute_type='int8',
                                    cpu_threads=int(gp('cpu_threads')),
                                    download_root=model_dir)
        self.vad = Vad()
        self.device = gp('device') or None
        self.mic = Mic(self.device)

        self.phrases = queue.Queue()
        self.running = True
        threading.Thread(target=self._listen, daemon=True).start()
        threading.Thread(target=self._transcribe, daemon=True).start()
        self.get_logger().info('listening')

    def _listen(self):
        """Cut the mic stream into phrases and hand them to _transcribe."""
        p = self.p
        step = CHUNK / RATE
        pre_roll = deque(maxlen=max(1, int(p['pre_roll_s'] / step)))
        phrase, speech_s, silence_s = None, 0.0, 0.0
        backoff = 1.0
        while self.running:
            try:
                chunk = self.mic.read()
            except Exception as e:
                # Wait, reopen the mic and start clean; back off while it keeps failing.
                self.get_logger().error(f'{e}; reopening the mic in {backoff:.0f} s',
                                        throttle_duration_sec=10.0)
                try:
                    self.mic.close()
                except Exception:
                    pass
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                try:
                    self.mic = Mic(self.device)
                except Exception as e:
                    self.get_logger().error(str(e), throttle_duration_sec=10.0)
                    continue
                self.vad.reset()
                pre_roll.clear()
                phrase = None
                continue
            backoff = 1.0
            prob = self.vad(chunk)
            if phrase is None:
                pre_roll.append(chunk)
                if prob >= p['start_prob']:
                    phrase, speech_s, silence_s = list(pre_roll), step, 0.0
                    pre_roll.clear()
                continue
            phrase.append(chunk)
            if prob < p['stop_prob']:
                silence_s += step
            else:
                silence_s = 0.0
                speech_s += step
            if silence_s >= p['end_silence_s'] or len(phrase) * step >= p['max_phrase_s']:
                if speech_s >= p['min_speech_s']:
                    self.phrases.put((time.monotonic(), np.concatenate(phrase)))
                phrase = None

    def _transcribe(self):
        while self.running:
            ended, audio = self.phrases.get()
            try:
                segments, _ = self.whisper.transcribe(
                    audio, language='en', beam_size=1, condition_on_previous_text=False,
                    without_timestamps=True)
                # Drop what whisper itself thinks is noise (it likes to invent "Thank you.").
                text = ' '.join(s.text.strip() for s in segments
                                if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0).strip()
            except Exception as e:
                # segments is lazy, so errors can come from the join too; skip the phrase.
                self.get_logger().error(f'transcribe failed: {e}', throttle_duration_sec=5.0)
                continue
            if not text:
                continue
            self.pub.publish(String(data=text))
            self.get_logger().info(
                f'heard: "{text}" ({len(audio) / RATE:.1f} s, '
                f'+{time.monotonic() - ended:.2f} s after it ended)')


def main(args=None):
    rclpy.init(args=args)
    node = VoiceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.running = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
