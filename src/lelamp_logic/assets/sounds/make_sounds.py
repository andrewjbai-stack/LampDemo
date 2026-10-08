"""Synthesize cute robot sounds for LeLamp. numpy + stdlib only, CPU, a few seconds to run.
Writes 22.05 kHz mono 16-bit WAVs: one folder per emotion, several different sounds in each,
3 variations of every sound: <emotion>/<emotion>_<sound>_<1..3>.wav
(1 = a bit lower and slower, 2 = original, 3 = a bit higher and quicker). All yours, no license strings."""
import os, sys, wave
import numpy as np

SR = 22050
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
VARIANTS = [(0.94, 1.06), (1.0, 1.0), (1.06, 0.94)]  # (pitch multiplier, duration multiplier)
P, T = 1.0, 1.0  # current variant
rng = None

# ---------- building blocks ----------
def env(n, a=0.01, r=0.06):
    e = np.ones(n)
    na, nr = min(int(a * SR), n // 3), min(int(r * SR), n // 2)
    if na: e[:na] = np.linspace(0, 1, na)
    if nr: e[-nr:] *= np.linspace(1, 0, nr)
    return e

def tone(f0, f1, dur, vib=0.0, vib_hz=18, curve=1.0, bright=0.25, a=0.01, r=0.06, fm=0.0, fm_ratio=2.0):
    """Pitch glide f0->f1 with optional vibrato and FM wobble; sine + a little 2nd/3rd harmonic = 'droid' tone."""
    n = max(int(dur * T * SR), 32)
    x = np.linspace(0, 1, n) ** curve
    f = (f0 + (f1 - f0) * x) * P
    t = np.arange(n) / SR
    f = f * (1 + vib * np.sin(2 * np.pi * vib_hz * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    if fm:
        ph = ph + fm * np.sin(ph * fm_ratio)
    s = np.sin(ph) + bright * np.sin(2 * ph) + bright * 0.4 * np.sin(3 * ph)
    return s * env(n, a, r)

def notes(freqs, dur, gap_s=0.03, **kw):
    """Steady notes in a row (each note can be a (f0, f1) glide)."""
    parts = []
    for f in freqs:
        f0, f1 = f if isinstance(f, tuple) else (f, f)
        parts += [tone(f0, f1, dur, **kw), gap(gap_s)]
    return seq(*parts)

def babble(n, lo, hi, dur=0.06, gap_lo=0.02, gap_hi=0.06, **kw):
    """R2-style random chatter."""
    parts = []
    for _ in range(n):
        f = rng.uniform(lo, hi)
        parts += [tone(f, f * rng.uniform(0.8, 1.3), dur * rng.uniform(0.7, 1.3), a=0.004, r=0.02, **kw),
                  gap(rng.uniform(gap_lo, gap_hi))]
    return seq(*parts)

def noise(dur, start=0.3, end=0.0, lp=0.9):
    """Breathy noise, simple one-pole low-pass."""
    n = max(int(dur * T * SR), 32)
    w = rng.standard_normal(n)
    out = np.empty(n); acc = 0.0
    for i in range(n):  # short buffers only
        acc = lp * acc + (1 - lp) * w[i]; out[i] = acc
    return out / (np.max(np.abs(out)) + 1e-9) * np.linspace(start, end, n)

def gap(d):
    return np.zeros(max(int(d * T * SR), 1))

def seq(*parts):
    return np.concatenate(parts)

def mix(a, b, gain_b=1.0):
    n = max(len(a), len(b)); out = np.zeros(n)
    out[:len(a)] += a; out[:len(b)] += b * gain_b
    return out

def save(emotion, name, s, vol=0.6, variant=1):
    s = s / (np.max(np.abs(s)) + 1e-9) * vol
    d = os.path.join(OUT, emotion); os.makedirs(d, exist_ok=True)
    with wave.open(os.path.join(d, f"{emotion}_{name}_{variant}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(s, -1, 1) * 32767).astype(np.int16).tobytes())

# ---------- sound designs: emotion -> {name: (builder, volume)} ----------
SOUNDS = {
    "wake": {  # engaged: bright, rising, "I'm here!"
        "bweeoop":  (lambda: seq(tone(500, 1400, 0.16, curve=0.6), gap(0.03), tone(1100, 1800, 0.09)), 0.6),
        "powerup":  (lambda: notes([523, 659, 784, 1047], 0.07, 0.015, bright=0.15), 0.55),
        "hello":    (lambda: seq(tone(700, 1200, 0.1), gap(0.04), tone(900, 1500, 0.08), tone(1500, 1100, 0.12, vib=0.04, vib_hz=20)), 0.6),
        "sparkle":  (lambda: seq(tone(600, 1800, 0.2, curve=0.5, fm=0.6, fm_ratio=3.0), babble(4, 1500, 2300, dur=0.04)), 0.55),
    },
    "notice": {  # attentive: soft, quiet, curious glance
        "hm":       (lambda: tone(700, 1000, 0.14, bright=0.1), 0.4),
        "blip":     (lambda: tone(1200, 1250, 0.05, bright=0.05, r=0.03), 0.35),
        "oh":       (lambda: tone(600, 850, 0.12, curve=0.4, bright=0.1, vib=0.02, vib_hz=12), 0.4),
        "tick":     (lambda: notes([1000, 1300], 0.04, 0.03, bright=0.05, r=0.02), 0.35),
    },
    "thinking": {  # waiting on the LLM: gentle, busy, loopable
        "burble":   (lambda: babble(6, 600, 1100, dur=0.08, gap_lo=0.08, gap_hi=0.16, bright=0.1), 0.35),
        "computing":(lambda: notes([880, 990, 880, 1175, 880, 990, 1320, 880], 0.06, 0.06, bright=0.05, r=0.02), 0.3),
        "hmmm":     (lambda: tone(500, 520, 1.0, vib=0.04, vib_hz=6, bright=0.1, a=0.1, r=0.2), 0.3),
        "whirr":    (lambda: tone(400, 700, 1.1, fm=1.2, fm_ratio=0.25, bright=0.1, a=0.15, r=0.25), 0.3),
    },
    "ack": {  # command understood: quick, happy, affirmative
        "blip_blip":(lambda: seq(tone(900, 1000, 0.07), gap(0.04), tone(1300, 1450, 0.09)), 0.55),
        "okay":     (lambda: seq(tone(800, 800, 0.08), gap(0.03), tone(1000, 1200, 0.12, curve=0.5)), 0.55),
        "yep":      (lambda: tone(900, 1500, 0.1, curve=0.4), 0.55),
        "roger":    (lambda: notes([784, 1047, 1319], 0.06, 0.02, bright=0.15), 0.5),
    },
    "confused": {  # didn't understand: falling, wobbly, questioning
        "uh_oh":    (lambda: seq(tone(900, 800, 0.12), gap(0.05), tone(700, 380, 0.25, vib=0.03, vib_hz=9)), 0.55),
        "huh":      (lambda: tone(500, 950, 0.25, curve=2.5, vib=0.05, vib_hz=14), 0.55),
        "wobble":   (lambda: tone(700, 500, 0.45, vib=0.12, vib_hz=7, fm=0.4), 0.5),
        "scramble": (lambda: seq(babble(5, 400, 1400, dur=0.05, gap_lo=0.01, gap_hi=0.03), tone(600, 350, 0.2)), 0.5),
    },
    "happy": {  # success / delight: trills, arpeggios, bouncy
        "trill":    (lambda: seq(tone(800, 1600, 0.3, vib=0.08, vib_hz=22, curve=0.7), gap(0.02), tone(1600, 2000, 0.08)), 0.6),
        "giggle":   (lambda: notes([(1200, 1400), (1100, 1300), (1300, 1500), (1200, 1450)], 0.05, 0.04, bright=0.15), 0.55),
        "yay":      (lambda: notes([659, 784, 1047, 1319, 1568], 0.06, 0.01, bright=0.15) , 0.55),
        "whistle":  (lambda: seq(tone(900, 1700, 0.15, curve=0.5, bright=0.05), tone(1700, 1300, 0.1, bright=0.05), tone(1300, 2000, 0.18, bright=0.05)), 0.5),
    },
    "curious": {  # look_around / found something: rising questions
        "bip_bwip": (lambda: seq(tone(1000, 1000, 0.05), gap(0.05), tone(800, 1500, 0.12, curve=2.0)), 0.55),
        "ooh":      (lambda: tone(600, 1100, 0.35, curve=0.5, vib=0.03, vib_hz=8, a=0.04), 0.5),
        "whats_that":(lambda: seq(babble(3, 800, 1100, dur=0.05), tone(900, 1600, 0.15, curve=2.5)), 0.5),
        "peek":     (lambda: notes([(900, 1000), (1100, 1300)], 0.05, 0.08, bright=0.1), 0.45),
    },
    "sigh": {  # bored: slow, low, breathy
        "sigh":     (lambda: mix(tone(520, 260, 0.9, vib=0.015, vib_hz=5, curve=1.4, bright=0.15, a=0.08, r=0.3), noise(0.9, 0.25, 0), 0.4), 0.4),
        "meh":      (lambda: seq(tone(450, 430, 0.18, bright=0.1), gap(0.05), tone(420, 330, 0.3, bright=0.1, r=0.15)), 0.4),
        "deflate":  (lambda: tone(700, 200, 0.8, curve=0.6, fm=0.3, fm_ratio=0.5, a=0.02, r=0.3), 0.4),
        "hum":      (lambda: notes([392, 349, 330], 0.25, 0.04, bright=0.1, vib=0.01, vib_hz=5, a=0.05, r=0.1), 0.35),
    },
    "stretch": {  # bored stretch / yawn: up then down, slow
        "yawn":     (lambda: seq(tone(300, 700, 0.5, curve=0.5, a=0.05), tone(700, 280, 0.7, vib=0.02, vib_hz=6, r=0.25)), 0.45),
        "creak":    (lambda: seq(tone(200, 260, 0.35, fm=2.0, fm_ratio=0.1, bright=0.4), gap(0.05), tone(260, 180, 0.3, fm=2.0, fm_ratio=0.1, bright=0.4)), 0.4),
        "big_stretch":(lambda: seq(tone(350, 900, 0.7, curve=0.4, vib=0.02, vib_hz=5, a=0.1), tone(900, 850, 0.25), tone(850, 300, 0.5, r=0.2)), 0.45),
        "pop":      (lambda: seq(tone(400, 600, 0.4, a=0.05), gap(0.08), tone(1400, 900, 0.04, r=0.02), gap(0.05), tone(1300, 800, 0.04, r=0.02)), 0.45),
    },
    "sleep": {  # back to idle: descending, softening
        "goodnight":(lambda: seq(tone(1000, 950, 0.1), gap(0.04), tone(800, 760, 0.1), gap(0.04), tone(600, 420, 0.3, r=0.2)), 0.45),
        "powerdown":(lambda: tone(1200, 150, 0.8, curve=0.5, bright=0.2, r=0.3), 0.4),
        "snooze":   (lambda: seq(tone(330, 300, 0.4, vib=0.03, vib_hz=3, a=0.1, r=0.15), gap(0.15), tone(300, 280, 0.5, vib=0.03, vib_hz=3, a=0.1, r=0.25)), 0.35),
        "lullaby":  (lambda: notes([784, 659, 523, 392], 0.15, 0.04, bright=0.05, a=0.03, r=0.08), 0.4),
    },
}

if __name__ == "__main__":
    count = 0
    for v, (P, T) in enumerate(VARIANTS, 1):
        for emotion, sounds in SOUNDS.items():
            for i, (name, (build, vol)) in enumerate(sounds.items()):
                rng = np.random.default_rng(100 + i)  # same random chatter across a sound's 3 variations
                save(emotion, name, build(), vol, v)
                count += 1
    print(f"wrote {count} files to {OUT}")
