"""Where the lamp has seen things, in base_link, so it can look back at them.

Plain Python (no ROS). logic_node adds what object_node's detect_objects
service returns (points already in base_link):

    memory = ObjectMemory()
    memory.add('clock', (x, y, z), conf)
    memory.find('clock')                                # -> (x, y, z) or None

Lives in memory only, so every run of logic_node starts fresh.

Sightings of the same name less than MERGE_DEG apart (seen from the base) are
one object and get averaged; further apart they are two (two books).
"""
import math
import threading
import time

import numpy as np

MERGE_DEG = 10.0


def _bearing(p):
    return np.asarray(p, float) / max(np.linalg.norm(p), 1e-6)


class ObjectMemory:
    def __init__(self):
        self.objects = {}   # name: [{'xyz', 'conf', 'seen', 'last_seen'}]
        self.lock = threading.RLock()   # safe to share between threads

    def add(self, name, xyz, conf, stamp=None):
        """Record a sighting; merges it into an earlier one of the same object."""
        with self.lock:
            return self._add(name, xyz, conf, stamp)

    def _add(self, name, xyz, conf, stamp):
        xyz = np.asarray(xyz, float)
        stamp = time.time() if stamp is None else stamp
        entries = self.objects.setdefault(name, [])
        cos_merge = math.cos(math.radians(MERGE_DEG))
        for e in entries:
            if _bearing(e['xyz']) @ _bearing(xyz) > cos_merge:
                w = conf / (e['conf'] * e['seen'] + conf)   # confidence-weighted mean
                e['xyz'] = list((1 - w) * np.asarray(e['xyz']) + w * xyz)
                e['conf'] = max(e['conf'], conf)
                e['seen'] += 1
                e['last_seen'] = stamp
                return e
        e = {'xyz': list(xyz), 'conf': conf, 'seen': 1, 'last_seen': stamp}
        entries.append(e)
        return e

    def find(self, name):
        """(x, y, z) of the surest sighting of name (case-insensitive, and
        'the potted plant' / 'plant' both match), or None."""
        key = name.lower().strip()
        with self.lock:
            matches = [n for n in self.objects if n == key] or \
                      [n for n in self.objects if n in key or key in n]
            entries = [dict(e) for n in matches for e in self.objects[n]]
        if not entries:
            return None
        best = max(entries, key=lambda e: e['conf'] * min(e['seen'], 5))
        return tuple(best['xyz'])

    def names(self):
        return sorted(self.objects)

    def summary(self):
        return ', '.join(f'{n} x{len(es)}' for n, es in sorted(self.objects.items()))

    def clear(self):
        with self.lock:
            self.objects = {}
