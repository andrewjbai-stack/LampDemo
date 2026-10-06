"""Open-vocabulary object detection with YOLO-World, on the CPU only.

Plain Python (no ROS). Give it the things to look for once; each call to
detect() then returns what it found in one image:

    det = Detector('~/.lelamp/models/yolov8s-worldv2.pt', LABELS)
    for name, conf, box in det.detect(bgr_image):   # box = (x1, y1, x2, y2) pixels
        ...

LABELS maps the text prompt YOLO-World searches for to the name the lamp
remembers it by, so several prompts can feed one name ('framed picture' finds
posters better than 'poster' does).

pixel_ray() and locate() turn a box into a point in base_link, given the
camera pose (from tf):

    ray = pixel_ray(u, v, width, height, fovy_deg)      # in camera_link
    point = locate(origin, R @ ray, R @ bottom_ray)

One image gives a direction, not a distance. locate() gets the distance from
where the bottom of the box meets the desk or floor, which is right for things
standing on them; anything else (wall clock, poster) goes DEFAULT_DIST along
the ray. Looking back at it is accurate either way, because the head is close
to where it was when it saw the object.

Needs torch (CPU wheel) and ultralytics in the user site; see README.
"""
import math
import os

import numpy as np

# prompt: remembered name
LABELS = {
    'clock': 'clock',
    'bookshelf': 'bookshelf',
    'book': 'book',
    'trophy': 'trophy',
    'gold trophy cup': 'trophy',
    'poster': 'poster',
    'framed picture': 'poster',
    'potted plant': 'potted plant',
    'paper': 'paper',
    'sheet of paper': 'paper',
    'armchair': 'armchair',
    'television': 'tv',
    'desk': 'desk',
}


class Detector:
    def __init__(self, model_path, labels=LABELS, conf=0.15, imgsz=640, threads=4):
        import torch
        from ultralytics import YOLOWorld

        torch.set_num_threads(threads)   # leave cores for the sim and the 30 Hz loop
        self.prompts = list(labels)
        self.names = [labels[p] for p in self.prompts]
        self.conf, self.imgsz = conf, imgsz
        self.model = YOLOWorld(os.path.expanduser(model_path))
        self.model.set_classes(self.prompts)   # encodes the prompts once (CLIP)

    def detect(self, bgr):
        """[(name, conf, (x1, y1, x2, y2))] for one BGR image, best first. When
        several prompts for the same name hit the same object, keeps the best."""
        r = self.model.predict(bgr, device='cpu', conf=self.conf, imgsz=self.imgsz,
                               verbose=False)[0]
        found = []
        for cls, conf, box in sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                         r.boxes.xyxy.tolist()), key=lambda d: -d[1]):
            name = self.names[int(cls)]
            if any(n == name and _iou(box, b) > 0.5 for n, _, b in found):
                continue
            found.append((name, conf, tuple(box)))
        return found


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# Room surfaces in base_link (the sim's desk and floor; see lelamp_sim).
DESK_Z = 0.0
DESK_X = (-0.75, 0.25)
DESK_Y = (-0.7, 0.7)
FLOOR_Z = -0.75
DEFAULT_DIST = 2.5   # metres along the ray when it meets neither
MAX_DIST = 4.0


def pixel_ray(u, v, width, height, fovy_deg):
    """Unit ray through pixel (u, v) in camera_link (looks along +x, image
    right = -y, image up = +z), for a pinhole camera with vertical FOV fovy."""
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    ray = np.array([1.0, -(u - width / 2.0) / f, -(v - height / 2.0) / f])
    return ray / np.linalg.norm(ray)


def _surface_hit(origin, ray):
    """Distance along ray to the desk top or the floor, or None."""
    if ray[2] >= -1e-3:
        return None
    t = (DESK_Z - origin[2]) / ray[2]
    p = origin + t * ray
    if t > 0 and DESK_X[0] <= p[0] <= DESK_X[1] and DESK_Y[0] <= p[1] <= DESK_Y[1]:
        return t
    t = (FLOOR_Z - origin[2]) / ray[2]
    return t if t > 0 else None


def locate(origin, centre_ray, bottom_ray):
    """Point in base_link for an object seen along centre_ray, whose box bottom
    is along bottom_ray (both unit rays from the camera at origin)."""
    origin = np.asarray(origin, float)
    t = _surface_hit(origin, bottom_ray)
    if t is None or t > MAX_DIST:
        dist = DEFAULT_DIST
    else:
        # same horizontal distance as the foot of the box, along the centre ray
        flat_b = math.hypot(bottom_ray[0], bottom_ray[1]) * t
        flat_c = max(math.hypot(centre_ray[0], centre_ray[1]), 1e-3)
        dist = min(max(flat_b / flat_c, 0.15), MAX_DIST)
    return origin + dist * np.asarray(centre_ray, float)
