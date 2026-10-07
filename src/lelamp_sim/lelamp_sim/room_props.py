"""Furniture and objects around the lamp's room, for the head camera to look at.

add_props(spec) adds them before compiling; paint_props(model) then draws the
clock-face picture into its texture (call it once after compile,
before the viewer or a renderer is created, so both upload the painted pixels).

Room frame (same as mujoco_sim_node): the lamp base is at the origin facing -x,
the desk top is at z = 0, the floor at FLOOR_Z, the walls at x, y = +-4. The
lamp's left is -y. Every prop is its own named body, so its position can be
looked up by name (model.body('clock')) to check what the lamp is looking at.
"""
import math

import mujoco
import numpy as np

FLOOR_Z = -0.75
WALL = 3.975           # inner face of every wall, at x or y = +-WALL

# Spread around the lamp (bearing from the lamp; it faces -x and the TV is
# straight ahead, so nothing goes on the wall behind the TV).
# On the floor, (x, y); furniture is turned to face the lamp.
ARMCHAIR_POS = (-1.3, -2.1)    # front left
BOOKSHELF_POS = (-0.4, 2.4)    # right (trophy on top)
PLANT_POS = (1.6, 1.7)         # behind, to the right
# on the desk
PAPER_POS = (-0.42, 0.25)
# On a wall: (x, y, z of the centre, yaw); at yaw 0 the picture faces +x.
CLOCK_POS = (0.6, -WALL, 1.1, 90)     # left wall
CLOCK_R = 0.3
CLOCK_TEX = 256

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
ELL = mujoco.mjtGeom.mjGEOM_ELLIPSOID
CAP = mujoco.mjtGeom.mjGEOM_CAPSULE
MESH = mujoco.mjtGeom.mjGEOM_MESH


def _quat_z(deg):
    a = math.radians(deg) / 2
    return [math.cos(a), 0, 0, math.sin(a)]


def _mat_quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, float).ravel())
    return list(q)


def _rot(axis, deg):
    """3x3 rotation about 'x', 'y' or 'z'."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            'y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def _facing_lamp(x, y):
    """Yaw (deg) that turns a body's +x toward the lamp at the origin."""
    return math.degrees(math.atan2(-y, -x))


def _add(body, type, size, pos, **kw):
    return body.add_geom(type=type, size=list(size), pos=list(pos),
                         contype=0, conaffinity=0, **kw)


def _materials(spec):
    def flat(name, rgba, specular=0.1, shininess=0.3, reflectance=0.0):
        spec.add_material(name=name, rgba=rgba, specular=specular, shininess=shininess,
                          reflectance=reflectance)
    flat('shelf_wood', [0.36, 0.24, 0.15, 1], 0.2)
    flat('fabric', [0.16, 0.36, 0.38, 1], 0.02, 0.05)
    flat('fabric_dark', [0.12, 0.28, 0.3, 1], 0.02, 0.05)
    flat('chair_leg', [0.2, 0.13, 0.08, 1], 0.2)
    flat('gold', [0.95, 0.72, 0.2, 1], 0.9, 0.9, 0.1)
    flat('trophy_base', [0.1, 0.08, 0.07, 1], 0.4, 0.6)
    flat('terracotta', [0.72, 0.36, 0.22, 1], 0.1)
    flat('soil', [0.2, 0.13, 0.08, 1], 0.0)
    flat('paper', [0.96, 0.96, 0.93, 1], 0.05)
    flat('ink', [0.25, 0.25, 0.28, 1], 0.0)
    flat('clock_rim', [0.12, 0.12, 0.13, 1], 0.5, 0.6)
    spec.add_texture(name='clock_face', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     width=CLOCK_TEX, height=CLOCK_TEX, rgb1=[1, 1, 1])
    spec.add_material(name='clock_face', textures=['', 'clock_face'], specular=0.05,
                      emission=0.15)


def _bookshelf(spec):
    x, y = BOOKSHELF_POS
    shelf = spec.worldbody.add_body(name='bookshelf', pos=[x, y, FLOOR_Z],
                                    quat=_quat_z(_facing_lamp(x, y)))
    d, w, h, t = 0.15, 0.45, 1.8, 0.015   # half depth, half width, height, half board
    wood = dict(material='shelf_wood')
    for side in (-1, 1):
        _add(shelf, BOX, [d, t, h / 2], [0, side * (w - t), h / 2], **wood)
    _add(shelf, BOX, [0.006, w, h / 2], [-d + 0.006, 0, h / 2], **wood)   # back
    _add(shelf, BOX, [d, w, t], [0, 0, h - t], **wood)                     # top
    levels = [0.08, 0.5, 0.92, 1.34]
    for z in levels:
        _add(shelf, BOX, [d, w - 2 * t, t], [0, 0, z], **wood)
    _add(shelf, BOX, [0.01, w - 2 * t, levels[0] / 2], [d - 0.01, 0, levels[0] / 2], **wood)

    # Books: upright rows with a few gaps, plus a lying stack on the bottom shelf.
    rng = np.random.default_rng(7)
    colors = [[0.62, 0.12, 0.1], [0.12, 0.25, 0.5], [0.15, 0.4, 0.2], [0.85, 0.7, 0.25],
              [0.3, 0.15, 0.35], [0.85, 0.85, 0.8], [0.1, 0.1, 0.12], [0.75, 0.4, 0.15]]
    n = 0
    for z, fill in zip(levels[1:], (0.85, 0.6, 0.75)):
        yy = -w + 2 * t + 0.01
        end = -w + 2 * t + fill * 2 * (w - 2 * t)
        while yy < end:
            half_w = rng.uniform(0.012, 0.024)
            half_h = rng.uniform(0.1, 0.17)
            half_d = rng.uniform(0.08, 0.11)
            _add(shelf, BOX, [half_d, half_w, half_h],
                 [d - half_d - 0.01, yy + half_w, z + t + half_h],
                 rgba=colors[rng.integers(len(colors))] + [1], name=f'book_{n}')
            n += 1
            yy += 2 * half_w + (0.03 if rng.random() < 0.12 else 0.002)
    zz = levels[0] + t
    for k in range(4):
        half_h = rng.uniform(0.015, 0.025)
        _add(shelf, BOX, [0.11, 0.16 - 0.01 * k, half_h], [0.01, 0.15, zz + half_h],
             rgba=colors[(k * 3) % len(colors)] + [1], name=f'book_{n}')
        n += 1
        zz += 2 * half_h

    # Trophy standing on top of the shelf.
    trophy = shelf.add_body(name='trophy', pos=[0.02, -0.22, h])
    _add(trophy, BOX, [0.045, 0.045, 0.02], [0, 0, 0.02], material='trophy_base')
    _add(trophy, BOX, [0.035, 0.035, 0.015], [0, 0, 0.055], material='trophy_base')
    _add(trophy, CYL, [0.012, 0.04], [0, 0, 0.11], material='gold')
    _add(trophy, CYL, [0.03, 0.006], [0, 0, 0.152], material='gold')
    _add(trophy, ELL, [0.055, 0.055, 0.06], [0, 0, 0.21], material='gold')
    _add(trophy, CYL, [0.058, 0.004], [0, 0, 0.27], material='gold')
    for side in (-1, 1):   # handles
        _add(trophy, CAP, [0.008, 0.03], [0, side * 0.07, 0.22], material='gold',
             quat=_mat_quat(_rot('x', side * 25)))


def _armchair(spec):
    x, y = ARMCHAIR_POS
    chair = spec.worldbody.add_body(name='armchair', pos=[x, y, FLOOR_Z],
                                    quat=_quat_z(_facing_lamp(x, y)))
    for lx in (-0.32, 0.32):
        for ly in (-0.36, 0.36):
            _add(chair, BOX, [0.025, 0.025, 0.05], [lx, ly, 0.05], material='chair_leg')
    _add(chair, BOX, [0.38, 0.42, 0.12], [0, 0, 0.22], material='fabric')         # base
    _add(chair, BOX, [0.3, 0.29, 0.06], [0.05, 0, 0.4], material='fabric_dark')   # cushion
    _add(chair, BOX, [0.09, 0.42, 0.36], [-0.3, 0, 0.66], material='fabric',
         quat=_mat_quat(_rot('y', -10)))                                          # back
    for side in (-1, 1):
        _add(chair, BOX, [0.38, 0.09, 0.14], [0, side * 0.33, 0.48], material='fabric')
        _add(chair, CYL, [0.09, 0.38], [0, side * 0.33, 0.62], material='fabric',
             quat=_mat_quat(_rot('y', 90)))                                       # rolled arm


def _plant(spec):
    x, y = PLANT_POS
    plant = spec.worldbody.add_body(name='potted_plant', pos=[x, y, FLOOR_Z])
    _add(plant, CYL, [0.15, 0.16], [0, 0, 0.16], material='terracotta')
    _add(plant, CYL, [0.168, 0.025], [0, 0, 0.32], material='terracotta')
    _add(plant, CYL, [0.145, 0.004], [0, 0, 0.342], material='soil')
    rng = np.random.default_rng(3)
    base = np.array([0, 0, 0.34])
    for k in range(14):
        tilt = 15 + 40 * (k % 3) / 2 + rng.uniform(-5, 5)
        length = rng.uniform(0.2, 0.3) * (1.25 if k % 3 == 0 else 1.0)
        R = _rot('z', k * 360 / 14 + rng.uniform(-10, 10)) @ _rot('y', tilt)
        centre = base + R @ np.array([0, 0, length])
        green = [0.12 + rng.uniform(0, 0.1), 0.38 + rng.uniform(0, 0.15), 0.12, 1]
        _add(plant, ELL, [0.05, 0.012, length], centre, rgba=green,
             quat=_mat_quat(R))


def _paper(spec):
    x, y = PAPER_POS
    paper = spec.worldbody.add_body(name='paper', pos=[x, y, 0.0], quat=_quat_z(-18))
    _add(paper, BOX, [0.148, 0.105, 0.0006], [0, 0, 0.0006], material='paper')
    # a heading and some lines of text
    _add(paper, BOX, [0.004, 0.05, 0.0002], [-0.115, 0.03, 0.0014], material='ink')
    for k in range(10):
        _add(paper, BOX, [0.0016, 0.08 - (0.03 if k % 4 == 3 else 0), 0.0002],
             [-0.085 + 0.019 * k, 0.0 + (0.03 if k % 4 == 3 else 0), 0.0014],
             material='ink')


def _clock(spec):
    n = 48
    verts, uv, faces = [0, 0, 0], [0.5, 0.5], []
    for k in range(n):
        a = 2 * math.pi * k / n
        verts += [0, CLOCK_R * math.cos(a), CLOCK_R * math.sin(a)]
        uv += [0.5 + 0.5 * math.cos(a), 0.5 - 0.5 * math.sin(a)]
        faces += [0, 1 + k, 1 + (k + 1) % n]
    spec.add_mesh(name='clock_face', inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
                  uservert=verts, usertexcoord=uv, userface=faces)
    x, y, z, yaw = CLOCK_POS
    clock = spec.worldbody.add_body(name='clock', pos=[x, y, z], quat=_quat_z(yaw))
    _add(clock, CYL, [CLOCK_R + 0.02, 0.02], [0.02, 0, 0], material='clock_rim',
         quat=_mat_quat(_rot('y', 90)))
    _add(clock, MESH, [0, 0, 0], [0.041, 0, 0], meshname='clock_face', material='clock_face')


def add_props(spec):
    """Bookshelf with books and a trophy, armchair, potted plant, paper on the
    desk and a clock, spread around the lamp (none behind the TV)."""
    _materials(spec)
    _bookshelf(spec)
    _armchair(spec)
    _plant(spec)
    _paper(spec)
    _clock(spec)


# --- pictures, drawn with numpy (no image files) ---

def _clock_image(s, hour=10, minute=10, second=37):
    yy, xx = (np.mgrid[0:s, 0:s] + 0.5) / s * 2 - 1     # -1..1, y down
    r, ang = np.hypot(xx, yy), np.arctan2(xx, -yy)       # ang 0 at 12, clockwise
    img = np.full((s, s, 3), 0.97)
    img[r > 0.97] = [0.12, 0.12, 0.13]
    for k in range(60):
        a = k * np.pi / 30
        d = np.abs(np.angle(np.exp(1j * (ang - a))))
        big = k % 5 == 0
        tick = (d * r < (0.03 if big else 0.012)) & (r > (0.75 if big else 0.86)) & (r < 0.92)
        img[tick] = [0.1, 0.1, 0.1]

    def hand(a, length, width, color):
        ux, uy = np.sin(a), -np.cos(a)
        along = xx * ux + yy * uy
        across = np.abs(xx * uy - yy * ux)
        img[(along > -0.1) & (along < length) & (across < width)] = color

    hand(2 * np.pi * ((hour % 12) + minute / 60) / 12, 0.5, 0.035, [0.08, 0.08, 0.08])
    hand(2 * np.pi * minute / 60, 0.75, 0.025, [0.08, 0.08, 0.08])
    hand(2 * np.pi * second / 60, 0.8, 0.008, [0.8, 0.1, 0.1])
    img[r < 0.05] = [0.8, 0.1, 0.1]
    return (img * 255).astype(np.uint8)


def paint_props(model):
    img = _clock_image(CLOCK_TEX)
    adr = model.tex_adr[model.texture('clock_face').id]
    model.tex_data[adr:adr + img.size] = img.ravel()
