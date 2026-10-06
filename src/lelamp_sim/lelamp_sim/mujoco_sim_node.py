"""Simulated lamp body in MuJoCo, with a real spotlight in the shade.

Loads the lamp URDF into MuJoCo, sets it on a desk in a dim room, and shows it
in MuJoCo's viewer (rendered on the CPU). Joints are moved kinematically, like
joint_sim: each joint heads toward its target no faster than its URDF velocity
limit and is clamped to its URDF position limits (no physics, so no sag).

Inputs:  /lelamp/joint_commands  (sensor_msgs/JointState, target positions)
         /lelamp/light           (std_msgs/ColorRGBA)
             r, g, b = color, a = brightness (0 off .. 1 full)
         /camera/image_raw       (sensor_msgs/Image, param tv_topic)
             shown on the TV on the desk, updated tv_fps times a second
Outputs: /joint_states           (sensor_msgs/JointState)
         /lelamp/sim_camera/image_raw  (sensor_msgs/Image, rgb8)
             what the head camera (URDF camera_link) sees, rendered offscreen
             on the CPU at camera_width x camera_height, camera_fps

Requires the mujoco Python package (pip install --user mujoco).
"""
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import ColorRGBA

from lelamp_sim.joint_sim_node import parse_revolute_joints

BULB_RGB = (1.0, 0.95, 0.76)  # URDF "fixture_light" material: the bulb visuals
LIGHT_POS = (0.113, 0.0, 0.0)  # light_emitter_link inside lamp_head_link
# camera_link looks along its +x with +z up; a MuJoCo camera looks along its -z
# with +y up, so image right = -y and image up = +z of the link.
CAMERA_XYAXES = [0, -1, 0, 0, 0, 1]
CAMERA_FOVY = 60.0  # degrees, a typical small webcam
CAMERA_POS = (0.010, 0.0, 0.0)  # lens on the front face of the camera_link housing

# Viewer camera: off to the lamp's right side and a little behind it, so the
# lamp and the TV screen 2 m in front of it are both in view. With the
# lock_view parameter on (default) it is held here every frame, so mouse drags
# and camera switching in the viewer don't move it. Edit these to re-aim it.
VIEW = {'lookat': (-1.0, 0.0, 0.2), 'distance': 2.6, 'azimuth': 120.0, 'elevation': -30.0}


def apply_view(cam):
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = VIEW['lookat']
    cam.distance = VIEW['distance']
    cam.azimuth = VIEW['azimuth']
    cam.elevation = VIEW['elevation']


# TV showing the webcam feed, standing on the floor 2 m in front of the lamp
# (the lamp faces -x) and facing it, roughly where you'd sit. Screen is
# TV_SIZE (w, h) metres, centred at TV_POS, turned TV_YAW degrees; at yaw 0 the
# screen faces +x. Size and texture match the webcam's 640x480 (4:3) image.
TV_POS = (-2.0, 0.0, 0.35)
TV_YAW = 0.0
TV_SIZE = (1.2, 0.9)
TV_TEX = (640, 480)  # camera_node's default width x height
FLOOR_Z = -0.75


def add_tv(spec):
    """A screen (textured quad), bezel and stand; texture 'tv' is the picture."""
    w, h = TV_SIZE[0] / 2, TV_SIZE[1] / 2
    spec.add_texture(name='tv', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     width=TV_TEX[0], height=TV_TEX[1], rgb1=[0.02, 0.02, 0.03])
    spec.add_material(name='tv', textures=['', 'tv'], emission=0.8, specular=0.1)
    # Quad in the y-z plane facing +x. Seen from the front, +y is right; texture
    # row 0 (image top) is v = 0.
    spec.add_mesh(name='tv_screen', inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
                  uservert=[0, -w, -h, 0, w, -h, 0, w, h, 0, -w, h],
                  usertexcoord=[0, 1, 1, 1, 1, 0, 0, 0],
                  userface=[0, 1, 2, 0, 2, 3])

    yaw = np.radians(TV_YAW)
    tv = spec.worldbody.add_body(name='tv', pos=list(TV_POS),
                                 quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    dark = [0.05, 0.05, 0.06, 1]
    tv.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='tv_screen', material='tv',
                pos=[0.002, 0, 0], contype=0, conaffinity=0)
    tv.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, w + 0.015, h + 0.015],
                pos=[-0.011, 0, 0], rgba=dark, contype=0, conaffinity=0)
    floor = FLOOR_Z - TV_POS[2]  # floor height in the TV body's frame
    pole = (-h - floor) / 2      # half-length of the stand pole, screen bottom to floor
    tv.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.03, pole],
                pos=[-0.04, 0, floor + pole], rgba=dark, contype=0, conaffinity=0)
    tv.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.3, 0.01],
                pos=[-0.04, 0, floor + 0.01], rgba=dark, contype=0, conaffinity=0)


def _load_ascii_stl(path):
    """MuJoCo only reads binary STL; the lamp shade is ASCII."""
    with open(path) as f:
        verts = [[float(v) for v in line.split()[1:4]]
                 for line in f if line.strip().startswith('vertex')]
    return np.array(verts).ravel(), np.arange(len(verts))


def build_model(urdf, meshdir):
    """Lamp from the URDF plus desk, room, and the shade's spotlight."""
    compiler = (f'<mujoco><compiler meshdir="{meshdir}" strippath="true" '
                'discardvisual="false" fusestatic="false" balanceinertia="true"/></mujoco>')
    spec = mujoco.MjSpec.from_string(urdf.replace('</robot>', compiler + '</robot>', 1))

    for mesh in spec.meshes:
        path = os.path.join(meshdir, os.path.basename(mesh.file))
        with open(path, 'rb') as f:
            ascii_stl = f.read(5) == b'solid'
        if ascii_stl:
            mesh.uservert, mesh.userface = _load_ascii_stl(path)
            mesh.file = ''

    # Hide URDF collision shapes and the light_emitter_link marker sphere (it
    # would sit around the spotlight and shadow it); tag the light face in the
    # shade as the bulb so it can glow.
    spec.add_material(name='bulb', emission=0.9, specular=0.0)
    bulbs = []
    for geom in spec.geoms:
        if geom.group == 0 or geom.parent.name == 'light_emitter_link':
            geom.rgba = [0, 0, 0, 0]
            geom.contype = geom.conaffinity = 0
        elif np.allclose(geom.rgba[:3], BULB_RGB, atol=0.01):
            geom.name = f'bulb_{len(bulbs)}'
            geom.material = 'bulb'
            bulbs.append(geom.name)

    # Dim room so the lamp's own light is what you notice. Shadow map and
    # anti-aliasing are kept modest because everything renders on the CPU.
    spec.visual.headlight.ambient = [0.15, 0.15, 0.17]
    spec.visual.headlight.diffuse = [0.15, 0.15, 0.15]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.quality.shadowsize = 2048
    spec.visual.quality.offsamples = 0
    spec.visual.map.znear = 0.01

    spec.add_texture(name='wood', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, width=256, height=256,
                     rgb1=[0.55, 0.38, 0.22], rgb2=[0.45, 0.3, 0.18],
                     mark=mujoco.mjtMark.mjMARK_RANDOM, random=0.2, markrgb=[0.5, 0.35, 0.2])
    spec.add_texture(name='floor', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                     rgb1=[0.25, 0.24, 0.23], rgb2=[0.2, 0.19, 0.18])
    spec.add_material(name='wood', textures=['', 'wood'], specular=0.3, shininess=0.4)
    spec.add_material(name='floor', textures=['', 'floor'], texrepeat=[6, 6], reflectance=0.05)
    spec.add_material(name='wall', rgba=[0.75, 0.74, 0.72, 1], specular=0.05)

    world = spec.worldbody
    box, plane = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_PLANE
    world.add_geom(type=plane, size=[4, 4, 0.1], pos=[0, 0, -0.75], material='floor')
    world.add_geom(type=box, size=[0.5, 0.7, 0.02], pos=[-0.25, 0, -0.02], material='wood')  # desk top at z=0
    for x, y in [(0.2, 0.65), (0.2, -0.65), (-0.7, 0.65), (-0.7, -0.65)]:
        world.add_geom(type=box, size=[0.025, 0.025, 0.355], pos=[x, y, -0.395],
                       rgba=[0.15, 0.15, 0.15, 1])
        
    world.add_geom(type=box, size=[0.025, 4, 1.25], pos=[4.0, 0, 0.5], material='wall')
    world.add_geom(type=box, size=[0.025, 4, 1.25], pos=[-4.0, 0, 0.5], material='wall')
    world.add_geom(type=box, size=[4, 0.025, 1.25], pos=[0, -4.0, 0.5], material='wall')
    world.add_geom(type=box, size=[4, 0.025, 1.25], pos=[0, 4.0, 0.5], material='wall')

    world.add_light(name='room', type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    pos=[0, 0, 2.5], dir=[0.2, 0.1, -1], castshadow=0,
                    diffuse=[0.15, 0.15, 0.17], specular=[0, 0, 0])

    spec.body('lamp_head_link').add_light(
        name='lamp', type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=list(LIGHT_POS), dir=[1, 0, 0],
        cutoff=55, exponent=8, attenuation=[0.6, 0.3, 0.3], castshadow=1,
        diffuse=[1, 0.9, 0.7], specular=[0.3, 0.3, 0.3])

    spec.body('camera_link').add_camera(name='head_camera', pos=list(CAMERA_POS),
                                        xyaxes=CAMERA_XYAXES, fovy=CAMERA_FOVY)

    add_tv(spec)

    model = spec.compile()
    return model, [model.geom(name).id for name in bulbs], model.light('lamp').id


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.declare_parameter('robot_description', '')
        self.declare_parameter('rate_hz', 60.0)
        self.declare_parameter('lock_view', False)  # pin the viewer camera to VIEW
        # Kept small and slow: every frame is rendered on the CPU.
        self.declare_parameter('camera_width', 320)
        self.declare_parameter('camera_height', 240)
        self.declare_parameter('camera_fps', 10.0)

        urdf = self.get_parameter('robot_description').value
        if not urdf:
            raise RuntimeError('robot_description parameter is empty')
        meshdir = os.path.join(get_package_share_directory('lelamp_sim'), 'meshes')
        self.model, self.bulb_geoms, self.lamp_light = build_model(urdf, meshdir)
        self.data = mujoco.MjData(self.model)
        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)

        self.limits = parse_revolute_joints(urdf)
        self.names = [n for n in self.limits if mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
        self.qadr = {n: self.model.joint(n).qposadr[0] for n in self.names}
        self.pos = {n: 0.0 for n in self.names}
        self.target = dict(self.pos)
        self.set_light(1.0, 0.85, 0.6, 0.8)  # warm and on until logic says otherwise

        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_subscription(JointState, 'lelamp/joint_commands', self._on_cmd, 10)
        self.create_subscription(ColorRGBA, 'lelamp/light', self._on_light, 10)

        self.cam_pub = self.create_publisher(Image, 'lelamp/sim_camera/image_raw', 1)
        self.renderer = mujoco.Renderer(self.model,
                                        int(self.get_parameter('camera_height').value),
                                        int(self.get_parameter('camera_width').value))
        self.cam_period = 1.0 / float(self.get_parameter('camera_fps').value)
        self.next_cam = 0.0

        # Desk TV: the webcam feed, copied into the 'tv' texture a few times a second.
        self.declare_parameter('tv_topic', 'camera/image_raw')
        self.declare_parameter('tv_fps', 15.0)  # camera_node's default frame rate
        self.tv_tex = self.model.texture('tv').id
        adr = self.model.tex_adr[self.tv_tex]
        self.tv_pixels = self.model.tex_data[adr:adr + TV_TEX[0] * TV_TEX[1] * 3]
        self.tv_period = 1.0 / float(self.get_parameter('tv_fps').value)
        self.next_tv = 0.0
        self.tv_frame = None
        self.tv_renderer_stale = False  # head camera renderer has its own GPU copy
        self.create_subscription(Image, self.get_parameter('tv_topic').value,
                                 self._on_tv_image, qos_profile_sensor_data)
        self.get_logger().info(f'MuJoCo lamp ready, joints: {", ".join(self.names)}')

    def _on_cmd(self, msg):
        for name, p in zip(msg.name, msg.position):
            if name in self.target:
                lim = self.limits[name]
                self.target[name] = min(max(p, lim['lower']), lim['upper'])

    def _on_tv_image(self, msg):
        self.tv_frame = msg

    def update_tv(self):
        """Copy the latest webcam frame into the TV texture, at most tv_fps times
        a second. Returns True when the texture changed (caller re-uploads it)."""
        now = time.monotonic()
        if self.tv_frame is None or now < self.next_tv:
            return False
        self.next_tv = now + self.tv_period
        msg, self.tv_frame = self.tv_frame, None

        channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}.get(msg.encoding)
        if channels is None:
            self.get_logger().warn(f'TV: unsupported encoding {msg.encoding}',
                                   throttle_duration_sec=10.0)
            return False
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        elif channels == 1:
            img = np.repeat(img, 3, axis=2)
        # Nearest-neighbour resize to the texture size; cheap enough for the CPU.
        rows = np.arange(TV_TEX[1]) * msg.height // TV_TEX[1]
        cols = np.arange(TV_TEX[0]) * msg.width // TV_TEX[0]
        self.tv_pixels[:] = img[rows][:, cols].ravel()
        self.tv_renderer_stale = True
        return True

    def _on_light(self, msg):
        self.set_light(msg.r, msg.g, msg.b, msg.a)

    def set_light(self, r, g, b, level):
        level = min(max(level, 0.0), 1.0)
        rgb = np.array([r, g, b])
        self.model.light_diffuse[self.lamp_light] = rgb * level
        self.model.light_specular[self.lamp_light] = 0.3 * rgb * level
        glow = rgb * (0.2 + 0.8 * level)  # an "off" bulb still shows its color faintly
        for gid in self.bulb_geoms:
            self.model.geom_rgba[gid] = [*glow, 1.0]
            self.model.mat_emission[self.model.geom_matid[gid]] = 0.9 * level

    def step(self):
        """Advance joints one tick, update the scene, and publish /joint_states."""
        for n in self.names:
            max_step = self.limits[n]['velocity'] * self.dt
            err = self.target[n] - self.pos[n]
            self.pos[n] += max(-max_step, min(max_step, err))
            self.data.qpos[self.qadr[n]] = self.pos[n]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)  # moves the spotlight with the head

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = [self.pos[n] for n in self.names]
        self.pub.publish(msg)

    def render_camera(self):
        """Publish the head camera view, at most camera_fps times a second."""
        now = time.monotonic()
        if now < self.next_cam:
            return
        self.next_cam = now + self.cam_period
        if self.tv_renderer_stale:  # so the head camera sees the current TV picture
            self.renderer._gl_context.make_current()
            mujoco.mjr_uploadTexture(self.model, self.renderer._mjr_context, self.tv_tex)
            self.tv_renderer_stale = False
        self.renderer.update_scene(self.data, camera='head_camera')
        rgb = self.renderer.render()

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.height, msg.width = rgb.shape[:2]
        msg.encoding = 'rgb8'
        msg.step = msg.width * 3
        msg.data = rgb.tobytes()
        self.cam_pub.publish(msg)

    def destroy_node(self):
        self.renderer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        with mujoco.viewer.launch_passive(node.model, node.data,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            apply_view(viewer.cam)
            lock_view = bool(node.get_parameter('lock_view').value)
            while viewer.is_running() and rclpy.ok():
                start = time.monotonic()
                rclpy.spin_once(node, timeout_sec=0.0)
                with viewer.lock():
                    if lock_view:
                        apply_view(viewer.cam)
                    node.step()
                    tv_changed = node.update_tv()
                    node.render_camera()
                if tv_changed:
                    viewer.update_texture(node.tv_tex)
                viewer.sync()
                time.sleep(max(0.0, node.dt - (time.monotonic() - start)))
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
