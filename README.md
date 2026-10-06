# LeLamp ROS 2 workspace (Jazzy)

| Package | What it does |
|---|---|
| `lelamp_camera` | `camera_node` publishes the webcam on `/camera/image_raw`. Falls back to a test pattern if no camera opens. |
| `lelamp_sim` | Starter-pack lamp URDF + mesh, `joint_sim` (moves joints toward `/lelamp/joint_commands` within URDF limits, publishes `/joint_states`), `robot_state_publisher`, RViz config. |
| `lelamp_logic` | `logic_node` stub: subscribes to the camera, publishes an idle motion to `/lelamp/joint_commands`. |

## Build and run

```bash
source /opt/ros/jazzy/setup.bash
cd ~/lelamp_ws && colcon build --symlink-install
source install/setup.bash
ros2 launch ~/lelamp_ws/launch/lelamp.launch.py            # rviz:=false device:=/dev/video0
```

## Webcam on WSL2

WSL2 can't see the laptop camera until it's attached with usbipd. From an
**admin** PowerShell on Windows (BUSID from `usbipd list`, e.g. `2-4` for "HD Camera"):

```powershell
usbipd bind --busid 2-4          # one time
usbipd attach --wsl --busid 2-4  # after each reboot / replug
```

While attached, Windows apps can't use the camera. `usbipd detach --busid 2-4` gives it back.
Inside WSL check `ls /dev/video*`; if nothing shows up, run `sudo modprobe uvcvideo`.

## Object memory (YOLO-World, CPU only)

`object_node` runs YOLO-World (`lelamp_logic/detector.py`) on the newest head-camera
frame whenever `/lelamp/detect_objects` (`lelamp_interfaces/srv/DetectObjects`) is
called, and returns each object's name, confidence and position in `base_link`.
While looking around, `logic_node` calls it at each stop and stores the results
(`lelamp_logic/object_memory.py`, in memory only, so each run starts fresh). The boxes
show in RViz's "Object detection" panel. Try it, or make the lamp look back:

```bash
ros2 service call /lelamp/detect_objects lelamp_interfaces/srv/DetectObjects
ros2 topic pub --once /lelamp/look_at_object std_msgs/msg/String "{data: clock}"
```

One-time install, in the user site so ROS keeps the system numpy 1.26 and OpenCV:

```bash
echo "numpy==1.26.4" > /tmp/c.txt
PIP="pip install --user --break-system-packages -c /tmp/c.txt"
$PIP torch torchvision --index-url https://download.pytorch.org/whl/cpu
$PIP --no-deps ultralytics ultralytics-thop "git+https://github.com/ultralytics/CLIP.git"
$PIP pyyaml pillow requests scipy tqdm psutil polars matplotlib ftfy regex
yolo settings weights_dir=$HOME/.lelamp/models/weights sync=False
```

The model (`~/.lelamp/models/yolov8s-worldv2.pt`) and its CLIP text encoder
download on first run.
