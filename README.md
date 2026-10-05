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
