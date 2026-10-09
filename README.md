# LeLamp ROS 2 workspace (Jazzy)

A simulated LeLamp in MuJoCo with face, object and voice control. Runs on the CPU only.

## Setup

In Windows PowerShell:

```powershell
wsl --install Ubuntu-24.04
```

Then inside Ubuntu:

```bash
git clone <this-repo-url> ~/lelamp_ws
cd ~/lelamp_ws
./setup.sh
```

## Run

In a new terminal:

```bash
ros2 launch ~/lelamp_ws/launch/lelamp.launch.py
```

Say "friend" to wake the lamp, then a command like "turn the light blue".

Try asking it to look around, and ask about the items in the room

## Webcam (optional)

Without a webcam the camera shows a test pattern. To use the laptop camera, install
[usbipd-win](https://github.com/dorssel/usbipd-win), find the camera's BUSID with
`usbipd list`, then in PowerShell:

```powershell
usbipd bind --busid {bus_id}                      # once, as admin
usbipd attach --wsl --busid {bus_id} --auto-attach
```
