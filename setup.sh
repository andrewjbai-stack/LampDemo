#!/usr/bin/env bash
# One-shot LeLamp setup for a fresh WSL Ubuntu 24.04: ROS 2 Jazzy, Python packages
# (CPU only), models in ~/.lelamp/models, and a colcon build of this workspace.
# Safe to re-run; finished steps are skipped or are no-ops.
#
#     ./setup.sh
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$HOME/.lelamp/models"
step() { echo; echo "==> $*"; }

if [ "$(. /etc/os-release && echo "$VERSION_CODENAME")" != "noble" ]; then
    echo "This needs Ubuntu 24.04 (noble); ROS 2 Jazzy doesn't support other versions." >&2
    exit 1
fi

step "ROS 2 Jazzy apt source"
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y software-properties-common curl
    sudo add-apt-repository -y universe
    ver=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
          | grep -F tag_name | awk -F\" '{print $4}')
    curl -L -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.noble_all.deb"
    sudo dpkg -i /tmp/ros2-apt-source.deb
fi

step "System packages (ROS 2 Jazzy desktop, colcon, rosdep, v4l-utils, ...)"
sudo apt update
sudo apt upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt install -y \
    ros-jazzy-desktop ros-dev-tools python3-pip git curl v4l-utils mesa-utils \
    python3-scipy python3-matplotlib python3-psutil python3-pil python3-requests python3-yaml

step "Shell setup (~/.bashrc)"
add_line() { grep -qxF "$1" ~/.bashrc || echo "$1" >> ~/.bashrc; }
add_line 'source /opt/ros/jazzy/setup.bash'
add_line 'export PATH="$HOME/.local/bin:$PATH"'
add_line "[ -f $WS/install/setup.bash ] && source $WS/install/setup.bash"
export PATH="$HOME/.local/bin:$PATH"
# ROS setup scripts reference unset variables; source them without -u.
source /opt/ros/jazzy/setup.bash

step "ROS dependencies (rosdep)"
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
rosdep update
rosdep install --from-paths "$WS/src" --ignore-src -y

step "Python packages (user site, CPU only, numpy pinned to the system 1.26.4)"
# Ultralytics goes in without its dependencies so it can't pull in NumPy 2 or
# opencv-python, which would shadow the system packages ROS's cv_bridge needs.
echo "numpy==1.26.4" > /tmp/lelamp-constraints.txt
PIP="pip install --user --break-system-packages -c /tmp/lelamp-constraints.txt"
$PIP torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
$PIP --no-deps ultralytics==8.4.173 ultralytics-thop==2.2.2 "git+https://github.com/ultralytics/CLIP.git"
$PIP ftfy regex tqdm polars
$PIP mujoco==3.15.0 faster-whisper==1.2.1 onnxruntime==1.30.0
$PIP llama-cpp-python==0.3.36 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python3 -c "import numpy, cv2, torch, ultralytics, mujoco, faster_whisper, llama_cpp
assert numpy.__version__ == '1.26.4', numpy.__version__
print('numpy', numpy.__version__, 'cv2', cv2.__version__, 'torch', torch.__version__)"

step "Models in $MODELS"
mkdir -p "$MODELS/llm" "$MODELS/weights" ~/.config/Ultralytics
# Ultralytics downloads go to ~/.lelamp, and its telemetry stays off.
yolo settings weights_dir="$MODELS/weights" sync=False >/dev/null
LLM=qwen2.5-1.5b-instruct-q4_k_m.gguf
[ -f "$MODELS/llm/$LLM" ] || hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF "$LLM" --local-dir "$MODELS/llm"
# YOLO-World + its CLIP text encoder and whisper base.en; otherwise they download on first launch.
(cd ~ && python3 -c "
import os
from ultralytics import YOLOWorld
YOLOWorld(os.path.expanduser('$MODELS/yolov8s-worldv2.pt')).set_classes(['cup'])
from faster_whisper import WhisperModel
WhisperModel('base.en', device='cpu', compute_type='int8', download_root='$MODELS/whisper')
")

step "Build ($WS)"
cd "$WS"
# A build right after new asset files sometimes fails once; a second run fixes it.
colcon build --symlink-install || colcon build --symlink-install

step "Done"
echo "Open a new terminal (or: source ~/.bashrc), then run:"
echo "    ros2 launch $WS/launch/lelamp.launch.py"
