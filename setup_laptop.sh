#!/bin/bash


####

# DropletRunner: Laptop (NO GPU) environment setup (Ubuntu 22.04 + ROS2 Humble)

####

# This script sets up the complete software environment on the laptop used for
# data collection and policy deployment. Run it step-by-step or as a script.

# Prerequisites:
#   - Ubuntu 22.04 LTS
#   - Conda (Miniconda or Anaconda) installed
#   - Internet connection

# Usage:
#   chmod +x setup_laptop.sh
#   ./setup_laptop.sh


set -e

echo "============================================"
echo "DropletRunner — Laptop Setup"
echo "============================================"


# Step 1: Install ROS2 Humble (if not installed)

echo ""
echo "[Step 1] Checking ROS2 Humble..."

if [ -f /opt/ros/humble/setup.bash ]; then
    echo "  ROS2 Humble already installed."
else
    echo "  Installing ROS2 Humble..."
    sudo apt update && sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    sudo apt update && sudo apt install -y curl gnupg lsb-release

    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    sudo apt update
    sudo apt install -y ros-humble-desktop
fi


# Step 2: Install system dependencies

echo ""
echo "[Step 2] Installing system dependencies..."

sudo apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    v4l-utils \
    git

if [ ! -d /etc/ros/rosdep/sources.list.d ]; then
    sudo rosdep init
fi
rosdep update --rosdistro humble 2>/dev/null || true


# Step 3: Create conda environment

echo ""
echo "[Step 3] Setting up conda environment 'droplet'..."

if conda info --envs 2>/dev/null | grep -q "droplet"; then
    echo "  Environment 'droplet' already exists."
    eval "$(conda shell.bash hook)"
    conda activate droplet
else
    conda create -n droplet python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate droplet
fi


# Step 4: Install Python packages

echo ""
echo "[Step 4] Installing Python packages..."

pip install numpy==1.26.4
pip install opencv-python-headless==4.10.0.84
pip install opencv-contrib-python==4.10.0.84

pip install "jax[cpu]"                                      #JAX (CPU only for laptop inference) 

pip install dynamixel-sdk

pip install tensorflow-probability                          #dreamerV3 dependency
pip install ruamel.yaml
pip install rich
pip install cloudpickle

pip install pyyaml
pip install setuptools==58.2.0   #required for ROS2 colcon build

pip install matplotlib
pip install pandas
pip install scipy


#Step 5: Clone and install DreamerV3
echo ""
echo "[Step 5] Installing DreamerV3 (CyberRunner fork)..."

if [ ! -d "$HOME/cyberrunner" ]; then
    git clone https://github.com/thomasbi1/cyberrunner.git "$HOME/cyberrunner"
fi

pip install --upgrade pip==24.0
pip install -e "$HOME/cyberrunner/dreamerv3"
pip install --upgrade pip   # Restore latest pip


# Step 6: create ROS2 workspace

echo ""
echo "[Step 6] Setting up ROS2 workspace..."

mkdir -p ~/droplet_runner_ws/src
mkdir -p ~/droplet_runner_ws/episodes
mkdir -p ~/droplet_runner_ws/policy/replay
mkdir -p ~/droplet_runner_ws/action_logs
mkdir -p ~/droplet_runner_ws/pid_logs

echo "  Workspace directory created at ~/droplet_runner_ws/"
echo "  Copy the three ROS2 packages into ~/droplet_runner_ws/src/:"
echo "    - droplet_camera/"
echo "    - droplet_motor/"
echo "    - droplet_state/"


# Step 7: add USB permissions

echo ""
echo "[Step 7] Setting up USB permissions..."

if groups | grep -q dialout; then
    echo "  User already in dialout group."
else
    sudo usermod -aG dialout "$USER"
    echo "  Added $USER to dialout group (log out and back in for this to take effect)."
fi


# Step 8: add environment variables to .bashrc

echo ""
echo "[Step 8] Configuring environment variables..."

BASHRC_MARKER="# --- DropletRunner environment ---"
if grep -q "$BASHRC_MARKER" ~/.bashrc; then
    echo "  DropletRunner environment already in .bashrc"
else
    cat >> ~/.bashrc << 'DROPLET_ENV'


source /opt/ros/humble/setup.bash
if [ -f ~/droplet_runner_ws/install/setup.bash ]; then
    source ~/droplet_runner_ws/install/setup.bash
fi
export PYTHONPATH="$HOME/cyberrunner/dreamerv3:$HOME/cyberrunner/dreamerv3/dreamerv3:$PYTHONPATH"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
DROPLET_ENV
    echo "  Added environment variables to ~/.bashrc"
fi


# Step 9: Verification

echo ""
echo "[Step 9] Verifying installation..."

echo -n "  Python: "
python3 --version

echo -n "  NumPy: "
python3 -c "import numpy; print(numpy.__version__)"

echo -n "  OpenCV: "
python3 -c "import cv2; print(cv2.__version__)"

echo -n "  JAX: "
python3 -c "import jax; print(jax.__version__); print(f'  Devices: {jax.devices()}')"

echo -n "  DreamerV3: "
python3 -c "import dreamerv3; import embodied; print('OK')"

echo -n "  Dynamixel SDK: "
python3 -c "from dynamixel_sdk import PortHandler; print('OK')"

echo -n "  ROS2: "
ros2 --version 2>/dev/null || echo "not available in this shell (source setup.bash)"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy the three ROS2 packages to ~/droplet_runner_ws/src/"


echo "  2. Build the workspace:"

echo "       cd ~/droplet_runner_ws"
echo "       colcon build --symlink-install"
echo "       source install/setup.bash"


echo "  3. Connect camera and motors, then test:"

echo "       ros2 run droplet_camera cam_publisher --ros-args -p device_id:=4"  #device_id may vary, check with `ls /dev/video*`
echo "       ros2 run droplet_motor motor_controller --ros-args -p device_id:=4"  #device_id may vary, check with `ls /dev/ttyUSB*`

