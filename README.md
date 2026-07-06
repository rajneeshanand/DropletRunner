# DropletRunner: Autonomous Droplet Navigation via Model-Based Reinforcement Learning

A complete guide to setting up, running, and training the DropletRunner system — from hardware assembly through data collection, offline DreamerV3 training on HPC, and real-time policy deployment.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Hardware Setup](#3-hardware-setup)
4. [Software Setup: Laptop (Ubuntu)](#4-software-setup--laptop-ubuntu)
5. [Software Setup: HPC (Sol Cluster)](#5-software-setup--hPC-sol-cluster)
6. [Camera Calibration (OCamCalib)](#6-camera-calibration-ocamcalib)
7. [Defining a new geometry (End-to-End workflow)](#7-defining-a-new-geometry-end-to-end-example)
8. [Phase 1: Data collection with random mode](#8-phase-1--random-data-collection)
9. [Phase 2: Transfer episodes to HPC and Train](#9-phase-2--transfer-episodes-to-hpc-and-train)
10. [Phase 3: Deploy trained policy on hardware](#10-phase-3--deploy-trained-policy-on-hardware)
11. [Phase 4: Iterative improvement (Cycle 2+)](#11-phase-4--iterative-improvement-cycle-2)
12. [Switching between geometries](#12-switching-between-geometries)
13. [Running the PID Baseline](#13-running-the-pid-baseline)
14. [Running the open-loop ablation](#14-running-the-open-loop-ablation)
15. [Troubleshooting](#15-troubleshooting)
16. [Configuration reference](#16-configuration-reference)
17. [Important rules and lessons learned](#17-important-rules-and-lessons-learned)

---

## 1. System Overview

DropletRunner uses a two-axis tilt platform to navigate a liquid droplet through maze-like geometries. A DreamerV3 world-model agent learns to control the platform from offline experience. The workflow follows an **offline MBRL paradigm**:

```
[Laptop] Collect random episodes on hardware
        ↓  scp episodes/
[HPC]   Train DreamerV3 offline (HPC GPU takes approximately 45 min for 50,000 steps)
        ↓  scp checkpoint.ckpt
[Laptop] Deploy trained policy (CPU-only inference)
```
This cycle repeats: if the policy is not good enough, collect more episodes (using the policy with exploration noise) and retrain.


## 2. Repository Structure

### Laptop Workspace (`~/droplet_runner_ws/`)

```
droplet_runner_ws/
├── src/
│   ├── droplet_camera/         # ROS2 package: camera publisher
│   │   └── droplet_camera/
│   │       └── cam_publisher.py
│   ├── droplet_motor/          # ROS2 package: Dynamixel motor driver
│   │   └── droplet_motor/
│   │       └── motor_driver.py
│   └── droplet_state/          # ROS2 package: state estimation + control
│       ├── calib/
│       │   └── calib_results.txt       # OCamCalib calibration file
│       ├── config/
│       |   ├── corner_markers.json                     # Board corner pixel coordinates
│       │   ├── path_waypoints_mm.json                  # SYMLINK → waypoints for active geometry 
│       │   ├── path_waypoints_mm_Ishape.json           # Waypoints for I-shape
│       │   ├── path_waypoints_mm_Lshape.json           # Waypoints for L-shape
│       │   ├── path_waypoints_mm_Lshape_outside.json   # Waypoints for L-out-shape    
│       │   ├── path_waypoints_mm_ARC.json              # Waypoints for Arc-shape
│       │   └── path_waypoints_mm_ARC_outside.json      # Waypoints for Arc-out-shape
│       └── droplet_state/
│           ├── camera_model.py         # OCamCalib Scaramuzza model
│           ├── droplet_detector.py     # HSV red-droplet detection
│           ├── marker_detector.py      # HSV blue-corner detection
│           ├── estimator.py            # State estimator + Kalman filter
│           ├── path_tracker.py         # Progress + reward computation
│           ├── data_collector.py       # Episode collection (random or explore)
│           ├── policy_runner.py        # Trained policy deployment
│           ├── pid_controller.py       # PID baseline controller
│           └── open_loop_replay.py     # Open-loop ablation
├── episodes/                   # Collected episode .npz files
├── policy/
│   ├── checkpoint.ckpt         # Active DreamerV3 checkpoint
│   ├── checkpoint_Ishape.ckpt  # Saved checkpoints for I-shape
│   ├── checkpoint_Lshape.ckpt  # Saved checkpoints for L-shape
│   └── ...
├── action_logs/                # CSV logs from policy_runner
└── pid_logs/                   # CSV logs from pid_controller
```

### HPC Directory (`~/mvk2_proj/raa524/droplet_runner/`)

```
droplet_runner/
├── train_offline.py        # Offline DreamerV3 training script
├── train_dreamer.slurm     # SLURM job submission script
├── setup_hpc.sh            # One-time HPC environment setup
├── episodes/               # Episodes transferred from laptop
└── logdir/
    ├── checkpoint.ckpt     # Trained checkpoint (transfer back to laptop)
    ├── config.yaml         # DreamerV3 config 
    ├── metrics.jsonl       # Training metrics log
    └── replay/             # Embodied replay buffer (chunked format)
```


## 3. Hardware Setup

### Components

| Component | Model | Role |
|-----------|-------|------|
| Tilt motors (x2) | Dynamixel XL330-M077-T | Two-axis current-mode actuation |
| USB interface | U2D2  | Motor ↔ USB communication |
| Camera | See3CAM_24CUG  | Overhead vision at 1280×720 |
| Board | 3D-printed PLA | Tilting platform |
| Surface | AR 20 silicone oil film | Low-friction droplet transport |
| Droplet | 20-30 microliter red-dyed water droplet volume | Navigation target |
| Corner markers | 4 blue dots on board corners | Tilt estimation reference |

### Physical Assembly (See Figure 1)

1. Mount the two Dynamixel motors orthogonally beneath the PLA board (Motor 1 = axis 1, Motor 2 = axis 2).
2. Connect both motors to the U2D2 interface (daisy-chain with 3-pin Dynamixel cable).
3. Connect U2D2 to laptop via USB (appears as `/dev/ttyUSB0`).
4. Mount See3CAM_24CUG directly above the board center, looking down.
5. Connect camera to laptop via USB.
6. Paint or mark the desired path geometry on the board surface (black dotted line).
7. Place 4 small blue markers at the board corners (TL, TR, BR, BL).
8. Coat the board surface with silicone oil (AR 20).

<p align="center">
  <img src="images/Figure S1.png" alt="DropletRunner Hardware Assembly" width="700">
</p>
<p align="center"><b>Figure 1.</b> DropletRunner platform assembly.</p>

## 4. Software Setup — Laptop (Ubuntu) (see setup_laptop.sh for details)

### 4.1 Prerequisites

- Ubuntu 22.04 LTS
- ROS2 Humble (full desktop install)
- Python 3.10
- Conda (Miniconda or Anaconda)

### 4.2 Install ROS2 Humble

```bash
# Follow official ROS2 installation:
# https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html

sudo apt update && sudo apt install -y \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-rosdep

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 4.3 Create a Conda Environment for CPU Inference 

```bash
conda create -n droplet python=3.10 -y
conda activate droplet
```

### 4.4 Install Python Dependencies 

```bash
# Core dependencies
pip install numpy opencv-python-headless opencv-contrib-python

# ROS2 bridge
pip install cv-bridge-msgs

# Dynamixel SDK
pip install dynamixel-sdk

# JAX
pip install "jax[cpu]"

# DreamerV3 dependencies
pip install tensorflow-probability ruamel.yaml rich cloudpickle

# For analysis/plotting (optional)
pip install matplotlib pandas scipy
```

### 4.5 Clone and Install DreamerV3 (Thomas's CyberRunner Fork)

```bash
cd ~
git clone https://github.com/thomasbi1/cyberrunner.git

# Need pip<=24.0 for gym==0.19.0 compatibility
pip install --upgrade pip==24.0
pip install -e ~/cyberrunner/dreamerv3
pip install --upgrade pip   # restore latest pip
```

### 4.6 Create and Build the ROS2 Workspace 

```bash
mkdir -p ~/droplet_runner_ws/src
# Copy the three packages into src/:
#   droplet_camera/
#   droplet_motor/
#   droplet_state/

cd ~/droplet_runner_ws
colcon build --symlink-install
source install/setup.bash
```

### 4.7 Create Supporting Directories

```bash
mkdir -p ~/droplet_runner_ws/episodes
mkdir -p ~/droplet_runner_ws/policy
mkdir -p ~/droplet_runner_ws/policy/replay
mkdir -p ~/droplet_runner_ws/action_logs
mkdir -p ~/droplet_runner_ws/pid_logs
```

### 4.8 Set Environment Variables

Add to `~/.bashrc` (or run before every session):

```bash
# ROS2
source /opt/ros/humble/setup.bash
source ~/droplet_runner_ws/install/setup.bash

# DreamerV3 on CPU
export PYTHONPATH="$HOME/cyberrunner/dreamerv3:$HOME/cyberrunner/dreamerv3/dreamerv3:$PYTHONPATH"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
```


## 5. Software Setup — HPC (see setup_hpc.sh for details)

### 5.1 One-Time Setup

Follow setup_hpc.sh

### 5.2 Verify GPU Manually (Optional)

```bash
srun -p ima40-gpu -A mvk2_113026 --gres=gpu:1 --time=00:10:00 --pty bash

module load miniconda3/24.7.1
module load cuda/12.6.2
conda activate dreamer

export LD_LIBRARY_PATH=/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi/extras/CUPTI/lib64:/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi"

python3 -c "import jax; print(jax.devices())"
# Should show: [GpuDevice(id=0, ...)]
```


## 6. Camera Calibration (OCamCalib)

The system uses the Scaramuzza omnidirectional camera model (OCamCalib toolbox for MATLAB). Calibration was performed at 1024×768 resolution, and the camera runs at 1280×720. The code handles the rescaling internally.

The calibration file is at `src/droplet_state/calib/calib_results.txt` and contains:
- Direct polynomial (`ss`): pixel → 3D ray direction
- Inverse polynomial (`invpol`): 3D point → pixel coordinates
- Optical center (`cx`, `cy`) in calibration resolution
- Affine parameters (`c`, `d`, `e`)

**One should not need to recalibrate** unless the camera or its mounting changes. If recalibration is needed, use the OCamCalib MATLAB toolbox with a checkerboard pattern at the native 1024×768 resolution.

## WORKFLOW of the code/commands in each terminal

## 7. Defining a New Geometry (End-to-End Example)

This section walks through adding a completely new path geometry. We will use a hypothetical **"U-shape"** as the example.

### Step 7.1: Paint the path on the board

Using a marker or paint, draw the desired path on the silicone-oil-coated PLA board. Mark the start point and end point clearly. For curved paths, add evenly-spaced reference dots along the curve.

### Step 7.2: Measure waypoint coordinates

Launch the estimator to get real-time mm coordinates:

```
ros2 run droplet_state estimator
```

**Terminal 1: Camera:**
```bash
cd ~/droplet_runner_ws && source install/setup.bash
ros2 run droplet_camera cam_publisher --ros-args -p device_id:=4 -p width:=1280 -p height:=720    # device_id:= xxxx depends on the cable number connected with laptop, check it
```

**Terminal 2: Motor driver** (needed for estimator to initialize, even if not used):
```bash
cd ~/droplet_runner_ws && source install/setup.bash
ros2 run droplet_motor motor_driver --ros-args -p port:=/dev/ttyUSB0
```

**Terminal 3: State estimator:**
```bash
cd ~/droplet_runner_ws && source install/setup.bash
export PYTHONPATH="$HOME/cyberrunner/dreamerv3:$HOME/cyberrunner/dreamerv3/dreamerv3:$PYTHONPATH"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
ros2 run droplet_state estimator
```

Now place a droplet on each reference point along the path and read off the `(x_mm, y_mm)` coordinates from the estimator log. Record 10–15 points along the path, always starting with the start point and ending with the end point.

**Example measurements for our U-shape:**
```
Point 1 (start):  (80.0, -50.0) mm
Point 2:          (80.0, -20.0) mm
Point 3:          (80.0,  10.0) mm
Point 4 (curve):  (60.0,  30.0) mm
Point 5 (curve):  (50.0,  30.0) mm
Point 6 (curve):  (40.0,  30.0) mm
Point 7 (curve):  (35.0,  30.0) mm
Point 8 (curve):  (30.0,  30.0) mm
Point 9 (curve):  (25.0,  30.0) mm
Point 10 (curve):  (20.0,  30.0) mm
Point 11:          (10.0,  10.0) mm
Point 12:          (10.0, -20.0) mm
Point 13 (end):    (10.0, -50.0) mm
```

### Step 7.3: Generate dense waypoints

There are two cases:

**Case A: Straight-line segments (I-shape, L-shape, L-out-shape)**

For paths made of straight segments connected by corners, generate waypoints by linearly interpolating between measured points at ~2 mm spacing.

```python
import numpy as np
import json

# Measured corner/turn points in order
measured_points = [
    [80.0, -50.0],   # start
    [80.0,  30.0],   # top of left side
    [10.0,  30.0],   # top of right side
    [10.0, -50.0],   # end
]

# Generate waypoints at ~2mm spacing along each segment
waypoints = []
pts = np.array(measured_points)
for i in range(len(pts) - 1):
    a, b = pts[i], pts[i + 1]
    dist = np.linalg.norm(b - a)
    n_points = max(int(dist / 2.0), 2)
    for t in np.linspace(0, 1, n_points, endpoint=(i == len(pts) - 2)):
        waypoints.append((a + t * (b - a)).tolist())

print(f"Generated {len(waypoints)} waypoints")
print(f"Total path length: {sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)):.1f} mm")

with open("path_waypoints_mm_Ushape.json", "w") as f:
    json.dump(waypoints, f, indent=2)
```

**Case B: Curved paths (Arc-shape, Arc-out-shape)**

For arc/curved paths, first fit a circle to the measured points using Kasa's least-squares method, then generate waypoints along the fitted arc.

```python
import numpy as np
import json

# Measured points along the curve
points = np.array([
    [80.0, -50.0],   # start
    [70.0, -30.0],
    [55.0, -15.0],
    [35.0,  -5.0],
    [15.0, -15.0],
    [ 5.0, -30.0],
    [-5.0, -50.0],   # end
])

# Kasa circle fit: solve x^2+y^2+Dx+Ey+F=0
x, y = points[:, 0], points[:, 1]
A = np.column_stack([x, y, np.ones(len(x))])
b = -(x**2 + y**2)
result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
D, E, F = result
cx, cy = -D/2.0, -E/2.0
R = np.sqrt(cx**2 + cy**2 - F)

print(f"Circle center: ({cx:.2f}, {cy:.2f}) mm")
print(f"Radius: {R:.2f} mm")

# Generate waypoints along the arc
theta_start = np.arctan2(points[0,1]-cy, points[0,0]-cx)
theta_end = np.arctan2(points[-1,1]-cy, points[-1,0]-cx)

# Ensure correct sweep direction (check midpoint)
theta_mid = np.arctan2(points[len(points)//2,1]-cy, points[len(points)//2,0]-cx)
if theta_start > theta_end:
    theta_range = np.unwrap([theta_start, theta_mid, theta_end])
    theta_start, theta_end = theta_range[0], theta_range[-1]

arc_length = abs(theta_end - theta_start) * R
n_waypoints = max(int(arc_length / 2.0), 10)
thetas = np.linspace(theta_start, theta_end, n_waypoints)
waypoints = [[float(cx + R*np.cos(t)), float(cy + R*np.sin(t))] for t in thetas]

print(f"Generated {len(waypoints)} waypoints over {np.degrees(abs(theta_end-theta_start)):.1f}° sweep")

with open("path_waypoints_mm_Ushape.json", "w") as f:
    json.dump(waypoints, f, indent=2)
```

### Step 7.4: Install the Waypoints File

```bash
# Copy the generated file to the config directory
cp ~/path_waypoints_mm_Ushape.json (#wherever your filepath) \~/droplet_runner_ws/src/droplet_state/config/

# Create (or update) the symlink to activate this geometry
cd ~/droplet_runner_ws/src/droplet_state/config/
ln -sf path_waypoints_mm_Ushape.json path_waypoints_mm.json
```

### Step 7.5: Adjust path tracker parameters (if needed)

Edit `path_tracker.py` if the new geometry requires different tolerances:

```python
self.max_deviation = 30.0    # mm, increase for wider paths, decrease for narrow channels
self.goal_radius = 10.0      # mm, success threshold near the end waypoint
```

For narrow-channel geometries (like the S-shape with painted walls), use `max_deviation = 20.0`. For open-surface geometries (I-shape, L-shape, arcs), use `max_deviation = 30.0`.

Also in `data_collector.py`:
```python
self.max_episode_steps = 600  # Increase for longer paths (e.g., 1000 for S-shape)
```

### Step 7.6: Rebuild the ROS2 Workspace

```bash
cd ~/droplet_runner_ws
colcon build --symlink-install
source install/setup.bash
```


## 8. Phase 1: Random Data Collection

### 8.1 Launch the Hardware Nodes

It needs **five terminals**. In each, first run:
```bash
cd ~/droplet_runner_ws && source install/setup.bash
export PYTHONPATH="$HOME/cyberrunner/dreamerv3:$HOME/cyberrunner/dreamerv3/dreamerv3:$PYTHONPATH"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
```

**Terminal 1: Camera**
```bash
ros2 run droplet_camera cam_publisher --ros-args -p device_id:=4 -p width:=1280 -p height:=720
```

> **Note:** The `device_id` changes when USB devices are plugged/unplugged. If the camera fails to open, find the correct device:
> ```bash
> v4l2-ctl --list-devices
> ```
> Look for the See3CAM entry and use the number from `/dev/videoN`.

**Terminal 2: Run RQT**
In the second terminal, just write ```rqt``` and hit enter


**Terminal 3: Motor driver**
```bash
ros2 run droplet_motor motor_driver --ros-args -p port:=/dev/ttyUSB0
```

**Terminal 4: State estimator**
```bash
ros2 run droplet_state estimator
```

**Terminal 5: Data collector**
```bash
ros2 run droplet_state data_collector
```

### 8.2 Collect Episodes

The data collector will print: `Place droplet at start, then press ENTER to begin episode.`

For each episode:
1. Place a fresh red-dyed droplet (20-30 microliter volume) at the path start position.
2. Press ENTER in Terminal 5.
3. The system applies random motor commands while recording (image, vector, action, reward).
4. The episode ends automatically when the droplet goes out of bounds (`max_deviation`), reaches the goal, or hits `max_episode_steps`.
5. Wipe the droplet (if it breaks) or bring it at start position, re-oil if needed, and repeat.

**Collect 50 episodes** for the first training cycle. Each episode saves as `episode_XXXX.npz` in `~/droplet_runner_ws/episodes/`.

### 8.3 What Each Episode Contains

```
episode_XXXX.npz:
  image:       (T, 64, 64, 3)   uint8    — 64×64 RGB crop centered on droplet
  vector:      (T, 14)          float32  — [x_mm, y_mm, alpha, beta, 10× lookahead]
  action:      (T, 2)           float32  — [motor1_current, motor2_current] in [-150, 150]
  reward:      (T,)             float32  — progress_delta - λ·distance_penalty
  is_first:    (T,)             bool     — True only at step 0
  is_terminal: (T,)             bool     — True at final step
```

### 8.4 Verify Collected Data (not necesaary always, can ignore)

```python
import numpy as np
import os

ep_dir = os.path.expanduser("~/droplet_runner_ws/episodes")
files = sorted([f for f in os.listdir(ep_dir) if f.endswith(".npz")])
print(f"Episodes: {len(files)}")

for f in files[:5]:
    data = np.load(os.path.join(ep_dir, f))
    T = len(data["reward"])
    total_r = data["reward"].sum()
    print(f"  {f}: {T} steps, reward={total_r:.1f}")
```


## 9. Phase 2: Transfer Episodes to HPC and Train

### 9.1 Critical pre-training cleanup on HPC

**ALWAYS perform this cleanup before starting a fresh training run.** Failure to do so causes the replay buffer to mix old and new episodes, leading to corrupted training.

```bash
ssh username

cd ~/directory_path/droplet_runner

# 1. Delete old replay buffer
rm -rf logdir/replay/*

# 2. Delete old checkpoint (for fresh training only)
rm -f logdir/checkpoint.ckpt

# 3. Delete old episodes
rm -f episodes/episode_*.npz
```

**Exception:** When doing continuation training (Cycle 2+), keep `logdir/checkpoint.ckpt` but still delete `logdir/replay/*` and replace episodes.

### 9.2 Transfer Episodes from Laptop to HPC

```bash
# From the laptop:
scp ~/droplet_runner_ws/episodes/episode_*.npz \
    raa524@sol.cc.lehigh.edu (username for HPC):~/directory_path/droplet_runner/episodes/
```

### 9.3 Adjust Training Parameters (if needed)

SSH into Sol and check `train_offline.py`:

```bash
ssh username
cd ~/directory_path/droplet_runner
grep -n "batch_length\|batch_size\|steps" train_offline.py
```

Default training parameters:
```python
'batch_size': 16,
'batch_length': 64,
# --steps 50000 (command-line argument)
```

**When to change `batch_length`:**
- If your episodes are shorter than 64 steps on average, set `batch_length = 32`:
  ```bash
  sed -i "s/'batch_length': 64/'batch_length': 32/" train_offline.py
  ```
- For very long episodes (>500 steps, e.g., S-shape), consider `batch_length = 128` and `--steps 100000`.

### 9.4 Submit the training job

```bash
cd ~/mvk2_proj/raa524/droplet_runner
mkdir -p logdir
sbatch train_dreamer.slurm
```

To use the faster `lake-gpu` partition:
```bash
sbatch --partition=lake-gpu train_dreamer.slurm
```

### 9.5 Monitor training

```bash
# Check job status
squeue -u raa524

# Watch training log in real time
tail -f logdir/train_JOBID.log

# Check for errors
nano logdir/train_JOBID.err
```

Training takes usually 45 minutes for 50,000 steps on `lake-gpu` (L40S), ~85 minutes on `ima40-gpu` (A40).

**Signs of successful training:**
- Log file shows training metrics updating every ~60 seconds.
- `logdir/checkpoint.ckpt` file grows (should be ~10–20 MB).
- `logdir/metrics.jsonl` is populated with loss values.

**Signs of failure:**
- Training completes in <1 minute → JAX fell back to CPU. Check the `.err` file for CUDA/cuPTI errors.
- No `checkpoint.ckpt` saved → training crashed. Check `.err` file.

### 9.6 Transfer Checkpoint Back to Laptop

```bash
# From the laptop:
scp raa524@sol.cc.lehigh.edu:~/mvk2_proj/raa524/droplet_runner/logdir/checkpoint.ckpt \
    ~/droplet_runner_ws/policy/checkpoint.ckpt
```

**Always back up the checkpoint with a geometry-specific name:**
```bash
cp ~/droplet_runner_ws/policy/checkpoint.ckpt \
   ~/droplet_runner_ws/policy/checkpoint_Ushape.ckpt
```


## 10. Phase 3: Deploy Trained Policy on Hardware

### 10.1 Launch hardware nodes (Same as Data collection)

**Terminals 1–3:** Camera, rqt, motor driver, estimator (same commands as Section 8.1).

**Terminal 4 — Policy runner (instead of data collector):**
```bash
cd ~/droplet_runner_ws && source install/setup.bash
export PYTHONPATH="$HOME/cyberrunner/dreamerv3:$HOME/cyberrunner/dreamerv3/dreamerv3:$PYTHONPATH"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
ros2 run droplet_state policy_runner
```

### 10.2 Run evaluation episodes

1. Place droplet at start position.
2. Press ENTER in Terminal 5.
3. Watch the trained policy navigate the droplet.
4. The episode ends when the droplet reaches the goal, drifts out of bounds, or hits the step limit.
5. Action logs are saved to `~/droplet_runner_ws/action_logs/mbrl_actions_epHHMMSS.csv`.

### 10.3 Evaluate success rate

Run 20 episodes and record success/failure. Typical results:
- **I-shape (straight):** 100% after Cycle 1 (50 episodes, 50,000 steps)
- **L-shape (90° turn):** 95% after Cycle 2 (100 episodes, 50,000 steps)
- **Arc-inside:** 100% after Cycle 3 (with `batch_length=32`)

If the success rate is below the desired target, proceed to Cycle 2 (Section 11).


## 11. Phase 4: Iterative improvement (Cycle 2+)

### 11.1 Collect policy-guided episodes

Instead of random episodes, use the trained policy with exploration noise:

```bash
ros2 run droplet_state data_collector
```

The data collector automatically detects `~/droplet_runner_ws/policy/checkpoint.ckpt` and uses the trained policy with `mode='explore'` (adds exploration noise). If no checkpoint exists, it falls back to the random policy.

Collect 50 more episodes with the trained policy. These episodes are richer because the droplet reaches further along the path, providing the world model with experience in regions the random policy rarely visits.

### 11.2 Retrain (Two strategies)

**Strategy A: Fresh training (recommended for first few cycles):**
```bash
# On HPC: full cleanup + transfer all episodes (old + new)
rm -rf logdir/replay/*
rm -f logdir/checkpoint.ckpt
rm -f episodes/episode_*.npz
```
Then transfer ALL episodes (from both cycles) and train.

**Strategy B: Continuation training:**
```bash
# On HPC: keep checkpoint, clean replay, transfer all episodes
rm -rf logdir/replay/*
rm -f episodes/episode_*.npz
```
Transfer all episodes and train. The checkpoint is loaded and training continues from the previous state. **Warning:** If episodes are short and `batch_length` is too large, continuation training can degrade performance.

### 11.3 Iterate

Repeat the cycle until the success rate is satisfactory. Typically 2–3 cycles suffice for most geometries.


## 12. Switching Between Geometries

### To switch to a different geometry (e.g., from U-shape to L-shape):

```bash
# Step 1: Switch the waypoints symlink
cd ~/droplet_runner_ws/src/droplet_state/config/
ln -sf path_waypoints_mm_Lshape.json path_waypoints_mm.json

# Step 2: Switch the checkpoint
cp ~/droplet_runner_ws/policy/checkpoint_Lshape.ckpt \
   ~/droplet_runner_ws/policy/checkpoint.ckpt
```

That is it. No code changes, no rebuild. The `path_tracker.py`, `data_collector.py`, and `policy_runner.py` all read `path_waypoints_mm.json` at startup.

### To save the current geometry's Checkpoint before switching:

```bash
cp ~/droplet_runner_ws/policy/checkpoint.ckpt \
   ~/droplet_runner_ws/policy/checkpoint_Ushape.ckpt
```


## 13. Running the PID Baseline

The PID controller serves as a model-free baseline for comparison:

```bash
# Terminals 1-4: camera, rqt, motor, estimator (same as always)
# Terminal 5:
ros2 run droplet_state pid_controller
```

It uses the same path tracker and waypoints. PID gains are tuned in `pid_controller.py`:
```python
self.K_p = 4.0
self.K_d = 2.0
self.K_i = 0.1
```

Results are saved to `~/droplet_runner_ws/pid_results.json` and per-episode CSV logs to `~/droplet_runner_ws/pid_logs/`.


## 14. Running the Open-Loop Ablation

The open-loop replay plays back motor commands from a successful closed-loop episode WITHOUT camera feedback, demonstrating that the MBRL policy actively uses visual feedback.

```bash
# Terminal 5 (camera, rqt, motor, estimator must be running):
ros2 run droplet_state open_loop_replay \
    --ros-args -p csv_file:=/path/to/successful_episode.csv
```

Use a CSV file from `action_logs/` recorded during a successful policy_runner episode. The replay sends the same motor commands but does not read the camera, the droplet will drift off the path, proving closed-loop necessity.


## 15. Troubleshooting

### Camera Not Found

The device ID shifts when USB devices are plugged/unplugged:
```bash
v4l2-ctl --list-devices
# Find the See3CAM entry, note the /dev/videoN number
ros2 run droplet_camera cam_publisher --ros-args -p device_id:=N
```

### Motor errors / torque overload

The motor driver automatically reboots motors on startup. If motors become unresponsive:
1. Power-cycle the U2D2.
2. Check `/dev/ttyUSB0` exists: `ls -la /dev/ttyUSB*`
3. Ensure your user is in the `dialout` group: `sudo usermod -aG dialout $USER`

### Droplet Not Detected

- Verify the red dye is dark enough (HSV saturation ≥ 150).
- Check proper lighting and whether it is visible in RQT.
- Check the debug image topic: `ros2 topic echo /debug_image --no-arr | head`
- Adjust HSV ranges in `droplet_detector.py` if using a different dye color.

### JAX falls back to CPU on HPC

If training completes in seconds instead of minutes, JAX is not using the GPU. Verify the SLURM script includes:
```bash
export LD_LIBRARY_PATH=/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi/extras/CUPTI/lib64:/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi"
```

### "Not enough data" Error during training

The total number of steps across all episodes must exceed `batch_size × batch_length` (default: 16 × 64 = 1024 steps). Collect more episodes or reduce `batch_length`.

### Replay Buffer contamination

If a trained policy behaves erratically, the replay buffer may contain episodes from a different geometry. **Always** run the full cleanup sequence (Section 9.1) before starting fresh training.


## 16. Important rules and lessons learned

1. **Never manually tilt the board during data collection.** Manual tilting corrupts the action data because recorded motor commands do not match the applied forces. Use only the random policy or trained policy for data collection.

2. **Always back up checkpoints before overwriting.** Name them by geometry: `checkpoint_Ishape.ckpt`, `checkpoint_Lshape.ckpt`, etc.

3. **Always run the pre-training cleanup sequence on HPC** (delete `replay/*`, `checkpoint.ckpt`, old `episodes/`) before fresh training. Mixing episodes from different geometries is the most common source of training failure.

4. **Set `batch_length = 32`** when the average episode length is less than 64 steps (common for arc geometries with tight deviation thresholds).

5. **Episode actions are stored in raw current units** ([-150, 150]). The `train_offline.py` script normalizes them by dividing by 150.0 to get [-1, 1] for the agent. The `policy_runner.py` multiplies the agent's output by 150.0 to convert back. These must match.

6. **DreamerV3 runs on CPU during deployment** Always set `JAX_PLATFORMS=cpu` and `CUDA_VISIBLE_DEVICES=""` on the laptop to avoid attempting GPU initialization.

7. **OCamCalib distortion amplifies coordinates near board edges.** For geometries that approach the board edges, add intermediate waypoints to prevent large coordinate jumps. 

8. **Fixed lighting inside the lab** — Do not light up too much near the setup, it degrades the performance. 

---

*DropletRunner — Rajneesh Anand from Kothare Research Group, Lehigh University*
