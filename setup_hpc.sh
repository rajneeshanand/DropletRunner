#!/bin/bash


###

# DropletRunner — HPC (Lehigh Sol, any other sol, or GPU) Environment Setup

### 

# This script sets up the training environment on Sol HPC cluster.
#
# Prerequisites:
#   - Active Sol account (for mine: raa524@sol.cc.lehigh.edu)


set -e

PROJ_DIR="$HOME/mvk2_proj/raa524/droplet_runner"  #prject directory on HPC (change as needed)


echo "DropletRunner: HPC Setup"
echo "Project dir: $PROJ_DIR"


# Step 1: create project directories

echo ""
echo "[Step 1] creating project directories..."

mkdir -p "$PROJ_DIR"/{episodes,logdir,logdir/replay}
cd "$PROJ_DIR"

# Step 2: load modules

echo ""
echo "[Step 2] loading modules..."

module load miniconda3/24.7.1
module load cuda/12.6.2 2>/dev/null || true


# Step 3: create conda environment

echo ""
echo "[Step 3] setting up conda environment 'dreamer'..."

if conda info --envs 2>/dev/null | grep -q "dreamer"; then
    echo "  Environment 'dreamer' already exists."
    source activate dreamer 2>/dev/null || conda activate dreamer
else
    conda create -n dreamer python=3.10 -y
    source activate dreamer 2>/dev/null || conda activate dreamer
fi


# Step 4: install JAX with CUDA support

echo ""
echo "[Step 4] Installing JAX with CUDA 12 support..."

pip install --upgrade "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# Step 5: install DreamerV3 from Cyberrunner 

echo ""
echo "[Step 5] Installing DreamerV3 (CyberRunner fork)..."

if [ ! -d "$HOME/cyberrunner" ]; then
    git clone https://github.com/thomasbi1/cyberrunner.git "$HOME/cyberrunner"
fi


pip install --upgrade pip==24.0
pip install -e "$HOME/cyberrunner/dreamerv3"
pip install --upgrade pip   # Restore latest pip


# Step 6: install additional dependencies

echo ""
echo "[Step 6] additional packages..."

pip install numpy opencv-python-headless tensorboard


# Step 7: set CUDA library paths

echo ""
echo "[Step 7] Configuring CUDA paths..."

CUDA_BASE="/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi"
export LD_LIBRARY_PATH="${CUDA_BASE}/extras/CUPTI/lib64:${CUDA_BASE}/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_BASE}"
export PYTHONPATH="$HOME/cyberrunner/dreamerv3:$HOME/cyberrunner/dreamerv3/dreamerv3:$PYTHONPATH"


# Step 8: check GPU access

echo ""
echo "[Step 8] Verifying GPU access..."

nvidia-smi

python3 -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
gpu_count = len([d for d in devices if d.platform == 'gpu'])
print(f'GPUs available: {gpu_count}')
assert gpu_count > 0, 'No GPU found!'
print('GPU verification: PASSED')
"


# Step 9: check DreamerV3 imports

echo ""
echo "[Step 9] checking DreamerV3 imports..."

python3 -c "
import dreamerv3
import embodied
print('DreamerV3 import: PASSED')
"


echo ""

echo "HPC Setup complete!"

echo "Key paths:"
echo "  Project:     $PROJ_DIR"
echo "  Episodes:    $PROJ_DIR/episodes/"
echo "  Logdir:      $PROJ_DIR/logdir/"
echo "  DreamerV3:   $HOME/cyberrunner/dreamerv3/"
echo "  Conda env:   dreamer"

echo "To submit a training job:"
echo "  cd $PROJ_DIR"
echo "  sbatch train_dreamer.slurm"
echo ""
echo "For faster training (L40S GPU):"
echo "  sbatch --partition=lake-gpu train_dreamer.slurm"  #for L40S GPU, if available

