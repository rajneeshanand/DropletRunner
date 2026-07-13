#!/bin/bash
# One-time setup for DreamerV3 training on Lehigh Sol HPC
#
# Run from an interactive GPU node:
#   srun -p ima40-gpu -A mvk2_113026 --gres=gpu:1 --time=01:00:00 --pty bash
#   cd ~/mvk2_proj/raa524/droplet_runner
#   bash setup_hpc.sh

set -e

PROJ_DIR="$HOME/mvk2_proj/raa524/droplet_runner"

echo "============================================"
echo "DropletRunner HPC Setup"
echo "Project dir: $PROJ_DIR"
echo "============================================"

cd "$PROJ_DIR"

# Load modules
module load anaconda3 2>/dev/null || true

# Create conda environment
echo ""
echo "Creating conda environment 'dreamer'..."
if conda info --envs 2>/dev/null | grep -q "dreamer"; then
    echo "Environment 'dreamer' already exists"
    source activate dreamer 2>/dev/null || conda activate dreamer
else
    conda create -n dreamer python=3.10 -y
    source activate dreamer 2>/dev/null || conda activate dreamer
fi

# Install JAX with CUDA
echo ""
echo "Installing JAX with CUDA support..."
pip install --upgrade "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Clone and install Thomas's DreamerV3
echo ""
echo "Installing DreamerV3 from CyberRunner..."
if [ ! -d "$HOME/cyberrunner" ]; then
    git clone https://github.com/thomasbi1/cyberrunner.git "$HOME/cyberrunner"
fi

# Need pip<=24.0 for gym==0.19.0 compatibility
pip install --upgrade pip==24.0
pip install -e "$HOME/cyberrunner/dreamerv3"
pip install --upgrade pip  # restore pip

# Additional deps
pip install numpy opencv-python-headless tensorboard

# Verify GPU
echo ""
echo "Verifying GPU access..."
python3 -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
gpu_count = len([d for d in devices if d.platform == 'gpu'])
print(f'GPUs available: {gpu_count}')
assert gpu_count > 0, 'No GPU found!'
print('GPU OK!')
"

# Verify DreamerV3 import
echo ""
echo "Verifying DreamerV3..."
python3 -c "
import dreamerv3
import embodied
print('DreamerV3 OK!')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Transfer episodes (already done if you followed instructions)"
echo "  2. Submit training job:"
echo "     cd $PROJ_DIR"
echo "     sbatch train_dreamer.slurm"
echo "============================================"
