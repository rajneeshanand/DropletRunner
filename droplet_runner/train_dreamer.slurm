#!/bin/bash
#SBATCH --job-name=dreamer_droplet
#SBATCH --partition=ima40-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logdir/train_%j.log
#SBATCH --error=logdir/train_%j.err
#SBATCH --account=mvk2_113026

# DropletRunner DreamerV3 training on Lehigh Sol
# Submit: cd ~/mvk2_proj/raa524/droplet_runner && sbatch train_dreamer.slurm

PROJ_DIR="$HOME/mvk2_proj/raa524/droplet_runner"

echo "============================================"
echo "DropletRunner DreamerV3 Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "============================================"

cd "$PROJ_DIR"
mkdir -p logdir

# Load environment
module load miniconda3/24.7.1
module load cuda/12.6.2 2>/dev/null || true
source activate dreamer 2>/dev/null || conda activate dreamer
export LD_LIBRARY_PATH=/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi/extras/CUPTI/lib64:/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/share/Apps/cascade24v2/gcc-12.4.0/cuda-12.6.2-4bnnvhtdkeytxwmiohbirolvkkxh6qpi"
export PYTHONPATH="$HOME/cyberrunner/dreamerv3:$HOME/cyberrunner/dreamerv3/dreamerv3:$PYTHONPATH"

# Add DreamerV3 to path
export PYTHONPATH="$HOME/cyberrunner/dreamerv3:$HOME/cyberrunner/dreamerv3/dreamerv3:$PYTHONPATH"

# Verify GPU
echo ""
nvidia-smi
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Run training
echo ""
echo "Starting training at $(date)..."
python3 train_offline.py \
    --episodes_dir episodes/ \
    --logdir logdir/ \
    --steps 50000

echo ""
echo "Training complete at $(date)"
echo "============================================"
