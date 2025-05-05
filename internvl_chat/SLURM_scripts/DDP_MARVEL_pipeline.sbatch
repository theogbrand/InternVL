#!/bin/bash
#SBATCH --job-name=marvel_dp
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12 # 1 (main) + 8 (workers) + 3 (overhead)
#SBATCH --mem-per-cpu=4G   # Provides 48GB total RAM per task as a safe starting point
#SBATCH --output=/mnt/weka/aisg/ob1/InternVL/internvl_chat/SLURM_scripts/output_logs/slurm-dp-%j.out
#SBATCH --error=/mnt/weka/aisg/ob1/InternVL/internvl_chat/SLURM_scripts/output_logs/slurm-dp-%j.err # Redirect stderr too

# DO NOT set AUTO_SPLIT

echo "Starting SLURM script setup..."

# Activate environment if needed
echo "Loading uv module..."
module load uv
echo "Changing directory..."
cd /mnt/weka/aisg/ob1/InternVL/internvl_chat || { echo "cd failed"; exit 1; }
echo "Activating virtual environment..."
source .venv/bin/activate
export PATH
export PYTHONPATH
export VIRTUAL_ENV
echo "Setup complete. Virtual env activated."

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500 # Or another free port
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

echo "--- DIAGNOSTICS START ---"
echo "Hostname: $(hostname)"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_STEP_GPUS: $SLURM_STEP_GPUS"
echo "CUDA_VISIBLE_DEVICES (before python): $CUDA_VISIBLE_DEVICES"

echo "Running nvidia-smi..."
nvidia-smi | cat

echo "Checking PyTorch CUDA status..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"

echo "Checking environment variables again..."
printenv | grep CUDA
echo "--- DIAGNOSTICS END ---"

# Explicitly set CUDA_VISIBLE_DEVICES based on the task's local ID on the node
echo "Explicitly set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Temporarily comment out the main command for debugging
srun python tools/reasoning_data_pipeline/MARVEL_visualprm_data_pieline.py
echo "Script finished (srun commented out)."