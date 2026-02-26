#!/bin/bash
#SBATCH --job-name=addr_pretrain
#SBATCH --output=/work/kliu14/jtransdata/log/%x_%j.out
#SBATCH --error=/work/kliu14/jtransdata/log/%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu4
#SBATCH --account=loni_insembed01

set -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

# GPU diagnostics
nvidia-smi

# Activate conda environment
source ~/.bashrc 
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/jtrans

# cd to script directory (critical for relative paths like vocab_addr.pkl)
cd /home/kliu14/AAE/extern/jTrans/pretrain/address_aware
echo "Working directory: $(pwd)"

# Pass arguments through: sbatch run_addressaware_pretrain_hpc.slurm [data_ratio] [num_epochs]
# Defaults: data_ratio=1.0, num_epochs=20
DATA_RATIO="${1:-1.0}"
NUM_EPOCHS="${2:-10}"

echo "Data ratio: $DATA_RATIO"
echo "Num epochs: $NUM_EPOCHS"

./run_addressaware_pretrain_hpc.sh "$DATA_RATIO" "$NUM_EPOCHS"

echo "=========================================="
echo "End time: $(date)"
echo "Done!"
