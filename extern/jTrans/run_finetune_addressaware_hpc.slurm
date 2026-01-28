#!/bin/bash
#SBATCH --job-name=finetune_addraware
#SBATCH --output=/work/kliu14/jtrans/log/finetune_addressaware_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --account=loni_insembed01

#===========================================
# Address-Aware Model Fine-tuning (SLURM)
#===========================================

# Set working directory
cd /home/kliu14/AAE/extern/jTrans

# Create log directory
mkdir -p /work/kliu14/jtransdata/log

source ~/.bashrc 
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/jtrans


# Print job info
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Print GPU info
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Run the training script
bash run_finetune_addressaware_hpc.sh

# Print completion info
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
