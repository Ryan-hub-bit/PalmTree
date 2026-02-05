#!/bin/bash
#SBATCH --job-name=addr_no_binpos
#SBATCH --output=/work/kliu14/jtransdata/log/%x_%j.out
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu4
#SBATCH --account=loni_insembed01

# SLURM: No binary_pos (MLM + JTP, only function_pos + bb_pos)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc 
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/jtrans

# Run no binary_pos experiment
./run_no_binary_pos_hpc.sh

echo "Done!"
