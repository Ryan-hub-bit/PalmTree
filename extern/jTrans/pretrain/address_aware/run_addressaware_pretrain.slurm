#!/bin/bash
#SBATCH --job-name=addr_pretrain
#SBATCH --output=/work/kliu14/jtransdata/log/%x_%j.out
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu4
#SBATCH --account=loni_insembed01

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
source ~/conda/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/jtrans

# Your requested command:
./run_addressaware_pretrain.sh

echo "Done!"
