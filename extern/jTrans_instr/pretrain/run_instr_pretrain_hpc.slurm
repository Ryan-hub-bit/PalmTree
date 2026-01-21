#!/bin/bash
#SBATCH --job-name=instr_pretrain
#SBATCH --output=/work/kliu14/jtrans_instr/log/%x_%j.out
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu2
#SBATCH --account=loni_insembed01

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc 
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/jtrans

# Your requested command:
./run_instr_pretrain_hpc.sh

echo "Done!"
