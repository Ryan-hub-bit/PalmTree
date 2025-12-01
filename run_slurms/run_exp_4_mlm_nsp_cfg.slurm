#!/bin/bash
#SBATCH --job-name=exp4_mlm_nsp
#SBATCH --output=/work/kliu14/haapr/log/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu2
#SBATCH --account=loni_progsyn01

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate /work/kliu14/conda/haapr

# Ensure we are in the repo root
cd /home/kliu14/PalmTree

# Run the script
bash run_scripts/run_exp_4_mlm_nsp_cfg.sh
