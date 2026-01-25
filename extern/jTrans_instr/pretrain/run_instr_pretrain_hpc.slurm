#!/bin/bash
#SBATCH --job-name=instr_pretrain
#SBATCH --output=/work/kliu14/jtrans/log/%x_%j.out
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu4
#SBATCH --account=loni_insembed01

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc 
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/jtrans

# Your requested command:
# Change the argument to control data ratio:
# ./run_instr_pretrain_hpc.sh 0.01   # 1% of data
# ./run_instr_pretrain_hpc.sh 0.1    # 10% of data
# ./run_instr_pretrain_hpc.sh 1.0    # 100% of data (full training)
./run_instr_pretrain_hpc.sh

echo "Done!"
