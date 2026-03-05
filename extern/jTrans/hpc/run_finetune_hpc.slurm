#!/bin/bash
#SBATCH --job-name=aae_finetune_v2
#SBATCH --output=/work/kliu14/jtransdata/log/%x_%j.out
#SBATCH --error=/work/kliu14/jtransdata/log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu4
#SBATCH --account=loni_insembed01
# ============================================================================
# AAE Address-Aware Finetuning v2 on HPC (SLURM)
#
# Usage:
#   sbatch run_finetune_hpc.slurm
#
# Key changes in v2:
#   - freeze_cnt 4 (8/12 layers trainable, was 10 frozen)
#   - Cosine LR schedule with warmup
#   - Cross-accumulation InfoNCE (128 effective negatives)
#   - Mean pooling (replaces CLS token)
# ============================================================================

set -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node:   $(hostname)"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Date:   $(date)"
echo "============================================"

# GPU diagnostics
nvidia-smi

# --- Paths (match HPC layout) ---
CODE_DIR="/home/kliu14/AAE/extern/jTrans"
DATA_DIR="/work/kliu14/jtransdata"
TOKENIZER_DIR="$CODE_DIR/pretrain/address_aware"
CHECKPOINT_DIR="/work/kliu14/jtransdata/output/addressaware_pretrain/checkpoint_epoch_9"
OUTPUT_DIR="/work/kliu14/jtransdata/output/addressaware_finetune_v2"

# Create output & log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p /work/kliu14/jtransdata/log

# --- Environment ---
source ~/.bashrc
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/jtrans

# Use all available GPUs from SLURM
# Don't override CUDA_VISIBLE_DEVICES - let SLURM manage it
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=42

# --- Run ---
cd "$CODE_DIR"

echo ""
echo "=== Starting finetuning v2 ==="
echo "Data:       $DATA_DIR"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output:     $OUTPUT_DIR"
echo ""

python finetune.py \
  --model_type addressaware \
  --data_type json \
  --func_blocks "$DATA_DIR/func_blocks_addr.json" \
  --ground_truth "$DATA_DIR/ground_truth_addr.json" \
  --tokenizer "$TOKENIZER_DIR" \
  --model_path "$CHECKPOINT_DIR" \
  --output_path "$OUTPUT_DIR" \
  --batch_size 64 \
  --eval_batch_size 64 \
  --lr 2e-5 \
  --epoch 10 \
  --weight_decay 0.01 \
  --warmup 500 \
  --loss_type infonce \
  --temperature 0.07 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --data_ratio 1.0 \
  --use_projection \
  --embedding_dim 512 \
  --target_opt O3 \
  --freeze_cnt 4 \
  --pooling_type mean

echo ""
echo "=== Finetuning complete ==="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Date: $(date)"
