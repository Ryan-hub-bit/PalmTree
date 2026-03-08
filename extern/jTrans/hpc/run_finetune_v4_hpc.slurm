#!/bin/bash
#SBATCH --job-name=aae_finetune_v4
#SBATCH --output=/work/kliu14/jtransdata/log/%x_%j.out
#SBATCH --error=/work/kliu14/jtransdata/log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu4
#SBATCH --account=loni_insembed01
# ============================================================================
# AAE Address-Aware Finetuning v4 on HPC (SLURM)
#
# Usage:
#   sbatch run_finetune_v4_hpc.slurm
#
# Key changes in v4 (over v2):
#   - MoCo-style memory queue (65536 negatives) for InfoNCE
#     Each batch sees 64 in-batch + 65536 queue = 65600 negatives
#   - All opt-level pairs (no --target_opt): O0↔O1, O0↔O2, O0↔O3, etc.
#   - 30 epochs (was 10) — loss was still decreasing at epoch 9
#   - freeze_cnt=2 (10/12 layers trainable, was 8/12 in v2)
#   - Temperature 0.05 (was 0.07, sharper discrimination)
#   - Skip negative forward pass (saves ~33% GPU compute)
#   - Warmup 1000 steps (was 500)
#   - save_every 5 (save checkpoints at epochs 5,10,15,20,25,30)
#
# Why these changes:
#   v2 achieved 90% pairwise accuracy but only 1.5% R@1 on 10K pool.
#   To get 50% R@1 on 10K pool, need 99.993% pairwise accuracy.
#   MoCo queue gives 128x more negatives (65K vs 512), forcing sharper
#   discrimination. All opt pairs + more epochs provide more training signal.
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
CHECKPOINT_DIR="/work/kliu14/jtransoutput/addressaware_pretrain/checkpoint_epoch_9"
OUTPUT_DIR="/work/kliu14/jtransdata/output/addressaware_finetune_v4"

# Create output & log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p /work/kliu14/jtransdata/log

# --- Environment ---
source ~/.bashrc
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/jtrans

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=42

# --- Run ---
cd "$CODE_DIR"

echo ""
echo "=== Starting finetuning v4 (MoCo queue + all opt pairs) ==="
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
  --epoch 30 \
  --weight_decay 0.01 \
  --warmup 1000 \
  --loss_type infonce \
  --temperature 0.05 \
  --queue_size 65536 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --data_ratio 1.0 \
  --use_projection \
  --embedding_dim 512 \
  --freeze_cnt 2 \
  --pooling_type mean \
  --save_every 5

echo ""
echo "=== Finetuning v4 complete ==="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Date: $(date)"
