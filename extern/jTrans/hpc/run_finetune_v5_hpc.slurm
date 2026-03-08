#!/bin/bash
#SBATCH --job-name=aae_finetune_v5
#SBATCH --output=/work/kliu14/jtransdata/log/%x_%j.out
#SBATCH --error=/work/kliu14/jtransdata/log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu4
#SBATCH --account=loni_insembed01
# ============================================================================
# AAE Address-Aware Finetuning v5 on HPC (SLURM)
#
# Usage:
#   sbatch run_finetune_v5_hpc.slurm
#
# Key changes in v5 (over v4):
#   - Hard negative mining: top-256 hardest negatives from 65K queue
#     Only the 256 most confusing negatives contribute to loss
#   - ArcFace angular margin (0.15): positive must beat negatives by margin
#   - Symmetric InfoNCE: loss computed both directions (anchor->pos + pos->anchor)
#   - Attention pooling: learned token weighting instead of uniform mean
#   - Unfreeze ALL layers (freeze_cnt=0): full model adaptation
#   - 50 epochs (was 30): longer training to converge
#   - lr=3e-5 (was 2e-5): slightly higher for full unfreezing
#   - save_every=10 (save at 10,20,30,40,50)
#
# Target: 80% R@1 on 10K pool (requires ~99.998% pairwise accuracy)
#
# Why these changes:
#   v4 MoCo queue (65K negatives) helps but random negatives are still
#   too easy. Hard negative mining forces the model to discriminate the
#   most confusing pairs. Angular margin prevents lazy solutions where
#   positive barely beats negatives. Attention pooling learns to focus
#   on the most discriminative tokens. Unfreezing all layers maximizes
#   adaptation capacity.
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
OUTPUT_DIR="/work/kliu14/jtransdata/output/addressaware_finetune_v5"

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
echo "=== Starting finetuning v5 (hard neg + margin + symmetric + attention) ==="
echo "Data:       $DATA_DIR"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output:     $OUTPUT_DIR"
echo ""

DEDUP_GT="$DATA_DIR/dedup/ground_truth_addr_dedup.json"
if [ ! -f "$DEDUP_GT" ]; then
  echo "ERROR: Dedup ground truth not found: $DEDUP_GT"
  echo "Copy from local: scp /data/kun/jtrans/addressaware/dedup/ground_truth_addr_dedup.json kliu14@qbd.loni.org:$DEDUP_GT"
  exit 1
fi

python finetune.py \
  --model_type addressaware \
  --data_type json \
  --func_blocks "$DATA_DIR/func_blocks_addr.json" \
  --ground_truth "$DEDUP_GT" \
  --tokenizer "$TOKENIZER_DIR" \
  --model_path "$CHECKPOINT_DIR" \
  --output_path "$OUTPUT_DIR" \
  --batch_size 64 \
  --eval_batch_size 64 \
  --lr 3e-5 \
  --epoch 50 \
  --weight_decay 0.01 \
  --warmup 1000 \
  --loss_type infonce \
  --temperature 0.05 \
  --queue_size 65536 \
  --hard_neg_k 256 \
  --margin 0.15 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --data_ratio 1.0 \
  --use_projection \
  --embedding_dim 512 \
  --freeze_cnt 0 \
  --pooling_type attention \
  --save_every 10

echo ""
echo "=== Finetuning v5 complete ==="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Date: $(date)"
