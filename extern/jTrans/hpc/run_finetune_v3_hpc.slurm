#!/bin/bash
#SBATCH --job-name=aae_finetune_v3
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
# AAE Address-Aware Finetuning v3 on HPC (SLURM)
#
# Usage:
#   sbatch run_finetune_v3_hpc.slurm
#
# Key changes in v3 (over v2):
#   - Deduped training data: cross-binary duplicate functions removed
#     (108539 pairs removed → 287000 remaining, removes compiler stubs like
#      _init, register_tm_clones, frame_dummy, __do_global_dtors_aux, etc.)
#   - Same training recipe as v2 otherwise
#
# Prerequisite:
#   Run dedup_cross_binary.py on HPC first:
#     python dedup_cross_binary.py --mode train \
#       --ground_truth /work/kliu14/jtransdata/ground_truth_addr.json \
#       --output_dir /work/kliu14/jtransdata/dedup
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
OUTPUT_DIR="/work/kliu14/jtransdata/output/addressaware_finetune_v3"

# Deduped ground truth (cross-binary duplicates removed)
GROUND_TRUTH_DEDUP="$DATA_DIR/dedup/ground_truth_addr_dedup.json"

# Create output & log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p /work/kliu14/jtransdata/log

# --- Verify deduped data exists ---
if [ ! -f "$GROUND_TRUTH_DEDUP" ]; then
    echo "ERROR: Deduped ground truth not found: $GROUND_TRUTH_DEDUP"
    echo "Run dedup_cross_binary.py first:"
    echo "  python dedup_cross_binary.py --mode train \\"
    echo "    --ground_truth $DATA_DIR/ground_truth_addr.json \\"
    echo "    --output_dir $DATA_DIR/dedup"
    exit 1
fi

# --- Environment ---
source ~/.bashrc
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/kliu14/.conda/envs/jtrans

# Use all available GPUs from SLURM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=42

# --- Run ---
cd "$CODE_DIR"

echo ""
echo "=== Starting finetuning v3 (deduped) ==="
echo "Data:        $DATA_DIR"
echo "Ground Truth: $GROUND_TRUTH_DEDUP (deduped)"
echo "Checkpoint:  $CHECKPOINT_DIR"
echo "Output:      $OUTPUT_DIR"
echo ""

python finetune.py \
  --model_type addressaware \
  --data_type json \
  --func_blocks "$DATA_DIR/func_blocks_addr.json" \
  --ground_truth "$GROUND_TRUTH_DEDUP" \
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
echo "=== Finetuning v3 complete ==="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Date: $(date)"
