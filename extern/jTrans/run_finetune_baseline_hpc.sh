#!/bin/bash

# Baseline jTrans Fine-tuning Script for HPC
# Fine-tune baseline model on function similarity task
# Optimized for Recall@1 performance
#
# Usage: ./run_finetune_baseline_hpc.sh [data_ratio]
# Example: ./run_finetune_baseline_hpc.sh 0.00001  # Use tiny subset for testing
#          ./run_finetune_baseline_hpc.sh 1.0      # Use 100% of data (default)

# Data ratio (default: 0.00001 for quick testing)
DATA_RATIO="${1:-0.00001}"

# Use all available GPUs (default: 0,1,2,3 for 4 GPUs)
# SLURM will set CUDA_VISIBLE_DEVICES automatically, but if running locally, set it here
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

echo "=========================================="
echo "Starting Baseline Fine-tuning"
echo "Data ratio: $DATA_RATIO ($(echo "$DATA_RATIO * 100" | bc)% of training data)"
echo "=========================================="

python finetune.py \
    --data_type json \
    --func_blocks /work/kliu14/jtransdata/baseline/func_blocks_baseline.json \
    --ground_truth /work/kliu14/jtransdata/baseline/ground_truth_baseline.json \
    --tokenizer /work/kliu14/jtransdata/baseline_pretrain_tokenizer \
    --model_path /work/kliu14/jtransoutput/baseline_pretrain/checkpoint_epoch_10 \
    --output_path /work/kliu14/jtransoutput/baseline_finetune \
    --batch_size 16 \
    --eval_batch_size 64 \
    --lr 2e-5 \
    --epoch 5 \
    --weight_decay 0.01 \
    --warmup 500 \
    --triplet_margin 0.5 \
    --max_grad_norm 1.0 \
    --freeze_cnt 10 \
    --data_ratio "$DATA_RATIO"

echo "Done!"
