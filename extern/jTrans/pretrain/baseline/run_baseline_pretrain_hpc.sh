#!/bin/bash

# Baseline jTrans Pretraining Script for HPC
# Pure MLM + JTP (Jump-Target Prediction) without address-aware features
# PRETRAINING: Uses single combined file (no train/val/test split)
#
# Usage: ./run_baseline_pretrain_hpc.sh [data_ratio]
# Example: ./run_baseline_pretrain_hpc.sh 0.1  # Use 10% of data
#          ./run_baseline_pretrain_hpc.sh 1.0  # Use 100% of data (default)

# Data ratio (default: 1.0 = 100% of data)
DATA_RATIO="${1:-0.0001}"

# Use all available GPUs (default: 0,1,2,3 for 4 GPUs)
# SLURM will set CUDA_VISIBLE_DEVICES automatically, but if running locally, set it here
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
fi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
# Uncomment for debugging (makes CUDA synchronous, slower)
# export CUDA_LAUNCH_BLOCKING=1

echo "=========================================="
echo "Starting Baseline Pretraining"
echo "Data ratio: $DATA_RATIO ($(echo "$DATA_RATIO * 100" | bc)% of training data)"
echo "=========================================="

# Debug: Check vocab size before training
# Check if tokenizer files exist
if [ ! -f "./vocab.txt" ]; then
    echo "Error: vocab.txt not found in current directory"
    exit 1
fi
if [ ! -f "./tokenizer_config.json" ]; then
    echo "Error: tokenizer_config.json not found"
    exit 1
fi
echo "✓ Tokenizer files found: vocab.txt, tokenizer_config.json, special_tokens_map.json"

python3 train_baseline.py \
  --train_path /work/kliu14/jtransdata/baseline_pretrain.txt \
  --tokenizer_path . \
  --output_dir /work/kliu14/jtransoutput/baseline_pretrain \
  --batch_size 64 \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --warmup_steps 10000 \
  --max_len 512 \
  --mlm_probability 0.15 \
  --jtp_probability 0.20 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --num_attention_heads 12 \
  --save_every 1 \
  --num_workers 16 \
  --data_ratio "$DATA_RATIO"

echo "Done!"
