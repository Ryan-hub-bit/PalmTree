#!/bin/bash

# HPC: No binary_pos (MLM + JTP, only function_pos + bb_pos)
# Usage: ./run_no_binary_pos_hpc.sh [data_ratio]
# Example: ./run_no_binary_pos_hpc.sh 1.0

# Data ratio (default: 1.0 = 100% of data)
DATA_RATIO="${1:-1.0}"

# Use all available GPUs (SLURM will set this automatically)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

echo "=========================================="
echo "HPC: No binary_pos Pretraining (func + bb only)"
echo "Data ratio: $DATA_RATIO"
echo "=========================================="

python3 train_addressaware.py \
  --train_path /work/kliu14/jtransdata/addr_pretrain.txt \
  --vocab_path ./vocab_addr.pkl \
  --output_dir /work/kliu14/jtransoutput/addressaware_no_binary_pos \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --warmup_steps 10000 \
  --max_len 512 \
  --token_mask_prob 0.15 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --num_attention_heads 12 \
  --save_every 1 \
  --num_workers 12 \
  --data_ratio "$DATA_RATIO" \
  --no_binary_pos
