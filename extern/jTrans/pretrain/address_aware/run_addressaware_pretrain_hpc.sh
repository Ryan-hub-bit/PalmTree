#!/bin/bash

# Address-Aware jTrans Pretraining Script (HPC - Improved)
# Uses hierarchical address embeddings with instruction-level masking,
# mixed precision, validation split, and loss weighting.
#
# Usage: ./run_addressaware_pretrain_hpc.sh [data_ratio] [num_epochs]
# Example: ./run_addressaware_pretrain_hpc.sh 0.1 5    # 10% data, 5 epochs
#          ./run_addressaware_pretrain_hpc.sh 1.0 20   # Full data, 20 epochs

# Data ratio (default: 1.0 = 100% of data)
DATA_RATIO="${1:-1.0}"
NUM_EPOCHS="${2:-20}"

# Use all available GPUs (default: 0,1,2,3 for 4 GPUs)
# SLURM will set CUDA_VISIBLE_DEVICES automatically, but if running locally, set it here
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

echo "=========================================="
echo "Starting Address-Aware Pretraining (HPC Improved)"
echo "Data ratio: $DATA_RATIO"
echo "Epochs: $NUM_EPOCHS"
echo "Masking strategy: mixed (token + instruction)"
echo "Mixed precision: enabled"
echo "=========================================="

# Debug: Check vocab size before training
echo "Checking vocabulary..."
python3 <<'VOCABCHECK'
import pickle
vocab = pickle.load(open('./vocab_addr.pkl', 'rb'))
print(f"Vocabulary file: ./vocab_addr.pkl")
print(f"Vocabulary size: {len(vocab)}")
print(f"Max token ID: {max(vocab.stoi.values())}")
print(f"Special tokens: <pad>={vocab.stoi.get('<pad>')}, <unk>={vocab.stoi.get('<unk>')}, <mask>={vocab.stoi.get('<mask>')}")
VOCABCHECK

python3 train_addressaware.py \
  --train_path /work/kliu14/jtransdata/addr_pretrain.txt \
  --vocab_path ./vocab_addr.pkl \
  --output_dir /work/kliu14/jtransoutput/addressaware_pretrain \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --num_epochs "$NUM_EPOCHS" \
  --warmup_ratio 0.06 \
  --max_len 512 \
  --token_mask_prob 0.15 \
  --instruction_mask_prob 0.15 \
  --masking_strategy mixed \
  --jtp_weight 1.0 \
  --val_split 0.05 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --num_attention_heads 12 \
  --save_every 1 \
  --num_workers 12 \
  --data_ratio "$DATA_RATIO"
