#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate jtrans

# Run evaluation
python evaluate_checkpoint.py \
  --checkpoint_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_10 \
  --data_path /home/kun/Document/AAE/dstask/funcsim/func_blocks_addr.json \
  --ground_truth_path /home/kun/Document/AAE/dstask/funcsim/ground_truth_addr.json \
  --vocab_dir /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
  --model_type addressaware \
  --batch_size 32 \
  --data_ratio 0.1
