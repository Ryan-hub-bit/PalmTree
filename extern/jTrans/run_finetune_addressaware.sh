#!/bin/bash
# Fine-tune address-aware model on function similarity task

# Set GPUs (use GPU 0 and 1 for multi-GPU tra ining)
export CUDA_VISIBLE_DEVICES=0,1

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

python finetune.py \
    --model_type addressaware \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_addr.json \
    --ground_truth /data/kun/jtransdata/ground_truth_addr.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_15 \
    --output_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune \
    --batch_size 16 \
    --eval_batch_size 32 \
    --lr 1e-5 \
    --epoch 20 \
    --weight_decay 0.01 \
    --data_ratio 1
