#!/bin/bash
# Fine-tune address-aware model on function similarity task

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

python finetune.py \
    --model_type addressaware \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_addr.json \
    --ground_truth /data/kun/jtransdata/ground_truth_addr.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10 \
    --output_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune \
    --batch_size 32 \
    --eval_batch_size 64 \
    --lr 1e-5 \
    --epoch 1 \
    --weight_decay 0.01 \
    --freeze_cnt 10 \
    --data_ratio 0.001
