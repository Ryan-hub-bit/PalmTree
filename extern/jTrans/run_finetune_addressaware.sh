#!/bin/bash
# Fine-tune address-aware model on function similarity task

python3 finetune.py \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_addr.json \
    --ground_truth /data/kun/jtransdata/ground_truth_addr.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/best_model \
    --output_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune \
    --batch_size 32 \
    --eval_batch_size 64 \
    --lr 2e-5 \
    --epoch 10 \
    --weight_decay 0.01 \
    --freeze_cnt 10
