#!/bin/bash
# Fine-tune address-aware model on function similarity task

python3 finetune.py \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks.json \
    --ground_truth /data/kun/jtransdata/ground_truth.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/best_model.pth \
    --batch_size 32 \
    --eval_batch_size 64 \
    --lr 2e-5 \
    --epoch 10 \
    --weight_decay 0.01 \
    --freeze_cnt 10 \
    --load_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune
