#!/bin/bash
# Fine-tune baseline model on function similarity task

python3 finetune.py \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --ground_truth /data/kun/jtransdata/ground_truth_baseline.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/jtrans_tokenizer \
    --model_path /home/kun/Document/AAE/output/jtrans/baseline_pretrain/best_model \
    --output_path /home/kun/Document/AAE/output/jtrans/baseline_finetune \
    --batch_size 32 \
    --eval_batch_size 64 \
    --lr 1e-5 \
    --epoch 10 \
    --weight_decay 0.01 \
    --freeze_cnt 10
