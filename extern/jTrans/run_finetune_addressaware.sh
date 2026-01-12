#!/bin/bash
# Fine-tune address-aware model on function similarity task

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

python finetune.py \
    --data_type json \
    --func_blocks /work/kliu14/jtransdata/func_blocks_addr.json \
    --ground_truth /work/kliu14/jtransdata/ground_truth_addr.json \
    --tokenizer /home/kliu14/AAE/extern/jTrans/pretrain/address_aware \
    --model_path /work/kliu14/jtransoutput/addressaware_pretrain/checkpoint_epoch_10 \
    --output_path /work/kliu14/jtransoutput/addressaware_finetune \
    --model_type addressaware \
    --batch_size 32 \
    --eval_batch_size 64 \
    --lr 2e-5 \
    --epoch 5 \
    --weight_decay 0.01 \
    --freeze_cnt 10
