#!/bin/bash
# Test fine-tuning process with small data subset to verify everything works

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Test parameters - small scale for quick verification
echo "============================================"
echo "Testing Fine-tune Process"
echo "============================================"
echo "Model type: addressaware"
echo "Epochs: 2 (for quick testing)"
echo "Batch size: 8 (small for faster iteration)"
echo "Data: Using first 10% of data"
echo "============================================"

python finetune.py \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_addr.json \
    --ground_truth /data/kun/jtransdata/ground_truth_addr.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10 \
    --output_path /home/kun/Document/AAE/output/jtrans/test_finetune \
    --model_type addressaware \
    --batch_size 8 \
    --eval_batch_size 16 \
    --lr 2e-5 \
    --epoch 2 \
    --weight_decay 0.01 \
    --freeze_cnt 10 \
    --data_ratio 0.1

echo ""
echo "============================================"
echo "Test complete! Check the output above for any errors."
echo "If successful, you should see:"
echo "  - Model loaded successfully"
echo "  - Training progress bars for 2 epochs"
echo "  - Model saved to: /home/kun/Document/AAE/output/jtrans/test_finetune/"
echo "============================================"
