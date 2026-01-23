#!/bin/bash
# Fine-tune jTrans_instr model on function similarity task

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Set GPU
export CUDA_VISIBLE_DEVICES=1

# Paths (modify these to match your setup)
FUNC_BLOCKS="/data/kun/jtrans_instr/func_blocks_instr.json"
GROUND_TRUTH="/data/kun/jtrans_instr/ground_truth_instr.json"
TOKENIZER="/home/kun/Document/AAE/extern/jTrans_instr/pretrain"
MODEL_PATH="/home/kun/Document/AAE/output/jtrans_instr/pretrain_run1/checkpoint_epoch_10"
OUTPUT_PATH="/home/kun/Document/AAE/output/jtrans_instr/finetune"

# Training hyperparameters
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
LR=1e-5
EPOCHS=5
WEIGHT_DECAY=0.01
FREEZE_CNT=10
DATA_RATIO=1.0  # Use all data (set to 0.001 for quick testing)

echo "=========================================="
echo "jTrans_instr Finetuning"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Epochs: $EPOCHS"
echo "Data ratio: $DATA_RATIO"
echo "=========================================="

# Run finetuning
python finetune_instr.py \
    --func_blocks $FUNC_BLOCKS \
    --ground_truth $GROUND_TRUTH \
    --tokenizer $TOKENIZER \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --lr $LR \
    --epoch $EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --freeze_cnt $FREEZE_CNT \
    --data_ratio $DATA_RATIO \
    --max_instructions 201

echo "Finetuning complete!"
