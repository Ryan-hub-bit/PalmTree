#!/bin/bash
# Fine-tune address-aware model on function similarity task (HPC version)
# Optimized for Recall@1 performance on HPC cluster

#===========================================
# Configuration
#===========================================

# Data paths (HPC storage)
DATA_DIR="/work/kliu14/jtransdata"
FUNC_BLOCKS="$DATA_DIR/func_blocks_addr.json"
GROUND_TRUTH="$DATA_DIR/ground_truth_addr.json"

# Model paths
TOKENIZER="/work/kliu14/AAE/extern/jTrans/pretrain/address_aware"
MODEL_PATH="/work/kliu14/jtransoutput/addressaware_pretrain/checkpoint_epoch_14"
OUTPUT_PATH="/work/kliu14/jtransoutput/addressaware_finetune"

# Training hyperparameters
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
LR=2e-5
EPOCHS=2
WARMUP=500
TRIPLET_MARGIN=0.5
MAX_GRAD_NORM=1.0
WEIGHT_DECAY=0.01
DATA_RATIO=0.00001  # Use full dataset (change to smaller value for testing)

# Optional: Target optimization level (O0, O1, O2, O3)
# Uncomment to train for specific Ox→O3 retrieval task
# TARGET_OPT="O3"

#===========================================
# Setup
#===========================================

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Log file
LOG_FILE="$OUTPUT_PATH/finetune_$(date +%Y%m%d_%H%M%S).log"

#===========================================
# Training
#===========================================

echo "=========================================="
echo "Address-Aware Model Fine-tuning (HPC)"
echo "=========================================="
echo "Model type: addressaware"
echo "Tokenizer: $TOKENIZER"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_PATH"
echo "=========================================="
echo "Hyperparameters:"
echo "  Batch size: $BATCH_SIZE"
echo "  Eval batch size: $EVAL_BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Epochs: $EPOCHS"
echo "  Warmup steps: $WARMUP"
echo "  Triplet margin: $TRIPLET_MARGIN"
echo "  Max grad norm: $MAX_GRAD_NORM"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Data ratio: $DATA_RATIO"
if [ -n "$TARGET_OPT" ]; then
    echo "  Target opt: $TARGET_OPT"
fi
echo "=========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"
echo "=========================================="

# Build command
CMD="python finetune.py \
    --model_type addressaware \
    --data_type json \
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
    --warmup $WARMUP \
    --triplet_margin $TRIPLET_MARGIN \
    --max_grad_norm $MAX_GRAD_NORM \
    --data_ratio $DATA_RATIO"

# Add target_opt if specified
if [ -n "$TARGET_OPT" ]; then
    CMD="$CMD --target_opt $TARGET_OPT"
fi

# Run training
echo "Starting training..."
echo "Command: $CMD"
echo ""

eval $CMD 2>&1 | tee "$LOG_FILE"

#===========================================
# Completion
#===========================================

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Output: $OUTPUT_PATH"
    echo "Log: $LOG_FILE"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed!"
    echo "Check log: $LOG_FILE"
    echo "=========================================="
    exit 1
fi
