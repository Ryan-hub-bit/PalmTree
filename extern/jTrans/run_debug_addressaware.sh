#!/bin/bash
# Debug address-aware evaluation issues
# Run comprehensive diagnostics on model and embeddings

export CUDA_VISIBLE_DEVICES=0

source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Configuration - update these paths
MODEL_PATH="/home/kun/Document/AAE/output/jtrans/addressaware_finetune_sincos/finetune_epoch_2"
VOCAB_PATH="/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware"
FUNC_BLOCKS="/data/kun/jtrans/addressaware/eval/func_blocks_addr.json"
GROUND_TRUTH="/data/kun/jtrans/addressaware/eval/ground_truth_addr.json"

echo "=========================================="
echo "Running Address-Aware Debug Script"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Vocab: $VOCAB_PATH"
echo ""

python debug_addressaware_evaluation.py \
  --model_path "$MODEL_PATH" \
  --vocab_path "$VOCAB_PATH" \
  --func_blocks "$FUNC_BLOCKS" \
  --ground_truth "$GROUND_TRUTH" \
  --device cuda:0

echo ""
echo "=========================================="
echo "Debug Complete!"
echo "=========================================="
echo ""
echo "Analysis Steps:"
echo "1. Check STEP 1: Are address tokens properly tokenized?"
echo "2. Check STEP 2: Is vocab_stoi loaded in the model?"
echo "3. Check STEP 3: Are embeddings diverse (std > 0.01)?"
echo "4. Check STEP 4: Do ground truth pairs have high similarity (>0.5)?"
echo "5. Check STEP 5: Do random pairs have low similarity (<0.5)?"
echo ""
echo "If ground truth pairs have LOW similarity:"
echo "  → Model hasn't learned to recognize similar functions"
echo "  → Check if training converged, loss went down"
echo "  → May need more training epochs or different hyperparameters"
echo ""
echo "If random pairs have HIGH similarity:"
echo "  → Embedding collapse - all functions mapped to similar vectors"
echo "  → Check model architecture, learning rate, normalization"
