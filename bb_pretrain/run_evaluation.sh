#!/bin/bash
# Run instruction-level evaluation comparing Address-Aware PalmTree vs Vanilla PalmTree

echo "Starting instruction-level evaluation..."
echo "========================================"

cd /home/louie/PalmTree/bb_pretrain

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate palmtree

# Run evaluation on test set
python evaluate_instruction_level.py \
    --bb_pairs_file /home/louie/PalmTree/bb_pretrain/data/train_subset_100k.txt \
    --vocab_file /home/louie/PalmTree/pre-trained_model/palmtree/vocab \
    --addr_model_path /home/louie/PalmTree/bb_pretrain/output/best_model.pt

echo ""
echo "========================================"
echo "✅ Evaluation complete!"
echo "Results saved to: /home/louie/PalmTree/bb_pretrain/output/instruction_level_comparison.json"
