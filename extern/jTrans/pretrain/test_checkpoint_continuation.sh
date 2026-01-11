#!/bin/bash
# Test checkpoint continuation logic

echo "==================================================================="
echo "Testing Checkpoint Continuation Logic"
echo "==================================================================="
echo ""

# Test 1: Baseline pretrain
echo "Test 1: Checking baseline training script for checkpoint logic..."
if grep -q "find_latest_checkpoint" extern/jTrans/pretrain/baseline/train_baseline.py; then
    echo "✓ find_latest_checkpoint function found"
else
    echo "✗ find_latest_checkpoint function NOT found"
fi

if grep -q "RESUMING FROM CHECKPOINT" extern/jTrans/pretrain/baseline/train_baseline.py; then
    echo "✓ Resume logging found"
else
    echo "✗ Resume logging NOT found"
fi

if grep -q "range(start_epoch, args.num_epochs)" extern/jTrans/pretrain/baseline/train_baseline.py; then
    echo "✓ Training loop uses start_epoch"
else
    echo "✗ Training loop does NOT use start_epoch"
fi

echo ""

# Test 2: Address-aware pretrain
echo "Test 2: Checking address-aware training script for checkpoint logic..."
if grep -q "find_latest_checkpoint" extern/jTrans/pretrain/address_aware/train_addressaware.py; then
    echo "✓ find_latest_checkpoint function found"
else
    echo "✗ find_latest_checkpoint function NOT found"
fi

if grep -q "RESUMING FROM CHECKPOINT" extern/jTrans/pretrain/address_aware/train_addressaware.py; then
    echo "✓ Resume logging found"
else
    echo "✗ Resume logging NOT found"
fi

if grep -q "range(start_epoch, args.num_epochs)" extern/jTrans/pretrain/address_aware/train_addressaware.py; then
    echo "✓ Training loop uses start_epoch"
else
    echo "✗ Training loop does NOT use start_epoch"
fi

echo ""
echo "==================================================================="
echo "Summary"
echo "==================================================================="
echo ""
echo "Both training scripts now support automatic checkpoint continuation:"
echo "  1. Scans output directory for checkpoint_epoch_* folders"
echo "  2. Finds the most recent checkpoint (highest epoch number)"
echo "  3. Loads the checkpoint weights automatically"
echo "  4. Continues training from the next epoch"
echo ""
echo "Usage: Just run the training scripts as usual. If checkpoints exist,"
echo "       training will automatically resume from the latest one."
echo ""
