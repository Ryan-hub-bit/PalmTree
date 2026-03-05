#!/bin/bash
# ============================================================================
# Prepare files for HPC transfer
# Run this on the lab server to create a tar archive of everything needed
# ============================================================================

set -e

STAGING_DIR="/tmp/aae_hpc_transfer"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

echo "=== Staging files for HPC transfer ==="

# 1. Code files
echo "[1/5] Copying code files..."
mkdir -p "$STAGING_DIR/code/pretrain/address_aware"
cp /home/kun/Document/AAE/extern/jTrans/finetune.py "$STAGING_DIR/code/"
cp /home/kun/Document/AAE/extern/jTrans/data_json.py "$STAGING_DIR/code/"
cp /home/kun/Document/AAE/extern/jTrans/data.py "$STAGING_DIR/code/"
cp /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/address_embedding.py "$STAGING_DIR/code/pretrain/address_aware/"

# 2. Tokenizer files (vocab.txt + tokenizer_config.json)
echo "[2/5] Copying tokenizer files..."
mkdir -p "$STAGING_DIR/tokenizer"
cp /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt "$STAGING_DIR/tokenizer/"
cp /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/tokenizer_config.json "$STAGING_DIR/tokenizer/"

# 3. Pretrained checkpoint (only pytorch_model.bin + config.json needed)
echo "[3/5] Copying pretrained checkpoint..."
mkdir -p "$STAGING_DIR/checkpoint"
cp /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_9/pytorch_model.bin "$STAGING_DIR/checkpoint/"
cp /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_9/config.json "$STAGING_DIR/checkpoint/"

# 4. SLURM job script + run script
echo "[4/5] Copying HPC scripts..."
cp /home/kun/Document/AAE/extern/jTrans/hpc/run_finetune_hpc.slurm "$STAGING_DIR/code/"
cp /home/kun/Document/AAE/extern/jTrans/hpc/setup_hpc_env.sh "$STAGING_DIR/code/"

# 5. Training data (large — these are the bulk of the transfer)
echo "[5/5] Copying training data (~13GB + 82MB)..."
mkdir -p "$STAGING_DIR/data"
cp /data/kun/jtrans/addressaware/func_blocks_addr.json "$STAGING_DIR/data/"
cp /data/kun/jtrans/addressaware/ground_truth_addr.json "$STAGING_DIR/data/"

# Also copy eval data if available
if [ -d "/data/kun/jtrans/addressaware/eval" ]; then
    echo "  Copying eval data..."
    cp -r /data/kun/jtrans/addressaware/eval "$STAGING_DIR/data/eval"
fi

echo ""
echo "=== Staging complete ==="
echo ""
du -sh "$STAGING_DIR"/*
echo ""
echo "Total:"
du -sh "$STAGING_DIR"
echo ""

# Create tar archive
echo "=== Creating tar archive ==="
TAR_FILE="/tmp/aae_hpc_transfer.tar.gz"
cd /tmp
tar czf "$TAR_FILE" aae_hpc_transfer/
echo ""
ls -lh "$TAR_FILE"
echo ""
echo "=== Done! ==="
echo ""
echo "Transfer to HPC with:"
echo "  scp $TAR_FILE <your-hpc>:~/"
echo ""
echo "Then on HPC:"
echo "  tar xzf ~/aae_hpc_transfer.tar.gz"
echo "  cd aae_hpc_transfer/code"
echo "  bash setup_hpc_env.sh"
echo "  sbatch run_finetune_hpc.slurm"
