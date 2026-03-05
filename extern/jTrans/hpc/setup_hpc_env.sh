#!/bin/bash
# ============================================================================
# Setup conda environment on HPC
# Run this once before submitting SLURM jobs
# ============================================================================

set -e

echo "=== Setting up HPC environment for AAE finetuning ==="

# Load anaconda module (adjust for your HPC)
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true

# Create conda environment
echo "[1/3] Creating conda environment 'jtrans'..."
conda create -n jtrans python=3.8 -y

# Activate
echo "[2/3] Installing packages..."
conda activate jtrans

# Install PyTorch (CUDA 11.8 — adjust for your HPC's CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install transformers and other dependencies
pip install transformers==4.30.0 tqdm wandb

echo "[3/3] Verifying installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import BertTokenizer; print('Transformers OK')"

echo ""
echo "=== Setup complete! ==="
echo "Activate with: conda activate jtrans"
