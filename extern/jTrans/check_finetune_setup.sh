#!/bin/bash
# Pre-flight checks before running finetune

echo "============================================"
echo "Fine-tune Pre-flight Verification Checklist"
echo "============================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Conda environment
echo -n "1. Checking conda environment 'jtrans'... "
if conda env list | grep -q "jtrans"; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "   Run: conda create -n jtrans python=3.8"
fi

# Check 2: Pretrained model checkpoint
echo -n "2. Checking pretrained checkpoint... "
if [ -f "/home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10/pytorch_model.bin" ]; then
    echo -e "${GREEN}✓ Found${NC}"
    ls -lh /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10/pytorch_model.bin
else
    echo -e "${RED}✗ Not found${NC}"
    echo "   Expected: /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10/pytorch_model.bin"
fi

# Check 3: Tokenizer/vocab
echo -n "3. Checking tokenizer files... "
if [ -d "/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware" ]; then
    echo -e "${GREEN}✓ Found${NC}"
    ls /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/
else
    echo -e "${RED}✗ Not found${NC}"
    echo "   Expected: /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/"
fi

# Check 4: Training data
echo -n "4. Checking training data (func_blocks_addr.json)... "
if [ -f "/data/kun/jtransdata/func_blocks_addr.json" ]; then
    SIZE=$(du -h /data/kun/jtransdata/func_blocks_addr.json | cut -f1)
    echo -e "${GREEN}✓ Found ($SIZE)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "   Expected: /data/kun/jtransdata/func_blocks_addr.json"
fi

# Check 5: Ground truth data
echo -n "5. Checking ground truth (ground_truth_addr.json)... "
if [ -f "/data/kun/jtransdata/ground_truth_addr.json" ]; then
    SIZE=$(du -h /data/kun/jtransdata/ground_truth_addr.json | cut -f1)
    echo -e "${GREEN}✓ Found ($SIZE)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "   Expected: /data/kun/jtransdata/ground_truth_addr.json"
fi

# Check 6: finetune.py exists
echo -n "6. Checking finetune.py script... "
if [ -f "/home/kun/Document/AAE/extern/jTrans/finetune.py" ]; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check 7: Output directory
echo -n "7. Checking output directory... "
if [ -d "/home/kun/Document/AAE/output/jtrans" ]; then
    echo -e "${GREEN}✓ Exists${NC}"
else
    echo -e "${YELLOW}! Creating...${NC}"
    mkdir -p /home/kun/Document/AAE/output/jtrans
fi

# Check 8: Python packages (if jtrans env is active)
echo ""
echo "8. Python package check (requires jtrans env active):"
if [ "$CONDA_DEFAULT_ENV" = "jtrans" ]; then
    echo -n "   - torch: "
    python -c "import torch; print('✓ ' + torch.__version__)" 2>/dev/null || echo -e "${RED}✗ Not installed${NC}"
    echo -n "   - transformers: "
    python -c "import transformers; print('✓ ' + transformers.__version__)" 2>/dev/null || echo -e "${RED}✗ Not installed${NC}"
    echo -n "   - tqdm: "
    python -c "import tqdm; print('✓ ' + tqdm.__version__)" 2>/dev/null || echo -e "${RED}✗ Not installed${NC}"
else
    echo -e "   ${YELLOW}Skipped (activate jtrans env first)${NC}"
fi

# Check 9: GPU availability
echo ""
echo -n "9. Checking GPU availability... "
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        echo -e "${GREEN}✓ Found $GPU_COUNT GPU(s)${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | nl
    else
        echo -e "${YELLOW}! No GPUs detected${NC}"
    fi
else
    echo -e "${YELLOW}! nvidia-smi not found${NC}"
fi

# Check 10: Disk space
echo ""
echo "10. Disk space check:"
df -h /home/kun/Document/AAE/output | tail -1 | awk '{print "    Available: "$4" / "$2" ("$5" used)"}'

echo ""
echo "============================================"
echo "Verification complete!"
echo ""
echo "To run test finetune:"
echo "  cd /home/kun/Document/AAE/extern/jTrans"
echo "  bash run_test_finetune.sh"
echo "============================================"
