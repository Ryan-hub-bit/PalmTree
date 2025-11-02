# Address-Aware PalmTree Pretraining

This module extends PalmTree with address-aware embeddings for basic block-level pretraining.

## Overview

The goal is to improve PalmTree's instruction embeddings by training on basic block (BB) pairs with address information. This helps the model understand:
- Control flow relationships (branches, calls, returns)
- Address spatial locality
- Code vs data distinction
- BB boundaries and sizes

## Architecture

```
Input: BB Pair with Addresses
  ↓
Tokenization + Address Extraction
  ↓
PalmTree BERT Embeddings
  +
Sin/Cos Address Encodings
  ↓
Multi-Task Heads:
  - Next BB Prediction
  - Address Type Classification
  - Edge Type Classification
```

## Directory Structure

```
bb_pretrain/
├── config.py              # Configuration
├── data_loader.py         # Dataset and dataloader
├── train.py               # Training script
├── models/
│   └── addr_palmtree.py   # Address-aware model
├── data/                  # BB pairs data
└── output/                # Checkpoints and logs
```

## Setup

### 1. Generate BB Pairs

First, generate basic block pairs from binaries using the extraction script:

```bash
cd /home/louie/PalmTree
python scripts/bb_flow.py <binary_path> <output_file>

# Example:
python scripts/bb_flow.py ~/smallbinary/xinit xinit_bb_pairs.txt
```

### 2. Install Dependencies

```bash
pip install torch numpy tqdm tensorboard
```

### 3. Train

```bash
cd bb_pretrain
python train.py \
    --bb_pairs_file ../xinit_bb_pairs.txt \
    --vocab_file ../pre-trained_model/vocab.txt
```

## Address Token Format

The model uses special tokens for addresses:

- `<addr_start>`: Start of basic block
- `<addr_end>`: End of basic block
- `<addr_tgt>`: Branch/call target address
- `<addr_code>`: Code address reference
- `<addr_data>`: Data address reference
- `<addr_unknown>`: Unknown target (indirect branch)
- `<seq>`: Instruction separator

Example:
```
<addr_start> push rbp mov rbp rsp call <addr_tgt> <addr_end> 
-> 
<addr_start> lea rdi [rel <addr_data>] cmp rax rdi <addr_end>
```

## Pretraining Tasks

### 1. Next BB Prediction
Given source BB, predict target BB tokens.

**Loss**: Cross-entropy on next BB token prediction

### 2. Address Type Classification
Classify each address token as code/data/start/end/tgt.

**Loss**: Multi-class classification

### 3. Edge Type Classification
Classify control flow edge type (fallthrough/branch/call/return/indirect).

**Loss**: Multi-class classification

## Configuration

Edit `config.py` to adjust:

- Model architecture (hidden size, layers, etc.)
- Training hyperparameters (batch size, learning rate, etc.)
- Task weights for multi-task learning
- Address encoding parameters

## Monitoring

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir output/logs
```

## Output

The trained model is saved to:
- `output/best_model.pt`: Best model checkpoint
- `output/logs/`: TensorBoard logs

## Next Steps

After pretraining, you can:

1. **Fine-tune** on downstream tasks (binary similarity, function detection, etc.)
2. **Extract embeddings** for analysis and visualization
3. **Evaluate** on intrinsic tasks (address prediction accuracy, etc.)

## Example Usage

```python
import torch
from models.addr_palmtree import AddressAwarePalmTree, load_pretrained_palmtree
from data_loader import BBPairDataset

# Load model
palmtree, vocab = load_pretrained_palmtree('path/to/palmtree')
model = AddressAwarePalmTree(palmtree)
checkpoint = torch.load('output/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Load data
dataset = BBPairDataset('xinit_bb_pairs.txt', 'vocab.txt')
sample = dataset[0]

# Get predictions
outputs = model(
    sample['input_ids'].unsqueeze(0),
    sample['attention_mask'].unsqueeze(0),
    sample['address_encodings'].unsqueeze(0)
)
```

## Citation

If you use this code, please cite the original PalmTree paper and acknowledge this extension.
