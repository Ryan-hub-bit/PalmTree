# jTrans_instr Pretraining

This directory contains the pretraining code for jTrans_instr with instruction-level embeddings.

## Overview

jTrans_instr uses:
- **Standard BERT embeddings** (not position=word trick)
- **Instruction embeddings** to capture instruction-level structure
- **Instruction-level jump addressing** (instr_addr_{i} instead of JUMP_ADDR_X)

## Files

- `model_instr.py` - Model architecture with instruction embeddings
- `dataloader_instr.py` - Dataloader that generates instruction_ids
- `train_instr.py` - Training script for MLM + JTP tasks
- `run_instr_pretrain.sh` - Shell script to run training

## Data Format

Input data should be in text format (one function per line):
```
[CLS] push ebp mov ebp esp sub esp 0x10 call instr_addr_5 ... [SEP]
```

Where:
- `instr_addr_{i}` represents a jump to instruction index i (0-200)
- Each instruction is a sequence of tokens (mnemonic + operands)

## Model Architecture

```
Embedding = Token Embedding + Position Embedding + Instruction Embedding
```

1. **Token Embedding**: Standard word embeddings for vocabulary
2. **Position Embedding**: Token-level positions (0-511)
3. **Instruction Embedding**: Instruction-level indices (0-200)

**Special handling**: `instr_addr_{i}` tokens directly use `instruction_embedding[i]`, creating a semantic link between jump addresses and instruction representations.

## Training Tasks

### 1. Masked Language Modeling (MLM)
- Mask 15% of regular tokens
- Predict original tokens from vocabulary
- Loss: CrossEntropyLoss over vocabulary

### 2. Jump Target Prediction (JTP)
- Mask 20% of instr_addr tokens
- Predict target instruction index (0-200)
- Loss: CrossEntropyLoss over instruction indices

## Usage

### 1. Prepare Data

First, generate the data using the data pipeline:
```bash
cd ../datautils
bash generate_baseline.sh  # Extract from binaries
bash generate_text.sh      # Convert to text
bash build_vocab.sh        # Build vocabulary
```

This creates:
- `/data/kun/jtrans_instr/instr_pretrain.txt` (training data)
- `/home/kun/Document/AAE/extern/jTrans_instr/jtrans_tokenizer/vocab.txt` (vocabulary)

### 2. Configure Training

Edit `run_instr_pretrain.sh` to set:
- GPU: `export CUDA_VISIBLE_DEVICES=0`
- Data paths: `TRAIN_PATH`, `TEST_PATH`, `TOKENIZER_PATH`
- Output directory: `OUTPUT_DIR`
- Architecture: `HIDDEN_SIZE`, `NUM_LAYERS`, `NUM_HEADS`, `MAX_INSTRUCTIONS`
- Hyperparameters: `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`

### 3. Run Training

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/pretrain
bash run_instr_pretrain.sh
```

Or run directly with Python:
```bash
python train_instr.py \
    --train_path /data/kun/jtrans_instr/instr_pretrain.txt \
    --test_path /data/kun/jtrans_instr/instr_test.txt \
    --tokenizer_path /home/kun/Document/AAE/extern/jTrans_instr/jtrans_tokenizer \
    --output_dir ./output_run1 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --max_instructions 201
```

## Output

Training produces:
```
output_dir/
├── checkpoint_epoch_1/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_info.json
├── checkpoint_epoch_2/
│   └── ...
├── best_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_info.json
├── training_history.json
└── train_instr_YYYYMMDD_HHMMSS.log
```

## Hyperparameters

Default configuration (similar to BERT-base):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_size` | 768 | Hidden dimension |
| `num_hidden_layers` | 12 | Number of transformer layers |
| `num_attention_heads` | 12 | Number of attention heads |
| `intermediate_size` | 3072 | FFN intermediate size |
| `max_instructions` | 201 | Max instructions (0-200) |
| `max_len` | 512 | Max sequence length (tokens) |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `warmup_steps` | 10000 | Warmup steps for scheduler |
| `mlm_probability` | 0.15 | MLM masking probability |
| `jtp_probability` | 0.20 | JTP masking probability |

## Differences from Baseline jTrans

| Feature | Baseline jTrans | jTrans_instr |
|---------|----------------|--------------|
| Position embeddings | position=word trick | Standard BERT absolute positions |
| Instruction understanding | Implicit | Explicit instruction_embeddings |
| Jump addresses | JUMP_ADDR_X (token pos) | instr_addr_{i} (instruction idx) |
| Jump-instruction link | No direct link | instr_addr_{i} uses instruction_embedding[i] |
| Token type embeddings | Has token_type | Removed (not needed) |
| JTP task | Predict token position | Predict instruction index |

## Model Loading

To load a pretrained model:

```python
from model_instr import create_instr_model
from transformers import BertTokenizer

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("path/to/jtrans_tokenizer")

# Create model
model = create_instr_model(
    vocab_size=len(tokenizer),
    max_instructions=201,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12
)

# Set instr_addr token IDs for special handling
model.set_instr_addr_token_ids(tokenizer)

# Load weights
model.bert.load_state_dict(torch.load("path/to/checkpoint/pytorch_model.bin"))
```

## Monitoring Training

Training logs include:
- Total loss (MLM + JTP)
- MLM loss and accuracy
- JTP loss and accuracy
- Learning rate

Check logs:
```bash
tail -f output_dir/train_instr_*.log
```

Check tensorboard (if configured):
```bash
tensorboard --logdir output_dir
```

## Troubleshooting

**Issue**: `CUDA_VISIBLE_DEVICES not set`
- Solution: Run via `run_instr_pretrain.sh` which sets the GPU

**Issue**: `Import "dataloader_instr" could not be resolved`
- Solution: Run from pretrain directory: `cd pretrain && python train_instr.py ...`

**Issue**: Out of memory
- Solution: Reduce `batch_size` or `max_len`

**Issue**: Low JTP accuracy
- Solution: Check that data contains instr_addr tokens with correct format

## Next Steps

After pretraining:
1. Fine-tune on downstream tasks (function similarity, malware detection, etc.)
2. Evaluate embeddings with probing tasks
3. Compare with baseline jTrans on benchmarks
4. Analyze learned instruction embeddings
