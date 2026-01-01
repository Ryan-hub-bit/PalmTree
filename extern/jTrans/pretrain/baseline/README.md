# Baseline jTrans Pretraining

This folder contains the **baseline** jTrans pretraining implementation with:
- **MLM (Masked Language Modeling)**: Predict masked regular tokens
- **JTP (Jump-Target Prediction)**: Predict masked jump target positions

**No address-aware features** - this is the original jTrans approach.

## Key Innovation: jTrans Trick

The baseline uses the original jTrans trick:
```python
position_embeddings = word_embeddings
```

This allows the model to learn position information from token context rather than absolute positions, which is beneficial for binary code where instruction positions vary across compilations.

## Files

### Core Files
- `model_baseline.py` - Baseline BERT model with jTrans trick (156 lines)
- `dataloader_baseline.py` - Data loading with MLM + JTP masking (280 lines)
- `train_baseline.py` - Training script (425 lines)

### Launcher
- `run_baseline.sh` - Shell script to run training

### Documentation
- `README.md` - This file

## Data Format

The dataloader expects one function per line, already tokenized:

```
push rbp mov rbp rsp sub rsp CONST call JUMP_ADDR_15 test eax eax je JUMP_ADDR_32 ...
```

Where:
- Regular tokens: `push`, `rbp`, `mov`, etc.
- Constants: `CONST` (normalized immediate values)
- Jump tokens: `JUMP_ADDR_X` where X is the target position

## Pretraining Tasks

### 1. MLM (Masked Language Modeling)
- Mask **15%** of regular tokens
- Predict the original token
- Standard BERT-style masking:
  - 80% replace with `[MASK]`
  - 10% replace with random token
  - 10% keep original

### 2. JTP (Jump-Target Prediction)
- Mask **20%** of `JUMP_ADDR_X` tokens
- Predict the target position X
- Same masking strategy as MLM

**Important**: MLM and JTP are applied to different tokens (regular vs jump), so they don't interfere.

## Usage

### 1. Prepare Data

Your data should be preprocessed into single-line format with jump tokens converted:

```bash
# Example: one function per line
cat train_data.txt
push rbp mov rbp rsp call JUMP_ADDR_10 ...
mov rdi rsi call JUMP_ADDR_5 ret
...
```

### 2. Configure Paths

Edit `run_baseline.sh` to set your paths:

```bash
TRAIN_PATH="/path/to/train_data.txt"
TEST_PATH="/path/to/test_data.txt"
TOKENIZER_PATH="/path/to/tokenizer"  # BertTokenizer directory
OUTPUT_DIR="./output_baseline"
```

### 3. Run Training

```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
chmod +x run_baseline.sh
./run_baseline.sh
```

Or run directly with custom parameters:

```bash
python train_baseline.py \
    --train_path ./data/train.txt \
    --test_path ./data/test.txt \
    --tokenizer_path ./tokenizer \
    --output_dir ./output \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --mlm_probability 0.15 \
    --jtp_probability 0.20
```

## Hyperparameters

### Model Architecture
- **Hidden size**: 768 (BERT-base)
- **Layers**: 12
- **Attention heads**: 12
- **Intermediate size**: 3072
- **Dropout**: 0.1

### Training
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Warmup steps**: 10,000
- **Max sequence length**: 512
- **MLM probability**: 0.15
- **JTP probability**: 0.20 (from jTrans paper)

### Optimizer
- **AdamW** with linear warmup schedule
- Gradient clipping: 1.0

## Output Structure

```
output_baseline/
├── best_model/              # Best model checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_info.json
├── checkpoint_epoch_1/      # Per-epoch checkpoints
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_info.json
├── checkpoint_epoch_2/
│   └── ...
├── training_history.json    # Full training metrics
└── train_baseline_*.log     # Training logs
```

## Training Metrics

The training script tracks:

### MLM Task
- **MLM Loss**: Cross-entropy on masked tokens
- **MLM Accuracy**: Token-level accuracy on predictions

### JTP Task
- **JTP Loss**: Cross-entropy on jump target positions
- **JTP Accuracy**: Position-level accuracy on predictions

### Combined
- **Total Loss**: MLM Loss + JTP Loss
- Reported separately for train and validation

## Comparison with Address-Aware

| Feature | Baseline | Address-Aware |
|---------|----------|---------------|
| **Position Embeddings** | word_embeddings | Hierarchical (binary/func/bb) |
| **MLM Task** | ✅ Yes | ✅ Yes |
| **JTP Task** | ✅ Yes | ✅ Yes |
| **Address Info** | ❌ No | ✅ Yes (ADDR, IMM, VAR, DADDR) |
| **Hierarchical Positions** | ❌ No | ✅ Yes (3 levels) |
| **Data Format** | Single line | 10 lines per function |
| **Complexity** | Simple | Complex |

## Expected Performance

### Training Time
- ~2-3 hours per epoch on single V100 GPU (depends on dataset size)
- Full training (10 epochs): ~20-30 hours

### Memory Usage
- ~10GB GPU memory with batch_size=32, max_len=512
- Reduce batch_size if OOM

### Typical Metrics (after 10 epochs)
- **MLM Accuracy**: ~60-70% (on validation)
- **JTP Accuracy**: ~30-50% (position prediction is harder)
- **Total Loss**: ~3-5 (combined)

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 16

# Reduce sequence length
--max_len 256

# Reduce model size
--hidden_size 512
--num_hidden_layers 6
```

### Data Loading Issues
- Ensure data is properly tokenized
- Check that JUMP_ADDR_X tokens use valid positions
- Verify tokenizer has all special tokens

### Poor JTP Performance
- JTP is inherently harder than MLM
- Consider increasing `jtp_probability` to 0.3
- Ensure jump tokens are correctly formatted
- Check that target positions are within max_len

## Citation

If you use this baseline implementation, please cite:

```bibtex
@article{jtrans,
  title={jTrans: Jump-Aware Transformer for Binary Code Similarity},
  author={...},
  journal={...},
  year={...}
}
```

## Next Steps

After baseline pretraining, you can:

1. **Fine-tune** on downstream tasks (function similarity, etc.)
2. **Compare** with address-aware version
3. **Experiment** with different masking strategies
4. **Analyze** learned representations

For address-aware pretraining with hierarchical positions, see:
- `../train_address_aware_enhanced.py`
- `../dataloader_address_aware.py`
