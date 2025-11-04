# CFG Pretrain

Configuration-based pretraining for assembly code understanding using control flow graph (CFG) information with **3-level positional embeddings**.

## Features

- **3-Level Embeddings Architecture:**
  1. **Semantic Embedding**: Instruction token meaning (vocabulary-based)
  2. **Address Position Embedding**: Global (binary-level) and local (function-level) position
  3. **Sequence Position Embedding**: Position in instruction sequence (standard positional encoding)

- **Multiple Pretraining Tasks:**
  - Masked Language Modeling (MLM)
  - CFG Edge Prediction
  - Address Value Prediction

- **Rich Address Information:**
  - 3-level address normalization (hex, binary-level, function-level)
  - Address type classification (code, data, unknown)
  - Special markers for external references

## Data Format

Input: Basic block pairs representing CFG edges
```
<addr_start:0x402118:0.269:0.5> mov rax rbx [SEP] <addr_start:0x402130:0.271:0.0> call <addr_code:...> [SEP]
```

## Usage

### 1. Filter CFG Data (Optional)
```bash
python ../scripts/filter_cfg_data.py \
  ../bb_pairs_output/all_bb_pairs.txt \
  ../bb_pairs_output/all_bb_pairs_filtered.txt \
  --no-unknown \
  --max-length 5000
```

### 2. Test Data Loader
```bash
cd cfg_pretrain
python test_dataloader.py
```

### 3. Train Model
```bash
python train.py \
  --data_file ../bb_pairs_output/all_bb_pairs.txt \
  --vocab_file ../pre-trained_model/palmtree/vocab \
  --output_dir checkpoints/cfg_pretrain \
  --batch_size 32 \
  --num_epochs 100
```

## Configuration

Edit `config.py` to adjust:
- Model architecture (embedding dim, layers, heads)
- Training hyperparameters
- Task weights
- MLM probability

## Data Loader

The `CFGPretrainDataset` provides:
- Automatic tokenization of assembly instructions
- Address feature extraction
- MLM mask generation
- Batch preparation with padding

Each batch contains:
- `input_ids`: Token IDs with MLM masking **(Level 1: Semantic)**
- `binary_positions`: Binary-level normalized positions **(Level 2: Global)**
- `function_positions`: Function-level normalized positions **(Level 2: Local)**
- `sequence_positions`: Sequential position indices **(Level 3: Sequence)**
- `attention_mask`: Valid token mask
- `segment_ids`: Source vs target BB
- `mlm_labels`: Ground truth for masked tokens
- `cfg_label`: Valid CFG edge indicator
- `source_addr_features`: Aggregated address features [5-dim]
- `target_addr_features`: Aggregated address features [5-dim]

## 3-Level Embeddings Explained

Your model should combine these three embedding levels:

1. **Semantic Embedding** (`input_ids`) - Learned from vocabulary
   - Maps token IDs to dense vectors representing instruction semantics
   - Example: `mov`, `rax`, `call` each get their own learned vectors

2. **Address Position Embedding** (`binary_positions` + `function_positions`)
   - **Binary-level** (global): Where in the entire binary (0.0 = start, 1.0 = end)
   - **Function-level** (local): Where in current function (0.0 = start, 1.0 = end)
   - Can use sinusoidal encoding or learned embeddings
   - Helps model understand relative code locations

3. **Sequence Position Embedding** (`sequence_positions`)
   - Standard transformer positional encoding
   - Position 0, 1, 2, ... in the token sequence
   - Helps model understand instruction order

**Final embedding** = Semantic + Address Position + Sequence Position

## Address Features (Aggregated)

5-dimensional feature vector per basic block:
1. Average binary normalization (position in binary)
2. Average function normalization (position in function)
3. Number of code references
4. Number of data references
5. Number of special markers

## Requirements

- Python 3.7+
- PyTorch 1.10+
- numpy

## Files

- `config.py` - Configuration parameters
- `data_loader.py` - Data loading and preprocessing
- `test_dataloader.py` - Test script
- `train.py` - Training script (TODO)
