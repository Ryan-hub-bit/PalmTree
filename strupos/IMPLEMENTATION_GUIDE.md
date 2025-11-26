# STRUPOS Implementation Guide

This document explains the complete implementation of STRUPOS - a BERT model trained from scratch for binary code understanding, based on the AddressAware architecture but without using pretrained PalmTree weights.

## Table of Contents
1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Data Pipeline](#data-pipeline)
4. [Training Tasks](#training-tasks)
5. [Configuration](#configuration)
6. [Step-by-Step Implementation](#step-by-step-implementation)

---

## Overview

**Goal**: Train a BERT model from scratch using CFG/DFG data with address-aware embeddings.

**Key Differences from AddressAware**:
- ❌ No pretrained PalmTree weights
- ✅ All components trained from scratch
- ✅ Standalone implementation (no parent directory dependencies)
- ✅ Configurable tasks for ablation studies

**Data Format**:
- Input: IDA Pro generated CFG/DFG files
- Format: `opcode(addr:binary_pos:function_pos:bb_pos) operand1 operand2 ...`
- Example: `mov(0x400000:0.5:0.3:0.2) rax rbx`

---

## Architecture Components

### 1. Model Components (`transformer_components.py`)

**Purpose**: All transformer building blocks copied locally from PalmTree

**Components**:
- `GELU`: Gaussian Error Linear Unit activation
- `LayerNorm`: Layer normalization
- `SublayerConnection`: Residual connection + layer norm
- `PositionwiseFeedForward`: FFN layer
- `Attention`: Single-head attention
- `MultiHeadedAttention`: Multi-head attention mechanism
- `TransformerBlock`: Complete transformer block
- `MaskedLanguageModel`: MLM prediction head
- `NextSentencePrediction`: NSP prediction head

**Why standalone?**: No dependencies on parent PalmTree directory.

### 2. Address-Aware Embeddings (`address_embedding.py`)

**Purpose**: Create embeddings with 3-level address hierarchy

**Components**:

#### a. `SequencePositionalEmbedding`
- Standard sinusoidal position encoding (like BERT)
- Encodes sequential position: 0, 1, 2, 3, ...
- Formula: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`

#### b. `AddressPositionalEmbedding`
- **3-level address encoding**:
  - `binary_pos`: func Position within entire binary (0.0-1.0)
  - `function_pos`: bb Position within current function (0.0-1.0)
  - `bb_pos`: instruction Position within current basic block (0.0-1.0)
- Each level gets sinusoidal encoding
- Concatenated and projected to embedding dimension
- **Learnable weights** to balance the three levels

#### c. `AddressAwareBERTEmbedding`
- **Combines 4 embedding types** (or 3 if address embedding disabled):
  1. Token embedding (vocab → hidden)
  2. Sequence position embedding (sinusoidal)
  3. Address position embedding (3-level, OPTIONAL)
  4. Segment embedding (for NSP: 0 or 1)
- Formula: `embedding = token + seq_pos + addr_pos + segment`
- **Optional address embedding**: Set `use_address_embedding=False` to use standard BERT

### 3. BERT Model (`model.py`)

#### a. `AddressAwareBERT`
- Base BERT encoder with address-aware embeddings
- N transformer blocks (default: 12 layers)
- Multi-head attention (default: 12 heads)
- Hidden size: 768

#### b. `AddressAwareBERTForPretraining`
- **4 prediction heads** (all optional):
  1. `MLM`: Masked Language Model (CFG only)
  2. `CWP`: CFG Next Sentence Prediction (order coherence)
  3. `DUP`: DFG Next Sentence Prediction (trace coherence)
  4. `SCOPE`: Scope prediction (3-class: same BB / same function / different)

**Configurable Tasks**:
```python
model = AddressAwareBERTForPretraining(
    bert_model,
    vocab_size,
    enable_mlm=True,          # Can disable for ablation
    enable_nsp_cfg=True,      # Can disable for ablation
    enable_nsp_dfg=True,      # Can disable for ablation
    enable_scope=True         # Can disable for ablation
)
```

---

## Data Pipeline

### 1. Data Format

**Input Files**:
- CFG: `/data/kun/train_cdfg/cfg/*.txt` (7,065 binaries)
- DFG: `/data/kun/train_cdfg/dfg/*.txt` (7,065 binaries)

**Line Format**:
```
inst1 \t inst2 \t inst3 \t ... \t inst8
```

**Instruction Format**:
```
opcode(hex_address:binary_norm:function_norm:bb_norm) operand1 operand2 ...
```

**Example**:
```
mov(0x400000:0.5:0.3:0.2) rax rbx	push(0x400005:0.51:0.35:0.5) rbp	call(0x40000a:0.52:0.4:0.8) address(0x401234:0.6:0.1:0.0)
```

### 2. DataLoader (`dataloader_all_pairs.py`)

**Strategy**: Create ALL consecutive instruction pairs for NSP training

**For 8-instruction line**:
- MLM: Uses all 8 instructions → 1 sample
- NSP-CFG: Creates 7 pairs → (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8)
- NSP-DFG: Creates consecutive pairs from DFG lines

**Key Methods**:

#### `_parse_instruction(inst_text)`
- Extracts tokens and 3-level positions from one instruction
- Returns: `tokens=['mov', 'rax', 'rbx']`, `positions=[(0.5, 0.3, 0.2), ...]`

#### `_parse_line(line)`
- Parses entire line into merged token sequence
- Used for: DFG processing

#### `_parse_line_with_separators(line)`
- Parses line keeping instruction boundaries with `<eos>` markers
- Format: `<sos> inst1_tokens <eos> inst2_tokens <eos> ... <eos>`
- Used for: CFG MLM (maintains instruction structure)

#### `_create_all_consecutive_pairs(line, line_idx)`
- Creates ALL consecutive pairs from a line
- For each pair, randomly decides positive (50%) or negative (50%)
- **Negative pair**: Replaces inst2 with random instruction (ensures it's different)
- Returns: `[(line_idx, inst1, inst2, label), ...]`

#### `_mask_tokens(tokens)`
- Applies MLM masking (15% of tokens)
- **Never masks special tokens**: `<sos>`, `<eos>`, `[PAD]`, `[CLS]`, `[SEP]`
- Distribution:
  - 80%: Replace with `[MASK]`
  - 10%: Replace with random token
  - 10%: Keep original

#### `__getitem__(index)`
Returns dictionary with:
```python
{
    # CFG MLM (full line with instruction separators)
    'cfg_mlm_input': masked_tokens,
    'cfg_mlm_label': original_tokens,
    'cfg_mlm_binary_pos': binary_positions,
    'cfg_mlm_function_pos': function_positions,
    'cfg_mlm_bb_pos': bb_positions,
    
    # CFG NSP (consecutive pair)
    'cfg_nsp_input': unmasked_pair,
    'cfg_segment_label': [0,0,0,1,1,1],  # Segment A vs B
    'cfg_is_next': 1_or_0,  # Is inst2 consecutive to inst1?
    'cfg_nsp_binary_pos': positions,
    'cfg_nsp_function_pos': positions,
    'cfg_nsp_bb_pos': positions,
    
    # DFG NSP (consecutive pair)
    'dfg_nsp_input': unmasked_pair,
    'dfg_segment_label': [0,0,0,1,1,1],
    'dfg_is_next': 1_or_0,
    'dfg_nsp_binary_pos': positions,
    'dfg_nsp_function_pos': positions,
    'dfg_nsp_bb_pos': positions,
}
```

**Important Design Decisions**:

1. **NSP uses unmasked tokens** (not masked like MLM)
   - Why: NSP needs full semantic understanding to judge if sequences are consecutive
   
2. **Special tokens never masked**
   - Why: They provide structural information

3. **Dynamic NSP randomization** (not pre-assigned)
   - Each epoch sees different positive/negative labels for same pair
   - Matches standard BERT training practice
   - Better generalization

4. **Ensures negative ≠ positive**
   - Checks that random instruction is different from true consecutive
   - Prevents false negative labels

### 3. Vocabulary (`create_vocab.py`)

**Purpose**: Generate vocabulary from all data (train + val + test)

**Process**:
1. Parse all CFG/DFG files
2. Extract tokens from instructions
3. **Address info excluded**: Only opcode + operands → vocab
4. Filter by frequency (min_freq=2)
5. Add special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, `<sos>`, `<eos>`
6. Save to `vocab.txt`

**Example vocab.txt**:
```
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
<sos>
<eos>
mov
push
call
rax
rbx
address
...
```

---

## Training Tasks

### 1. Masked Language Modeling (MLM)

**Goal**: Predict masked tokens from context

**Input**: CFG line with instruction separators
```
<sos> mov rax rbx <eos> [MASK] rbp <eos> call address <eos>
```

**Target**: Predict `[MASK]` = `push`

**Why instruction separators?**
- Maintains instruction boundaries
- Model learns instruction-level structure
- Similar to BERT's sentence structure

### 2. Next Sentence Prediction - CFG (NSP-CFG)

**Goal**: Predict if instruction B follows instruction A in control flow

**Input**:
```
<sos> mov rax rbx <eos> push rbp <eos>
```
**Segment labels**: `[0, 0, 0, 0, 0, 1, 1, 1, 1]`

**Label**:
- 1 (IsNext): B is the true consecutive instruction
- 0 (NotNext): B is a random instruction from elsewhere

**Semantics**: "Does this instruction continue the control flow?"

### 3. Next Sentence Prediction - DFG (NSP-DFG)

**Goal**: Predict if instruction B has correct data dependency on instruction A

**Input**: Same format as NSP-CFG but from DFG data

**Label**:
- 1 (IsNext): B has the correct data dependency on A
- 0 (NotNext): B is a random instruction with incorrect dependency

**Semantics**: "Is this the correct data flow dependency?"

### 4. Scope Prediction (SCOPE) - OPTIONAL

**Goal**: Classify relationship between two instructions

**Labels**:
- 0: Different functions
- 1: Same function, different basic blocks
- 2: Same basic block

**Status**: Currently disabled (no scope data available)

---

## Configuration

### Configuration File (`config.py`)

All hyperparameters and task flags in one place:

```python
# ==================== Data Paths ====================
cfg_train = "/data/kun/train_cdfg/cfg"
dfg_train = "/data/kun/train_cdfg/dfg"
vocab_path = "./vocab.txt"

# ==================== Model Architecture ====================
hidden = 768
layers = 12
attn_heads = 12
seq_len = 100
dropout = 0.1

# ==================== Task Selection (Ablation Study) ====================
enable_mlm = True           # Masked Language Modeling
enable_nsp_cfg = True       # CFG Next Sentence Prediction
enable_nsp_dfg = True       # DFG Next Sentence Prediction
enable_scope = False        # Scope Prediction (disabled - no data)
use_address_embedding = True  # 3-level address embeddings

# ==================== Training Hyperparameters ====================
epochs = 10
batch_size = 1024
lr = 1e-4
warmup_steps = 10000
mask_prob = 0.15
nsp_prob = 0.5

# ==================== Output ====================
output_dir = "./output_strupos"
log_dir = "./log_strupos"
```

**Ablation Study Examples**:

```python
# Test without address embeddings (standard BERT)
use_address_embedding = False

# Test only MLM
enable_mlm = True
enable_nsp_cfg = False
enable_nsp_dfg = False

# Test only NSP-CFG
enable_mlm = False
enable_nsp_cfg = True
enable_nsp_dfg = False

# Test MLM + NSP-CFG (no DFG)
enable_mlm = True
enable_nsp_cfg = True
enable_nsp_dfg = False
```

### Training Script (`train_from_scratch.py`)

**Now uses config.py for all parameters** instead of command-line arguments.

**Modified to**:
1. Import config at start of main()
2. Use config values as argparse defaults
3. Allow CLI overrides if needed

**Usage**:
```bash
# Use all config.py values
python train_from_scratch.py

# Override specific values
python train_from_scratch.py --batch_size 512 --lr 5e-5
```

---

## Step-by-Step Implementation

### Step 1: Create Vocabulary

```bash
cd /home/kun/Document/PalmTree/strupos
python create_vocab.py
```

**Output**: `vocab.txt` with all tokens from train/val/test data

### Step 2: Configure Training

Edit `config.py`:
```python
# Choose which tasks to enable
enable_mlm = True
enable_nsp_cfg = True
enable_nsp_dfg = True
enable_scope = False  # No scope data
use_address_embedding = True

# Set hyperparameters
batch_size = 1024
lr = 1e-4
epochs = 20
```

### Step 3: Initialize Model

```python
import config
from model import AddressAwareBERT, AddressAwareBERTForPretraining

# Create base BERT
bert = AddressAwareBERT(
    vocab_size=len(vocab),
    hidden=config.hidden,
    n_layers=config.layers,
    attn_heads=config.attn_heads,
    dropout=config.dropout,
    max_len=config.seq_len,
    use_address_embedding=config.use_address_embedding  # Can disable!
)

# Create pretraining model with optional tasks
model = AddressAwareBERTForPretraining(
    bert,
    vocab_size=len(vocab),
    enable_mlm=config.enable_mlm,
    enable_nsp_cfg=config.enable_nsp_cfg,
    enable_nsp_dfg=config.enable_nsp_dfg,
    enable_scope=config.enable_scope
)
```

### Step 4: Create DataLoader

```python
from dataloader_all_pairs import AllConsecutivePairsDataset

train_dataset = AllConsecutivePairsDataset(
    cfg_corpus_path=config.cfg_train,
    dfg_corpus_path=config.dfg_train,
    vocab=vocab,
    seq_len=config.seq_len,
    mask_prob=config.mask_prob,
    nsp_prob=config.nsp_prob,
    data_percentage=config.data_percentage,
    train_split=config.train_split,
    is_train=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers
)
```

### Step 5: Training Loop

```python
for epoch in range(config.epochs):
    for batch in train_loader:
        # Get data
        cfg_mlm_input = batch['cfg_mlm_input'].to(device)
        cfg_mlm_label = batch['cfg_mlm_label'].to(device)
        cfg_mlm_binary_pos = batch['cfg_mlm_binary_pos'].to(device)
        cfg_mlm_function_pos = batch['cfg_mlm_function_pos'].to(device)
        cfg_mlm_bb_pos = batch['cfg_mlm_bb_pos'].to(device)
        
        cfg_nsp_input = batch['cfg_nsp_input'].to(device)
        cfg_segment_label = batch['cfg_segment_label'].to(device)
        cfg_is_next = batch['cfg_is_next'].to(device)
        cfg_nsp_binary_pos = batch['cfg_nsp_binary_pos'].to(device)
        cfg_nsp_function_pos = batch['cfg_nsp_function_pos'].to(device)
        cfg_nsp_bb_pos = batch['cfg_nsp_bb_pos'].to(device)
        
        # Forward pass for CFG (MLM + NSP)
        mlm_output, nsp_output = model(
            cfg_nsp_input,  # Use NSP input (unmasked)
            cfg_segment_label,
            cfg_nsp_binary_pos,
            cfg_nsp_function_pos,
            cfg_nsp_bb_pos,
            corpus_type='cfg'
        )
        
        # Calculate losses (only if task enabled)
        loss = 0
        
        if config.enable_mlm and mlm_output is not None:
            # For MLM, use separate masked input
            mlm_pred, _ = model(
                cfg_mlm_input,
                torch.zeros_like(cfg_segment_label),  # MLM doesn't use segments
                cfg_mlm_binary_pos,
                cfg_mlm_function_pos,
                cfg_mlm_bb_pos,
                corpus_type='cfg'
            )
            mlm_loss = criterion(mlm_pred.transpose(1, 2), cfg_mlm_label)
            loss += mlm_loss
        
        if config.enable_nsp_cfg and nsp_output is not None:
            nsp_loss = criterion(nsp_output, cfg_is_next)
            loss += nsp_loss
        
        # DFG NSP (if enabled)
        if config.enable_nsp_dfg:
            dfg_nsp_input = batch['dfg_nsp_input'].to(device)
            # ... similar to CFG NSP
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Step 6: Run Training

```bash
cd /home/kun/Document/PalmTree/strupos
python train_from_scratch.py
```

---

## Key Design Principles

### 1. Standalone Implementation
- No dependencies on parent PalmTree directory
- All transformer components copied locally
- Self-contained in `/home/kun/Document/PalmTree/strupos/`

### 2. Train from Scratch
- No pretrained weights loaded
- All embeddings initialized randomly
- All transformer layers trained from scratch

### 3. Configurable Tasks
- Each task (MLM, NSP-CFG, NSP-DFG, SCOPE, Address Embedding) can be enabled/disabled
- Easy ablation studies
- All flags in `config.py`

### 4. Address-Aware Embeddings (Optional)
- **3-level hierarchy**: binary → function → basic block
- **Sinusoidal encoding** for each level
- **Learnable weights** to balance levels
- **Can be disabled** to become standard BERT

### 5. Data Augmentation
- Dynamic NSP randomization each epoch
- MLM masks different tokens each epoch
- Better generalization

### 6. Instruction-Level Structure
- MLM uses instruction separators (`<eos>`)
- Maintains semantic units (instructions)
- Not just token-level modeling

---

## File Structure

```
/home/kun/Document/PalmTree/strupos/
├── model.py                      # BERT models with optional tasks
├── address_embedding.py          # Address-aware embeddings (optional)
├── transformer_components.py     # All transformer building blocks
├── dataloader_all_pairs.py       # DataLoader for ALL consecutive pairs
├── train_from_scratch.py         # Training script (uses config.py)
├── config.py                     # Configuration (all hyperparameters + task flags)
├── create_vocab.py               # Vocabulary generation
├── vocab.txt                     # Generated vocabulary
├── output_strupos/               # Model checkpoints
├── log_strupos/                  # Training logs
└── IMPLEMENTATION_GUIDE.md       # This file
```

---

## Summary of Changes from AddressAware

| Aspect | AddressAware | STRUPOS |
|--------|-------------|---------|
| **Pretrained weights** | Loads frozen PalmTree BERT | Train all from scratch |
| **Dependencies** | Requires parent PalmTree/ | Standalone, local copies |
| **Configuration** | Command-line arguments | config.py + optional CLI |
| **Tasks** | All tasks always active | Each task optional (ablation) |
| **Address embedding** | Always used | Optional (can disable) |
| **NSP strategy** | Split into halves | ALL consecutive pairs |
| **Instruction structure** | Merged tokens | Instruction separators |
| **Special token masking** | Can mask any token | Never masks special tokens |

---

## Next Steps

1. **Generate vocabulary**: `python create_vocab.py`
2. **Configure tasks**: Edit `config.py`
3. **Run training**: `python train_from_scratch.py`
4. **Ablation studies**: Disable tasks one by one to measure contribution
5. **Evaluate**: Use trained model for downstream tasks

---

## Troubleshooting

### Issue: Out of memory
- Reduce `batch_size` in config.py
- Reduce `seq_len` in config.py

### Issue: Training too slow
- Increase `batch_size` (if memory allows)
- Reduce `num_workers` to reduce CPU overhead
- Use `multi_gpu = True` in config.py

### Issue: Model not learning
- Check if tasks are enabled: `enable_mlm=True`, etc.
- Verify data format matches expected format
- Check learning rate (try 1e-4 to 5e-4)
- Ensure warmup steps are set (default: 10000)

### Issue: Vocab file not found
- Run `python create_vocab.py` first
- Check paths in config.py match your data location

---

**End of Implementation Guide**
