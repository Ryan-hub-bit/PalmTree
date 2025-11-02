# Address-Aware PalmTree Training Tasks

## Overview
The model is trained to learn **address semantics and relationships** in binary code, not just token-level language modeling.

## Architecture Changes
- **Fusion Method**: Simple addition (like BERT) instead of MLP
  - `fused = semantic + address + position`
  - 0 extra parameters (was 147K with MLP)
- **Model Saved As**: `best_model_sum.pt`

## Training Tasks (Multi-Task Learning)

### Task 1: Masked Address Type Prediction (PRIMARY - weight 1.0)
**Goal**: Learn what each address type means from context

**How it works**:
- Randomly mask 15% of address types during training
- Model must predict the original type from surrounding tokens

**Examples**:
```
Input:  "call <MASK>"        → Target: <addr_code>
Input:  "mov rax [<MASK>]"   → Target: <addr_data>
Input:  "<MASK> push rbp"     → Target: <addr_start>
Input:  "ret <MASK>"          → Target: <addr_end>
```

**What model learns**:
- `<addr_code>` appears after `call`, `jmp`
- `<addr_data>` appears inside `[]` brackets
- `<addr_start>` appears at beginning of BB
- `<addr_end>` appears before control flow changes

---

### Task 2: Control Flow Edge Type (weight 0.8)
**Goal**: Understand relationships between basic blocks

**How it works**:
- Predict edge type between two consecutive BBs
- Uses [CLS] token representation

**Classes**:
- `fallthrough`: Sequential execution
- `branch`: Conditional jump
- `call`: Function call
- `return`: Function return
- `indirect`: Indirect jump/call

**What model learns**:
- BBs with `call` have call edges
- BBs with `jmp` have branch edges
- BBs without control flow have fallthrough edges

---

### Task 3: Address Distance Prediction (weight 0.5)
**Goal**: Learn spatial relationships between addresses

**How it works**:
- Sample pairs of addresses from same sequence
- Predict distance class between them

**Classes**:
- `same_bb` (0): Within same BB (start → end)
- `near` (1): < 1KB distance
- `medium` (2): 1KB - 64KB distance
- `far` (3): > 64KB distance

**Examples**:
```
<addr_start> 0x401000 ... <addr_end> 0x401020  → same_bb
call <addr_code> 0x401100                       → near
mov [<addr_data> 0x500000]                      → far
```

**What model learns**:
- `addr_start` and `addr_end` are close (same BB)
- Code addresses can be near (local) or far (cross-function)
- Data addresses are usually far (different memory region)
- Address value encodings capture spatial proximity

---

### Task 4: Next Token Prediction (OPTIONAL - weight 0.3)
**Goal**: General sequence modeling (secondary)

**How it works**:
- Standard autoregressive next-token prediction
- Lower weight because PalmTree already learned this

---

## Why This Approach Works

### 1. **Focused Learning**
- Primary task (masked address type) directly teaches address semantics
- Not distracted by general language modeling

### 2. **Context-Aware**
- Model learns: "after `call`, expect `<addr_code>`"
- Not just memorizing patterns, understanding context

### 3. **Relationship Learning**
- Distance task teaches spatial relationships
- Edge type task teaches control flow relationships
- Combined: complete understanding of address roles

### 4. **Simple Fusion**
- Addition instead of MLP prevents address dominance
- All three levels (semantic, address, position) contribute equally
- No learnable fusion parameters to amplify one signal

---

## Expected Results

### Before (with MLP, no masking):
- Address dominates: all tokens shift 25-40 units
- Opcodes corrupted: shift 31.9 (should be <5)
- No selectivity: uniform changes everywhere
- **Verdict**: DOMINATING (bad)

### After (with sum, masked training):
- Opcodes preserved: shift <5 units
- Address types enhanced: shift 8-15 units (only where relevant)
- Natural selectivity: opcodes clean, addresses enriched
- **Verdict**: SELECTIVE (good)

---

## Training Configuration

```python
# Primary task focused
TASK_WEIGHTS = {
    "addr_type": 1.0,      # Learn address semantics (PRIMARY)
    "edge_type": 0.8,      # Learn control flow
    "addr_distance": 0.5,  # Learn spatial relationships
    "next_bb": 0.3,        # Optional (PalmTree already knows this)
}

# Masking
ADDR_MASK_PROB = 0.15  # Mask 15% of addresses (like BERT)

# Architecture
Fusion: semantic + address + position (simple sum)
Parameters: 0 (no MLP, no learned weights)
```

---

## Usage

Train with:
```bash
python train.py --bb_pairs_file data/train_subset_100k.txt \
                --vocab_file ../pre-trained_model/palmtree/vocab
```

Model saved as: `output/best_model_sum.pt`

Evaluate with:
```bash
python visualize_token_level.py  # Check for SELECTIVE verdict
python analyze_token_shifts.py   # Verify opcode preservation
```
