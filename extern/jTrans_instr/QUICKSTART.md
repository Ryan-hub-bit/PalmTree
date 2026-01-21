# jTrans_instr Quick Start

## What Changed from Baseline jTrans

### 1. Jump Token Format
- **Baseline**: `JUMP_ADDR_{X}` where X is token position (0-511)
- **jTrans_instr**: `instr_addr_{i}` where i is instruction index (0-200)

### 2. Code Changes
**File**: `/home/kun/Document/AAE/extern/jTrans_instr/data.py`

- Changed `MAXLEN=512` → `MAXLEN=200` (line 15)
- Added `map_instr_id` to track instruction indices (not token positions)
- Changed jump tokens:
  - `JUMP_ADDR_{}` → `instr_addr_{}`
  - `JUMP_ADDR_EXCEEDED` → `instr_addr_exceed`
  - `UNK_JUMP_ADDR` → `unknown_instr_addr`

## How to Generate Data

### Option 1: Generate JSON format (for finetuning tasks)
```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
bash generate_baseline.sh
```

**What it does:**
1. Extracts functions from binaries using IDA Pro
2. Generates JSON dataset with instruction-level jumps

**Outputs:**
- `/data/kun/jtrans_instr/extract/` - Pickle files from IDA
- `/data/kun/jtrans_instr/func_blocks_baseline.json` - All functions
- `/data/kun/jtrans_instr/ground_truth_baseline.json` - Ground truth labels

### Option 2: Generate text file (for pretraining)
```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
bash generate_text.sh
```

**What it does:**
1. Converts pickle files to simple text format
2. One function per line with instr_addr_{i} jumps

**Outputs:**
- `/data/kun/jtrans_instr/instr_pretrain.txt` - Text file (like baseline_pretrain.txt)

**Note:** Run `generate_baseline.sh` first to create the pickle files!

### Step 3: Build vocabulary
```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
bash build_vocab.sh
```

**What it does:**
1. Extracts all unique tokens from instr_pretrain.txt
2. Creates vocabulary with instr_addr_{0..200} + special tokens + assembly tokens

**Outputs:**
- `/home/kun/Document/AAE/extern/jTrans_instr/jtrans_tokenizer/vocab.txt`

**Vocabulary structure:**
- Lines 1-201: `instr_addr_0` to `instr_addr_200`
- Lines 202-203: `instr_addr_exceed`, `unknown_instr_addr`
- Lines 204-208: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- Remaining lines: Assembly tokens from the data

### Output format example:
```
lea rdi GLOBAL_VAR lea rax GLOBAL_VAR cmp rax rdi jz instr_addr_2 mov rax cs:xxx test rax rax jz instr_addr_2 jmp rax retn
```

Where `instr_addr_2` means "jump to instruction 2" (not token 2).

## Key Files

| File | Purpose |
|------|---------|
| `data.py` | Core logic with instruction-level jump mapping |
| `datautils/generate_baseline.sh` | Main generation script |
| `datautils/run.py` | IDA Pro extraction runner |
| `datautils/create_baseline_dataset.py` | Pickle → JSON converter |

## Next Steps After Generation

1. **Build vocabulary** from the generated data
2. **Update pretrain config** to use `/data/kun/jtrans_instr/` paths
3. **Run pretraining** with instruction-level jumps
4. **Run finetuning** for downstream tasks

## Verification

Check that jump tokens are instruction-based:
```bash
# Should see instr_addr_0, instr_addr_1, etc. (not JUMP_ADDR_0)
head -1 /data/kun/jtrans_instr/func_blocks_baseline.json | grep -o "instr_addr_[0-9]*" | head -5
```

## Difference Summary

**What's the same:**
- Uses same binaries as baseline
- Uses same IDA Pro extraction
- Same CFG-based jump resolution
- Same token normalization (CONST, GLOBAL_VAR, etc.)

**What's different:**
- Jump addresses refer to **instructions** (0-200) not **tokens** (0-511)
- Token names: `instr_addr_*` instead of `JUMP_ADDR_*`
- Instruction counting logic in `data.py`
