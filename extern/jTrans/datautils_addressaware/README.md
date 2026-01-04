# Address-Aware Data Generation

**⚠️ IMPORTANT UPDATE (January 2, 2026):** This pipeline has been significantly improved! See below for changes.

This directory contains tools to generate address-aware training data from binaries using IDA Pro.

## 🎯 Recent Improvements (v2.0)

**Three major fixes have been implemented:**

1. **✨ Memory Operand Tokenization** - Properly splits `[rbp+var(0x8)]` → `[ rbp + var(0x8) ]`
2. **🔍 PLT Symbol Preservation** - Shows `.printf`, `.malloc` instead of generic addresses  
3. **📁 Log File Organization** - Separates logs into `logs/` subdirectory

**👉 For details, see:**
- `VISUAL_COMPARISON.md` - Before/after examples (**START HERE**)
- `CHANGES_SUMMARY.md` - Technical details
- `NEXT_STEPS.md` - How to regenerate data

## Overview

The address-aware format adds hierarchical positional information to each token:
- **bnorm**: Binary-level normalization (function position in binary)
- **fnorm**: Function-level normalization (BB position in function)  
- **bbnorm**: Basic block-level normalization (instruction position in BB)

**Important**: This pipeline saves **both**:
1. **Pickle files** (like baseline jTrans) - Contains function metadata, CFG, raw bytes for downstream tasks
2. **Address-aware text files** - Contains hierarchical position-encoded tokens for pretraining
3. **Log files** (NEW: in `logs/` subdirectory) - Processing logs for debugging

## Output Files

For each binary, three files are generated:

### 1. Pickle File: `{binary_name}_extract.pkl`
Contains per-function data (same as baseline jTrans):
- `func`: Function address
- `asm`: Assembly instruction list
- `raw`: Raw bytes
- `cfg`: Control flow graph (NetworkX)
- `bai`: BinaryAI features (if available)

**Used for**: Function similarity, cross-optimization matching, downstream tasks

### 2. Text File: `{binary_name}_addressaware.txt` (IMPROVED ✨)
Contains address-aware formatted instructions with proper tokenization:
```
# New format (v2.0):
push(0x401000:0.1:0.0:0.0) rbp
mov(0x401001:0.11:0.05:0.1) rbp rsp
mov(0x401004:0.12:0.10:0.2) rax [ rbp + var(0x8) ]  ← Spaces added!
call(0x401010:0.15:0.20:0.3) .printf  ← PLT symbol preserved!
```

**Used for**: Pretraining with hierarchical position embeddings

### 3. Log File: `{binary_name}_ida.log` (NEW LOCATION 📁)
Now saved in `logs/` subdirectory (set via `LOGROOT` environment variable):
```
[INFO] Binary: test_binary
[INFO] Processed 42 functions
[INFO] Total instructions: 1337
[SUCCESS] Files saved successfully
```

**Used for**: Debugging, error checking, processing verification

## Two Approaches

### Approach 1: Direct IDA Generation (Recommended)
Use IDA Pro to directly generate address-aware format from binaries.

**Advantages**:
- Preserves function order perfectly
- No intermediate pickle files needed
- More efficient pipeline
- Direct access to IDA's analysis

### Approach 2: Pickle Conversion
Convert existing jTrans pickle files to address-aware format.

**Advantages**:
- Works with existing pickle files
- No need to re-run IDA Pro
- Faster if pickles already exist

## Token Format

### Opcodes
```
push(0x401000:0.12345678:0.23456789:0.34567890)
     ^^^^^^^^ ^^^^^^^^^^:^^^^^^^^^^:^^^^^^^^^^
     address   bnorm      fnorm      bbnorm
```

### Jump Targets (Code Addresses)
```
address(0x401050:0.12345678:0.23456789:0.34567890)
```

### Data Addresses
```
daddr(0x600000:0.50000000:0.25000000:0.00000000)
      ^^^^^^^^ ^^^^^^^^^^:^^^^^^^^^^:^^^^^^^^^^
      address   sec_pos    addr_in_sec  bb_pos(0)
```

### Stack Variables
```
var(0x10)    # Stack offset
```

### Immediate Values
```
imm          # No brackets, just "imm"
```

### Displacements
```
disp         # Memory displacement (e.g., [rbp+0x10])
```

## Usage

### IDA-Based Generation (Recommended)

**Step 1**: Process binaries with IDA Pro
```bash
bash run_ida_addressaware.sh <input_binaries_dir> <output_dir> [dataroot_dir]

# Example: Process stripped binaries, save to extract/
bash run_ida_addressaware.sh /data/kun/jtransdata/small_train /data/kun/jtransdata/addr_extract

# Example: With separate unstripped binary directory for symbol info
bash run_ida_addressaware.sh \
    /data/kun/jtransdata/dataset \
    /data/kun/jtransdata/extract \
    /data/kun/jtransdata/unstripped
```

**What this does**:
- Processes each binary in `input_binaries_dir`
- Saves `{binary}_extract.pkl` to `output_dir` (for downstream tasks)
- Saves `{binary}_addressaware.txt` to `output_dir` (for pretraining)
- Uses `SAVEROOT` and `DATAROOT` environment variables

**Step 2**: Combine text files for training
```bash
bash combine_and_split.sh /data/kun/jtransdata/addr_extract /data/kun/jtransdata

```

This creates:
- `addressaware_train.txt` (90% of data)
- `addressaware_test.txt` (10% of data)

**Step 3**: Create vocabulary
cd ../pretrain/address_aware
```bash
python3 create_vocab.py

**Step 4**: Train model
```bash
bash run_addressaware_pretrain.sh
```

### Approach 2: Convert Pickles

**Quick Start**:
```bash
bash generate_addressaware_data.sh
```


## Files

### IDA-based Generation (Approach 1)
- **process.py**: IDA Pro script to generate address-aware format
- **run_ida_addressaware.sh**: Batch process binaries with IDA
- **combine_and_split.sh**: Combine files and split train/test


### Documentation
- **README.md**: This file

## Output Format

Each line represents one function:
```
push(0x401000:0.1:0.0:0.0) rbp mov(0x401001:0.11:0.05:0.1) rbp rsp lea(0x401005:0.15:0.1:0.2) rax address(0x40100c:0.2:0.15:0.3)
```

## Implementation Details

### process.py
- Runs inside IDA Pro environment
- Extracts functions in order
- Applies token preprocessing (based on cfg_hierarchical_icfg_ida.py)
- Computes hierarchical positions:
  - **bnorm**: `(func_start - min_addr) / (max_addr - min_addr)`
  - **fnorm**: `(bb_start - func_start) / (func_end - func_start)`
  - **bbnorm**: `(inst_addr - bb_start) / (bb_end - bb_start)`
- Handles different address types:
  - Code addresses: Use hierarchical positions
  - Data addresses: Use section-based positions
  - Stack variables: Convert to `var(0xOFFSET)`
  - Immediates: Replace with `imm`

### Token Preprocessing
- **Immediates**: Detected by `o_imm` operand type → `imm`
- **Stack variables**: `var_XX` → `var(0xXX)`
- **Displacements**: `o_displ`/`o_phrase` → `disp`
- **Code addresses**: Wrapped with `address(0xADDR:bnorm:fnorm:bbnorm)`
- **Data addresses**: Wrapped with `daddr(0xADDR:sec_pos:addr_in_sec:0.0)`

### Section Handling
- **.text**: Code addresses with hierarchical positions
- **.data/.rodata/.bss**: Data addresses with section-based positions

## Next Steps

After generating the data:
1. Create vocabulary: `WordVocab.create_vocab(train.txt, vocab.pkl)`
2. Update config paths in `run_addressaware_pretrain.sh`
3. Train model: `bash run_addressaware_pretrain.sh`
4. Compare with baseline: Check `../pretrain/address_aware/COMPARISON.md`
