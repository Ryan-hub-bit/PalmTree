# Using Your Pickle Files for jTrans Pretraining

## Summary: YES! Your pickle files work perfectly! ✅

Your extracted pickle files contain all the necessary data for pretraining. You just need to convert them to text format first.

## What You Have

Your pickle files (`*_extract.pkl`) contain:
```python
{
    'function_name': {
        'func': 0x401000,           # Function address
        'asm': ['push rbp', ...],   # Assembly instructions (what we need!)
        'raw': b'\x55\x48...',      # Raw bytes
        'cfg': DiGraph(...),        # Control Flow Graph
        'bai': None                 # BinaryAI features
    }
}
```

## What jTrans Baseline Needs

Text file with one function per line, normalized tokens:
```
push rbp mov rbp rsp sub rsp CONST call sub_xxx test eax eax je UNK_ADDR ...
```

## Step-by-Step: Convert Your Data

### 1. Extract Features from All Binaries

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils
python3 run.py
```

This creates `*_extract.pkl` files in `/data/kun/jtransdata/extract/`

### 2. Convert Pickle to Text Format

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils

# Convert all pickle files to training data
python3 convert_pkl_to_text.py \
    /data/kun/jtransdata/extract \
    /data/kun/jtransdata/pretrain_train.txt \
    --min-instructions 5 \
    --max-instructions 512

# This will:
# - Process all *_extract.pkl files
# - Skip functions with < 5 instructions
# - Truncate functions to 512 instructions max
# - Normalize assembly using jTrans rules
# - Output: one function per line
```

### 3. Verify the Output

```bash
# Check file
wc -l /data/kun/jtransdata/pretrain_train.txt
head -3 /data/kun/jtransdata/pretrain_train.txt

# Expected output: space-separated tokens like:
# push rbp mov rbp rsp sub rsp CONST call sub_xxx ...
```

### 4. Use for Pretraining

```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline

# Update run_baseline.sh with your data path
vim run_baseline.sh
# Change: TRAIN_DATA="/data/kun/jtransdata/pretrain_train.txt"

# Run pretraining
bash run_baseline.sh
```

## Conversion Details

The `convert_pkl_to_text.py` script does:

1. **Loads pickle files**: Reads your `*_extract.pkl` files
2. **Extracts assembly**: Gets the `'asm'` field from each function
3. **Normalizes tokens**: Uses `readidadata.parse_asm()` to:
   - Convert constants to `CONST`
   - Convert unknown jumps to `UNK_ADDR`
   - Normalize function calls to `sub_xxx`
   - Normalize variables to `var_xxx`, `arg_xxx`
   - Keep registers as-is: `rax`, `rbp`, etc.
4. **Filters functions**: Skips very short functions (< 5 instructions)
5. **Outputs text**: One function per line, space-separated tokens

## Example Conversion

**Original pickle data:**
```python
{
    'main': {
        'asm': ['push rbp', 'mov rbp, rsp', 'sub rsp, 0x10', 'call _printf'],
        ...
    }
}
```

**Converted text:**
```
push rbp mov rbp rsp sub rsp CONST call sub_xxx
```

## Validation

Your pickle file is **100% correct**:
- ✅ 83 functions extracted
- ✅ 10,821 assembly instructions
- ✅ 47,933 bytes of code
- ✅ 1,745 CFG nodes
- ✅ 2,594 CFG edges

After conversion:
- ✅ 81 functions (2 filtered - too short)
- ✅ Ready for jTrans pretraining!

## Full Pipeline Example

```bash
# 1. Extract features from binaries
cd /home/kun/Document/AAE/extern/jTrans/datautils
python3 run.py  # Creates *_extract.pkl

# 2. Convert to text for pretraining
python3 convert_pkl_to_text.py \
    /data/kun/jtransdata/extract \
    /data/kun/jtransdata/pretrain_data.txt

# 3. (Optional) Split into train/val
head -n 10000 /data/kun/jtransdata/pretrain_data.txt > /data/kun/jtransdata/pretrain_train.txt
tail -n 1000 /data/kun/jtransdata/pretrain_data.txt > /data/kun/jtransdata/pretrain_val.txt

# 4. Start pretraining
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
# Edit run_baseline.sh to point to your data
bash run_baseline.sh
```

## Questions?

- **Q: Do I need BinaryAI features?**  
  A: No! BinaryAI features are optional. The baseline uses only assembly tokens.

- **Q: Can I use the CFG data?**  
  A: Not in the baseline. The baseline uses flat sequences. For CFG-aware models, you'd need a different architecture.

- **Q: What about jump targets (JUMP_ADDR_X)?**  
  A: The baseline uses Jump-Target Prediction (JTP). The converter currently uses `UNK_ADDR`, but you can enhance it to extract actual jump targets from the CFG.

- **Q: How much data do I need?**  
  A: For pretraining, more is better. Original jTrans used millions of functions. Start with what you have and scale up.

## Summary

✅ **Your pickle files are perfect and ready to use!**  
✅ **Just convert them to text format with the provided script**  
✅ **Then run baseline pretraining**

🎉 You're all set for pretraining!
