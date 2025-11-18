# Testing Original PalmTree Model

## Overview

To test the original PalmTree model on the same test data, we need to convert the address-aware format to the original PalmTree format (without address annotations).

## Data Format Conversion

### Input Format (Address-Aware)
```
lea(0x4020f0:0.25433376:0.000000:0.0000) rdi [ rel address(0x405150:0.65485123:0.000000:0) ]
```

### Output Format (Original PalmTree)
```
lea rdi [ rel address ]
```

The conversion strips all address information while preserving:
- Token sequences
- Tab separations (for instruction grouping)
- Token order

## Usage

### Step 1: Convert Test Data

Run the conversion script to create PalmTree-compatible test files:

```bash
cd /home/kun/Document/PalmTree/addressaware
bash convert_test_data.sh
```

This will create:
- `../data/test/cfg/all_cfg_palmtree.txt` (CFG without addresses)
- `../data/test/dfg/all_dfg_palmtree.txt` (DFG without addresses)

### Step 2: Test Original PalmTree Model

```bash
bash test_palmtree.sh
```

The script automatically:
1. Checks if converted files exist
2. Runs conversion if needed
3. Tests the PalmTree model
4. Saves results to `palmtree_test_results.json`

### Manual Conversion

To convert individual files:

```bash
python3 convert_to_palmtree_format.py \
    --input ../data/test/cfg/all_cfg_combined.txt \
    --output ../data/test/cfg/all_cfg_palmtree.txt
```

## Files Created

### Conversion Scripts
- `convert_to_palmtree_format.py` - Python script to strip addresses
- `convert_test_data.sh` - Shell script to convert both CFG and DFG

### Test Scripts  
- `test_palmtree_model.py` - Python test script for PalmTree
- `test_palmtree.sh` - Shell script to run PalmTree tests

### Output Files
- `palmtree_test_results.json` - Test metrics for PalmTree model
- `../data/test/cfg/all_cfg_palmtree.txt` - Converted CFG data
- `../data/test/dfg/all_dfg_palmtree.txt` - Converted DFG data

## Comparison

After running both tests, you can compare:

**Address-Aware Model:**
```bash
cat output_addressaware/test_results.json
```

**Original PalmTree Model:**
```bash
cat palmtree_test_results.json
```

This allows you to evaluate the impact of adding address-aware positional embeddings.

## Notes

- The conversion is **lossless** for token sequences (only address metadata is removed)
- Tab separations are preserved (important for PalmTree's paired instruction format)
- Both models use the same vocabulary
- Test metrics (MLM, NSP) are directly comparable
