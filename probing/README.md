# Position Embedding Probing

This directory contains experiments to probe and analyze the learned position embeddings from the address-aware model compared to the baseline model.

## Overview

We test for **linear relationships** between position values and learned embeddings at three levels:
1. **Binary Level** - Instruction-by-instruction across entire binary
2. **Function Level** - Within longest function only
3. **Basic Block Level** - Within longest basic block only

## Probing Approach

### Data Format
- **One instruction per line** with inline addresses
- Format: `opcode(0xADDR:bnorm:fnorm:bbnorm) operands`
- Each instruction has three normalized position values [0, 1]:
  - `bnorm`: position in entire binary
  - `fnorm`: position in current function
  - `bbnorm`: position in current basic block

### Test Strategy
1. **Binary Level**: Extract ALL instructions from binaries → test instruction-by-instruction linear relationship
2. **Function Level**: Extract LONGEST function only → test function-level position encoding
3. **BB Level**: Extract LONGEST basic block only → test BB-level position encoding

## Models

- **Address-Aware Model**: `../addressaware/output_addressaware_new/best_model.pt`
  - Uses 3 learnable position embeddings (binary_pos, function_pos, bb_pos)
  - Each position is normalized to [0, 1] range
  
- **Baseline Model**: `../addressaware/output_baseline_new/best_model.pt`
  - Uses sinusoidal sequential positions
  - No address-specific information

## Probing Tasks

### 1. Position Prediction
Given the contextualized embeddings, predict the position values at each level:
- Binary position regression
- Function position regression
- Basic block position regression

### 2. Position Clustering
Analyze how similar positions cluster together:
- Do instructions at similar binary positions have similar representations?
- Do function prologues/epilogues cluster together?
- Do basic block boundaries have distinct patterns?

### 3. Embedding Visualization
- t-SNE/UMAP visualization of embeddings colored by position
- Analyze separation by binary/function/BB level

## Files

## Files

### Core Scripts
- `generate_probing_data_linear.sh` - **NEW: Generate test data with one instruction per line**
- `probe_linear_relationship.py` - **NEW: Test for linear relationships in embeddings**
- `run_linear_probing.sh` - **NEW: Complete pipeline for linear relationship testing**

### Analysis & Visualization
- `probe_positions.py` - Original probing script (multi-level prediction)
- `position_predictor.py` - Linear probe models
- `visualize_embeddings.py` - t-SNE/PCA visualizations
- `analyze_results.py` - Result analysis and comparison
- `run_probing.sh` - Original probing pipeline

## Test Data Source

Test data is generated from binaries in `/home/kun/testbinary/`:
- `389-ds-base__libderef-plugin.so`
- `6tunnel__6tunnel`
- `a2jmidid__a2jmidi_bridge`
- `a52dec__liba52.so.0.0.0`
- `aalib__aasavefont`

The data generation script creates inline address format with 8 instructions per line.

## Usage

### Quick Start (Recommended - Linear Relationship Testing)

```bash
# Run complete linear relationship probing pipeline
cd /home/kun/Document/PalmTree/probing
./run_linear_probing.sh
```

This will:
1. Generate test data from binaries (if not exists):
   - `binary_level_test.txt` - All instructions, one per line
   - `function_level_test.txt` - Longest function instructions
   - `bb_level_test.txt` - Longest BB instructions
2. Extract embeddings from both models
3. Test for linear relationships using:
   - Linear probes (R², MAE)
   - Correlation analysis (Pearson, Spearman)
4. Compare address-aware vs baseline

### Manual Steps

```bash
# Step 1: Generate test data (one instruction per line)
./generate_probing_data_linear.sh

# Step 2: Run linear relationship probing
python probe_linear_relationship.py \
  --addressaware_model ../addressaware/output_addressaware_new/best_model.pt \
  --baseline_model ../addressaware/output_baseline_new/best_model.pt \
  --binary_test ./test_data/binary_level_test.txt \
  --function_test ./test_data/function_level_test.txt \
  --bb_test ./test_data/bb_level_test.txt \
  --vocab ../pre-trained_model/palmtree/vocab \
  --output_dir ./results

# Step 3: Visualize embeddings (optional)
python visualize_embeddings.py \
  --addressaware_model ../addressaware/output_addressaware_new/best_model.pt \
  --baseline_model ../addressaware/output_baseline_new/best_model.pt \
  --test_cfg ./test_data/binary_level_test.txt \
  --test_dfg ./test_data/binary_level_test.txt \
  --vocab ../pre-trained_model/palmtree/vocab \
  --output_dir ./visualizations
```

## Expected Results

### Linear Relationship Test

**Address-Aware Model** should show:
- High R² values (> 0.7) = strong linear relationship
- Strong correlation (Pearson r > 0.8)
- Low prediction error (MAE < 0.1)
- This proves the model learned meaningful position encodings

**Baseline Model** should show:
- Lower R² values (< 0.3) = weak/no linear relationship
- Weak correlation
- Higher prediction error
- Only sequential information, no address-based patterns

**Interpretation**:
- R² > 0.7: **STRONG** evidence of linear relationship
- R² 0.4-0.7: **MODERATE** linear relationship
- R² < 0.4: **WEAK** or no linear relationship

### What Each Test Shows

1. **Binary Level Test**:
   - Tests: Can embeddings predict position across entire binary?
   - Linear relationship = instructions at similar addresses have similar embeddings

2. **Function Level Test**:
   - Tests: Can embeddings predict position within a function?
   - Linear relationship = function structure is encoded (prologue → body → epilogue)

3. **BB Level Test**:
   - Tests: Can embeddings predict position within a basic block?
   - Linear relationship = local instruction ordering is preserved
