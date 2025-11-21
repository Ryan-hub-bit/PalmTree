# BB Bucket Probe

This folder contains tools for probing BERT models using basic block (BB) size bucket prediction.

## Overview

The BB bucket probe evaluates how well BERT models (address-aware vs baseline) encode positional information within basic blocks by predicting which position bucket an instruction belongs to based on its `bb_norm` value.

### Bucketing Strategy

Instructions are classified into 10 buckets based on their normalized position within the basic block (`bb_norm` ∈ [0.0, 1.0]):
- **Bucket 0**: bb_norm ∈ [0.00, 0.10) - Beginning of BB
- **Bucket 1**: bb_norm ∈ [0.10, 0.20)
- **Bucket 2**: bb_norm ∈ [0.20, 0.30)
- **Bucket 3**: bb_norm ∈ [0.30, 0.40)
- **Bucket 4**: bb_norm ∈ [0.40, 0.50) - Middle of BB
- **Bucket 5**: bb_norm ∈ [0.50, 0.60)
- **Bucket 6**: bb_norm ∈ [0.60, 0.70)
- **Bucket 7**: bb_norm ∈ [0.70, 0.80)
- **Bucket 8**: bb_norm ∈ [0.80, 0.90)
- **Bucket 9**: bb_norm ∈ [0.90, 1.00] - End of BB

## Files

- `generate_bb_bucket_data.py`: Generate instruction-level data from binaries
- `bb_bucket_probe.py`: Run probe evaluation on BERT models
- `README.md`: This file

## Usage

### Step 1: Generate Data

**Option A: Process all binaries from `/home/kun/onebinary/`** (recommended)

```bash
# Process all binaries at once
python generate_bb_bucket_data.py \
    --binary_dir /home/kun/onebinary/ \
    --output_dir data \
    --num_buckets 10

# Or use the convenient script:
./run_generate_data.sh
```

This creates:
- `data/<binary_name>_instructions.txt`: One instruction per line for each binary
- `data/<binary_name>_instructions_labels.json`: Labels and metadata
- `data/all_metadata.json`: Combined metadata for all binaries

**Option B: Process a single binary**

```bash
python generate_bb_bucket_data.py \
    --binary /path/to/binary \
    --output data/instructions.txt \
    --metadata data/metadata.json \
    --num_buckets 10
```

### Step 2: Run Probe

Evaluate both models using the probe:

```bash
python bb_bucket_probe.py \
    --data_dir data \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_model ../addressaware/output_addressaware_new/best_bert.pt \
    --baseline_model ../addressaware/output_baseline_new/best_bert.pt \
    --output probe_results \
    --batch_size 32 \
    --seq_len 512
```

## Output

The probe generates:

1. **addressaware_results.json**: Metrics for address-aware model
   - Train/test accuracy
   - Classification report (precision, recall, F1)
   - Confusion matrix

2. **baseline_results.json**: Metrics for baseline model
   - Same metrics as above

3. **comparison.json**: Direct comparison
   - Accuracy for both models
   - Improvement (address-aware - baseline)

## Interpretation

- **Higher accuracy** = model better encodes positional information within BBs
- **Positive improvement** = address-aware embeddings capture more fine-grained position info
- **Per-bucket metrics** = shows which positions within BBs are easier/harder to identify

## Data Format

### Instruction Format (bb_instructions.txt)

```
opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2 ...
```

Example:
```
mov(0x401000:0.10:0.00:0.00) rax rbx
add(0x401004:0.10:0.05:0.10) rax 0x1
push(0x401008:0.10:0.10:0.20) rbp
```

Where:
- `0xADDR`: Instruction address
- `bnorm`: Binary-level normalized position [0, 1]
- `fnorm`: Function-level normalized position [0, 1]
- `bbnorm`: Basic block-level normalized position [0, 1]

### Labels Format (bb_instructions_labels.json)

```json
{
  "instructions": [
    {
      "instruction": "mov(0x401000:0.10:0.00:0.00) rax rbx",
      "bb_bucket": 0,
      "bb_norm": 0.00,
      "function": "main"
    },
    {
      "instruction": "add(0x401004:0.10:0.05:0.12) rax 0x1",
      "bb_bucket": 1,
      "bb_norm": 0.12,
      "function": "main"
    },
    ...
  ]
}
```

## Requirements

- Binary Ninja (for data generation)
- PyTorch
- scikit-learn
- numpy
- tqdm

## Example Workflow

```bash
# 1. Generate data from multiple binaries
for binary in /path/to/binaries/*; do
    python generate_bb_bucket_data.py \
        --binary "$binary" \
        --output "data/$(basename $binary)_instructions.txt"
done

# 2. Combine all instruction files (optional)
cat data/*_instructions.txt > data/all_bb_instructions.txt
cat data/*_labels.json | jq -s '{instructions: map(.instructions) | add}' > data/all_labels.json

# 3. Run probe
python bb_bucket_probe.py \
    --data data/all_bb_instructions.txt \
    --labels data/all_labels.json \
    --vocab ../data/vocab.json \
    --addressaware_model ../addressaware/output_addressaware/best_bert.pt \
    --baseline_model ../addressaware/output_baseline/best_bert.pt \
    --output results/bb_bucket_probe/ \
    --cuda

# 4. View results
cat results/bb_bucket_probe/comparison.json
```

## Notes

- Instructions are encoded one-by-one (no pairing like CFG training)
- Each instruction is labeled with its position bucket based on `bb_norm`
- Instructions at the beginning of a BB (bb_norm ~0.0) get bucket 0
- Instructions at the end of a BB (bb_norm ~1.0) get bucket 9
- The probe uses [CLS] token embedding to represent each instruction
- Logistic regression is used as the probe (linear classifier)
- Results show how much positional information is captured in embeddings
