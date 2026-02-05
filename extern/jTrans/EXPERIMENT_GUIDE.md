# Address-Aware jTrans Experimental Variations

This guide explains how to run 3 experimental variations to test:
1. **MLM-only** (no JTP task) - Does removing JTP improve downstream performance?
2. **No binary_pos** (only function_pos + bb_pos) - Is binary-level position redundant?
3. **Baseline** (MLM+JTP, all 3 positions) - Full model for comparison

## Quick Start

### 1. Pretrain (3 variations)

Edit paths in `pretrain/address_aware/run_experiments.sh`:
```bash
TRAIN_DATA="/path/to/corpus_train.txt"  # Update this
```

Then run all experiments:
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
chmod +x run_experiments.sh
./run_experiments.sh
```

This will create:
- `output/addressaware_experiments/exp1_mlm_only/` - MLM-only model
- `output/addressaware_experiments/exp2_no_binary_pos/` - No binary position
- `output/addressaware_experiments/exp3_baseline/` - Full baseline

### 2. Finetune (each variation)

For **Experiment 1 (MLM-only)**:
```bash
cd /home/kun/Document/AAE/extern/jTrans

# Use finetune.py with the pretrained checkpoint
python finetune.py \
    --model_path output/addressaware_experiments/exp1_mlm_only/checkpoint_epoch_40 \
    --train_data /path/to/funcsim_train.json \
    --vocab_path /home/kun/Document/AAE/strupos/vocab.py \
    --output_dir output/funcsim/exp1_mlm_only \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 5e-5
```

Repeat for exp2 and exp3.

### 3. Evaluate (each variation)

```bash
cd /home/kun/Document/AAE/extern/jTrans

# Evaluate exp1
python evaluate_addressaware_pools.py \
    --checkpoint output/funcsim/exp1_mlm_only/checkpoint_epoch_20 \
    --func_blocks /path/to/func_blocks.json \
    --pool_file /path/to/pool_O0_vs_O3_100.json \
    --vocab_path /home/kun/Document/AAE/strupos/vocab.py \
    --output results_exp1_mlm_only.json

# Repeat for exp2 and exp3
```

### 4. Compare Results

| Experiment | MLM Loss | JTP Loss | MRR | Recall@1 | Recall@5 | Recall@10 |
|------------|----------|----------|-----|----------|----------|-----------|
| Exp1 (MLM-only) | ? | N/A | ? | ? | ? | ? |
| Exp2 (No binary_pos) | ? | ? | ? | ? | ? | ? |
| Exp3 (Baseline) | ? | ? | ? | ? | ? | ? |

## Implementation Details

### Experiment 1: MLM-only (no JTP)

**Modification**: `--no_jtp` flag
- Removes JTP prediction head during pretraining
- Only trains on masked language modeling
- Hypothesis: JTP may be too noisy or irrelevant for function similarity

**Code changes**:
- `model_addressaware.py`: Made `jtp_head` optional
- `train_addressaware.py`: Added `--no_jtp` flag

### Experiment 2: No binary_pos

**Modification**: `--no_binary_pos` flag
- Disables binary-level position embeddings
- Only uses function_pos and bb_pos (2 hierarchical levels instead of 3)
- Hypothesis: Binary-level context may be too coarse-grained

**Code changes**:
- `address_embedding.py`: Made `AddressPositionalEmbedding` skip binary_pos
- Input dimension changes from 48 (3×16) to 32 (2×16) features

### Experiment 3: Baseline (Full model)

**Configuration**: Default (no flags)
- Both MLM and JTP tasks
- All 3 position levels (binary, function, bb)
- Standard address-aware architecture

## Expected Outcomes

### Scenario A: MLM-only performs better
→ JTP task is harmful or distracting
→ Future models should focus only on MLM

### Scenario B: No binary_pos performs better  
→ Binary-level position is redundant or confusing
→ Function and BB positions are sufficient

### Scenario C: Baseline performs best
→ Both JTP and all 3 position levels are beneficial
→ Current design is optimal

## Troubleshooting

### OOM errors
Reduce batch size in run_experiments.sh

### Checkpoints not loading
Verify paths match in finetune scripts

### Evaluation fails
Ensure pool files have 'instructions' field (not just 'tokens')

## Next Steps After Results

1. **If Exp1 wins**: Remove JTP permanently, simplify model
2. **If Exp2 wins**: Remove binary_pos, reduce embedding dimension
3. **If Baseline wins**: Keep current architecture, optimize hyperparameters

## Files Modified

- `pretrain/address_aware/train_addressaware.py` - Added `--no_jtp` and `--no_binary_pos` flags
- `pretrain/address_aware/model_addressaware.py` - Made JTP head optional
- `pretrain/address_aware/address_embedding.py` - Made binary_pos optional in embeddings
