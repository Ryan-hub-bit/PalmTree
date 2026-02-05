# Address-Aware jTrans Experimental Variants

## Summary

Three experimental variants have been implemented to test the importance of different components:

1. **Baseline**: MLM + JTP tasks, with full hierarchical positions (binary + function + bb)
2. **No JTP**: MLM only (no jump target prediction), with full hierarchical positions
3. **No binary_pos**: MLM + JTP, but only function + bb positions (no binary-level position)

## Code Changes

### Pretrain
- `train_addressaware.py`: Added `--no_jtp` and `--no_binary_pos` flags, saves flags to checkpoint config
- `model_addressaware.py`: Made JTP head optional, added use_binary_pos parameter
- `address_embedding.py`: Made binary_pos optional in AddressPositionalEmbedding

### Finetune
- `finetune.py`: Reads `use_jtp` and `use_binary_pos` from checkpoint config, creates matching model architecture

### Evaluation
- `evaluate_addressaware_pools.py`: Reads `use_binary_pos` from checkpoint config, creates matching model architecture

## Scripts Created

### Local Testing (small data, 2 epochs)
- `run_baseline_local.sh` - Baseline experiment
- `run_no_jtp_local.sh` - No JTP experiment  
- `run_no_binary_pos_local.sh` - No binary_pos experiment

Usage: `./run_baseline_local.sh [data_ratio]` (default: 0.1 = 10%)

### HPC (full data, 20 epochs)
- `run_baseline_hpc.sh` - Baseline experiment
- `run_no_jtp_hpc.sh` - No JTP experiment
- `run_no_binary_pos_hpc.sh` - No binary_pos experiment

Usage: `./run_baseline_hpc.sh [data_ratio]` (default: 1.0 = 100%)

### SLURM (submits to cluster)
- `run_baseline_hpc.slurm` - Baseline experiment
- `run_no_jtp_hpc.slurm` - No JTP experiment
- `run_no_binary_pos_hpc.slurm` - No binary_pos experiment

Usage: `sbatch run_baseline_hpc.slurm`

## Workflow

1. **Pretrain** (choose one variant):
   - Local: `./run_baseline_local.sh 0.1`
   - HPC: `./run_baseline_hpc.sh 1.0`
   - SLURM: `sbatch run_baseline_hpc.slurm`

2. **Finetune** (automatically adapts to checkpoint config):
   ```bash
   python ../../../finetune.py \
       --model_path /path/to/checkpoint_epoch_X \
       --data_path /path/to/funcsim_data.json \
       --target_opt O3 \
       --batch_size 32 \
       --epochs 20
   ```

3. **Evaluate** (automatically adapts to checkpoint config):
   ```bash
   python ../../../evaluate_addressaware_pools.py \
       --checkpoint /path/to/finetuned_checkpoint \
       --func_blocks /path/to/func_blocks.json \
       --pool pool_O0_vs_O3_1000.json
   ```

## Expected Behavior

- **Finetune and evaluation scripts automatically detect** experimental flags from checkpoint config
- No manual configuration needed - the model architecture matches the pretrained checkpoint
- All 3 variants use the same finetune/evaluation scripts

## Results to Compare

Compare these metrics across the 3 variants:
- MRR (Mean Reciprocal Rank)
- Recall@1, Recall@5, Recall@10

This will show:
- Does JTP task help function similarity? (baseline vs no_jtp)
- Is binary-level position needed? (baseline vs no_binary_pos)
