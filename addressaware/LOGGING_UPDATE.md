# Logging Updates

## Changes Made

Updated the training script to save all logs to the `log` folder with proper timestamps.

### Files Modified

1. **`train.py`**
   - Added `logging` module import
   - Added `--log_dir` argument (default: `./log`)
   - Configured dual logging: both to file and stdout
   - Log file format: `log/train_YYYYMMDD_HHMMSS.log`
   - All `print()` statements replaced with `logger.info()`
   - Progress bars (tqdm) still show in terminal, detailed logs go to file

2. **`train_addressaware.sh`**
   - Added `LOG_DIR="log"` variable
   - Added `mkdir -p ${LOG_DIR}` to create log directory
   - Added `--log_dir "${LOG_DIR}"` to training command

## Log File Structure

```
log/
├── train_20251120_143052.log  # Training session 1
├── train_20251120_151234.log  # Training session 2
└── ...
```

## Log Contents

Each log file contains:
- Timestamp for each log entry
- Device information (CPU/GPU)
- Model configuration (parameters count, etc.)
- Training progress (batch-level metrics every N batches)
- Epoch-level metrics (train and validation)
- Checkpoint saves
- Best model saves

## Example Log Output

```
2025-11-20 14:30:52,123 - INFO - Logging to: log/train_20251120_143052.log
2025-11-20 14:30:52,124 - INFO - Using device: cuda
2025-11-20 14:30:52,125 - INFO - Arguments saved to: output_addressaware_new/args.json
2025-11-20 14:30:53,456 - INFO - Loading vocabulary from ../pre-trained_model/palmtree/vocab
2025-11-20 14:30:53,789 - INFO - Vocabulary size: 5000
2025-11-20 14:30:54,012 - INFO - Creating training dataset...
2025-11-20 14:31:02,345 - INFO - Creating model...
2025-11-20 14:31:03,678 - INFO - Using 2 GPUs
2025-11-20 14:31:03,679 - INFO - Total parameters: 4,301,939
2025-11-20 14:31:03,680 - INFO - Starting training...
2025-11-20 14:31:03,681 - INFO - ================================================================================
2025-11-20 14:31:03,682 - INFO - 
2025-11-20 14:31:03,683 - INFO - Epoch 1/20
2025-11-20 14:31:03,684 - INFO - --------------------------------------------------------------------------------
2025-11-20 14:31:45,123 - INFO - Batch 50/6497 - Loss: 2.9792 | MLM: 2.4191 | NSP_CFG: 0.3417 | NSP_DFG: 0.2184 | DIR: 0.0000(50.00%) | SCOPE: 0.0000(33.33%, n=150)
...
```

## Terminal Output

Terminal still shows:
- tqdm progress bars (interactive, real-time)
- Epoch summaries
- Best model saves
- Important milestones

All detailed batch-level logs go to the log file only.

## Viewing Logs

### During Training
```bash
# Follow the latest log in real-time
tail -f log/train_*.log | grep -E "(Epoch|Loss|Saved)"

# Follow with full output
tail -f log/train_*.log
```

### After Training
```bash
# View specific log
less log/train_20251120_143052.log

# Search for best model saves
grep "Saved best model" log/train_*.log

# Extract epoch summaries
grep "Train Loss:" log/train_*.log
grep "Val Loss:" log/train_*.log
```

## Benefits

1. **Persistent Records**: All training details saved to disk
2. **Timestamped**: Easy to identify different training runs
3. **Searchable**: Use grep/awk to extract specific metrics
4. **Clean Terminal**: Progress bars still visible, but clutter reduced
5. **Debugging**: Full log available for troubleshooting

## Usage

Just run the training script as before:
```bash
bash train_addressaware.sh
```

Logs will automatically be saved to `log/train_YYYYMMDD_HHMMSS.log`
