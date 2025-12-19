# VS Code Debugging Guide for PalmTree Comparison Training

## Available Debug Configurations

You now have 3 debug configurations available in VS Code:

### 1. **Debug train.py** 
Original strupos training with address features
- Location: `strupos/train.py`
- Data: `/data/kun/dataset/`

### 2. **Debug BASELINE Training (No Address Features)** 🆕
Baseline comparison mode - standard BERT without address embeddings
- Location: `extern/PalmTree/src/train_comparison.py`
- Mode: `baseline`
- Features: Positions are MASKED (set to 0.0)
- Model: Standard BERT architecture

### 3. **Debug ADDRESS-AWARE Training (With Address Features)** 🆕
Address-aware comparison mode - your strupos version
- Location: `extern/PalmTree/src/train_comparison.py`
- Mode: `address_aware`
- Features: Positions are USED for embeddings
- Model: AddressAwareBERT with custom layers

## How to Debug

### Option 1: Using Debug Panel (Recommended)

1. **Open VS Code Debug Panel**
   - Press `Ctrl+Shift+D` (or `Cmd+Shift+D` on Mac)
   - Or click the Debug icon in the left sidebar (bug icon)

2. **Select Configuration**
   - At the top of the debug panel, you'll see a dropdown menu
   - Select either:
     - "Debug BASELINE Training (No Address Features)"
     - "Debug ADDRESS-AWARE Training (With Address Features)"

3. **Set Breakpoints**
   - Click in the gutter (left of line numbers) to set breakpoints
   - Suggested locations:
     - `train_comparison.py:main()` - Start of execution
     - `train_comparison.py:setup_baseline_mode()` - Baseline setup
     - `train_comparison.py:setup_address_aware_mode()` - Address-aware setup
     - Dataset `__getitem__()` - Data loading
     - Trainer `train()` - Training loop
     - Model `forward()` - Model execution

4. **Start Debugging**
   - Press `F5` or click the green play button
   - Your program will start and pause at breakpoints

### Option 2: Using Command Palette

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Debug: Select and Start Debugging"
3. Choose your configuration

### Option 3: Using Run Menu

1. Click "Run" in the top menu
2. Select "Start Debugging"
3. Choose your configuration

## Debug Configuration Details

### Baseline Mode Settings
```json
{
  "mode": "baseline",
  "data_percentage": "0.1",     // 10% of data for quick testing
  "batch_size": "16",
  "num_epochs": "20",
  "num_workers": "2",
  "log_freq": "10",             // Log every 10 batches
  "justMyCode": false           // Can debug into libraries
}
```

### Address-Aware Mode Settings
```json
{
  "mode": "address_aware",
  "address_embed_dim": "64",
  "var_embed_dim": "32",
  "data_percentage": "0.1",     // 10% of data for quick testing
  "batch_size": "16",
  "num_epochs": "20",
  "num_workers": "2",
  "log_freq": "10",
  "justMyCode": false
}
```

## Debugging Tips

### 1. Quick Test with Small Data
Both configs use `data_percentage=0.1` (10% of data) by default for fast debugging.

To use full data, modify in `.vscode/launch.json`:
```json
"--data_percentage", "1.0"
```

### 2. Reduce Epochs for Quick Testing
Default is 20 epochs. For quick testing:
```json
"--num_epochs", "2"
```

### 3. Check Variable Values
When paused at a breakpoint:
- Hover over variables to see values
- Use Debug Console (bottom panel) to evaluate expressions
- Check "Variables" panel (left side) for all local/global variables

### 4. Step Through Code
- `F10` - Step Over (execute line, don't go into functions)
- `F11` - Step Into (go into function calls)
- `Shift+F11` - Step Out (exit current function)
- `F5` - Continue (run until next breakpoint)

### 5. Useful Breakpoint Locations

**For Dataset Debugging:**
```python
# dataset_baseline.py or dataloader.py
def __getitem__(self, idx):
    # Set breakpoint here to inspect data loading
    
def _preprocess_line(self, line):
    # Check token preprocessing
    
def _parse_instruction_baseline(self, inst_text):
    # Verify instruction parsing
```

**For Model Debugging:**
```python
# model.py
def forward(self, x, ...):
    # Check input shapes and values
    
# Address-aware specific
def forward(self, x, positions, var_offsets):
    # Verify address embeddings
```

**For Training Debugging:**
```python
# train_comparison.py
def setup_baseline_mode(args, vocab):
    # Check baseline setup
    
def setup_address_aware_mode(args, vocab):
    # Check address-aware setup
    
# Trainer
def train(self, epoch):
    # Monitor training progress
```

### 6. Compare Baseline vs Address-Aware

**Strategy:**
1. Debug baseline first - set breakpoint in `setup_baseline_mode()`
2. Inspect data format, model architecture, token processing
3. Note down tensor shapes, token values
4. Then debug address-aware - set breakpoint in `setup_address_aware_mode()`
5. Compare the same inspection points
6. Verify differences: position embeddings, var embeddings, model layers

### 7. Watch Expressions

Add watch expressions in the Watch panel:
```python
len(vocab)
args.mode
sample['bert_input'].shape
model.address_embed_dim  # address-aware only
```

### 8. Debug Console Commands

While paused, type in Debug Console:
```python
# Check tensor
print(batch['bert_input'].shape)
print(batch['bert_label'][:10])

# Check vocab
print(len(vocab))
print(vocab.itos[:20])

# Check model
print(model)
print(list(model.parameters())[0].shape)
```

## Troubleshooting

### Issue: "Module not found"
**Solution:** Check `cwd` is set correctly in launch.json
```json
"cwd": "${workspaceFolder}/extern/PalmTree/src"
```

### Issue: Can't step into library code
**Solution:** Change `justMyCode` to `false`
```json
"justMyCode": false
```

### Issue: Out of memory
**Solution:** Reduce batch size or data percentage
```json
"--batch_size", "8",
"--data_percentage", "0.05"
```

### Issue: Too slow to debug
**Solution:** 
1. Use smaller data: `--data_percentage 0.01`
2. Reduce epochs: `--num_epochs 1`
3. Reduce workers: `--num_workers 0`

## Quick Start

**To debug baseline:**
1. Press `Ctrl+Shift+D`
2. Select "Debug BASELINE Training"
3. Set breakpoint in `train_comparison.py` line with `setup_baseline_mode()`
4. Press `F5`

**To debug address-aware:**
1. Press `Ctrl+Shift+D`
2. Select "Debug ADDRESS-AWARE Training"
3. Set breakpoint in `train_comparison.py` line with `setup_address_aware_mode()`
4. Press `F5`

**To compare both:**
1. Debug baseline, note down values at key points
2. Debug address-aware, compare values at same points
3. Focus on: data format, model inputs, embeddings, loss values

Happy debugging! 🐛🔍
