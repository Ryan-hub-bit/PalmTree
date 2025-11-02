# Training Strategies for Large Dataset (21GB all_bb_pairs.txt)

## Overview
You have a **21GB** BB pairs file - way too large to load into memory. Here are 4 strategies:

---

## Strategy 1: Streaming DataLoader ⭐ (RECOMMENDED)
**File**: `streaming_data_loader.py`

### How it works:
- Reads file line by line during training
- Uses shuffle buffer (e.g., 10K samples) for randomness
- Never loads full file into memory
- Supports multi-worker data loading

### Usage:
```python
from streaming_data_loader import create_streaming_dataloader

# Full file training
dataloader = create_streaming_dataloader(
    bb_pairs_file='../all_bb_pairs.txt',
    vocab_file='../pre-trained_model/palmtree/vocab',
    batch_size=8,
    num_workers=4,  # Parallel loading
    shuffle_buffer_size=50000,  # Larger = more random
    epoch_size=None  # None = full file each epoch
)

# Train as usual
for epoch in range(10):
    for batch in dataloader:
        # train...
```

### Pros:
- ✅ Handles unlimited file size
- ✅ Constant memory usage
- ✅ Can use full 21GB dataset
- ✅ Parallel loading with workers

### Cons:
- ⚠️ Shuffling is approximate (buffer-based, not perfect)
- ⚠️ Can't calculate exact dataset size upfront

---

## Strategy 2: Subset Training (Quick Start)
Use only part of the dataset

### Create subset:
```bash
# Take first 1M lines (~1GB)
head -n 1000000 all_bb_pairs.txt > subset_1M.txt

# Or random sample
shuf all_bb_pairs.txt | head -n 1000000 > random_subset_1M.txt
```

### Usage:
```python
# Use existing data_loader.py with subset
from data_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    bb_pairs_file='subset_1M.txt',  # Small subset
    vocab_file='../pre-trained_model/palmtree/vocab',
    batch_size=8
)
```

### Pros:
- ✅ Fast to start
- ✅ Perfect shuffling
- ✅ Known dataset size
- ✅ Use existing code

### Cons:
- ❌ Only using small portion of data
- ❌ May underfit (not seeing full diversity)

---

## Strategy 3: Chunked Training
Split file into chunks, train on each chunk

### Split file:
```bash
# Split into 1GB chunks
split -b 1G -d all_bb_pairs.txt chunk_

# Results: chunk_00, chunk_01, chunk_02, ...
```

### Usage:
```python
import glob

chunks = sorted(glob.glob('chunk_*'))

for epoch in range(10):
    for chunk_file in chunks:
        # Load one chunk
        train_loader, _, _ = create_dataloaders(
            bb_pairs_file=chunk_file,
            vocab_file='../pre-trained_model/palmtree/vocab',
            batch_size=8,
            max_pairs=None
        )
        
        # Train on this chunk
        for batch in train_loader:
            # train...
```

### Pros:
- ✅ Uses full dataset
- ✅ Good shuffling within chunks
- ✅ Memory efficient

### Cons:
- ⚠️ Samples from same chunk are correlated
- ⚠️ Need to shuffle chunks between epochs

---

## Strategy 4: Pre-processed Binary Format
Convert to efficient binary format (HDF5, PyTorch tensors)

### Preprocess:
```python
# Convert to binary format (one-time cost)
python preprocess_to_binary.py \
    --input all_bb_pairs.txt \
    --output processed_data.h5 \
    --vocab ../pre-trained_model/palmtree/vocab
```

### Then load efficiently:
```python
import h5py

with h5py.File('processed_data.h5', 'r') as f:
    input_ids = f['input_ids'][:]  # Memory mapped
    # Fast random access
```

### Pros:
- ✅ Very fast loading
- ✅ Perfect shuffling
- ✅ Random access

### Cons:
- ❌ Takes time to preprocess
- ❌ Need disk space (2x: text + binary)
- ❌ More complex code

---

## Recommendation for Your Use Case

### For Quick Experimentation:
**Use Strategy 2 (Subset)**
```bash
# Create 1M sample subset
head -n 1000000 all_bb_pairs.txt > data/subset_1M.txt

# Train
python train.py \
    --bb_pairs_file data/subset_1M.txt \
    --vocab_file ../pre-trained_model/palmtree/vocab
```

### For Full Dataset Training:
**Use Strategy 1 (Streaming)** ⭐
```python
# In train.py, replace create_dataloaders with:
from streaming_data_loader import create_streaming_dataloader

train_loader = create_streaming_dataloader(
    bb_pairs_file='../all_bb_pairs.txt',
    vocab_file=args.vocab_file,
    batch_size=config.BATCH_SIZE,
    num_workers=4,
    shuffle_buffer_size=50000,
    epoch_size=500000  # 500K samples per epoch (adjust as needed)
)
```

---

## Hybrid Approach (BEST)

1. **Start with subset** (fast iteration, debug model)
   ```bash
   head -n 100000 all_bb_pairs.txt > data/debug_100k.txt
   ```

2. **Scale to 1M** (validate approach)
   ```bash
   head -n 1000000 all_bb_pairs.txt > data/train_1M.txt
   ```

3. **Full training with streaming** (final model)
   ```python
   # Use streaming_data_loader for full 21GB
   ```

---

## Memory Estimates

| Strategy | Memory Usage | Training Time | Data Coverage |
|----------|--------------|---------------|---------------|
| Streaming | **~500MB** | Normal | **100%** (21GB) |
| Subset (1M) | ~2GB | Fast | ~5% |
| Chunked | ~2GB/chunk | Normal | **100%** |
| Binary | ~10GB | **Fast** | **100%** |

---

## Quick Start Command

```bash
# Option A: Quick test with subset
head -n 100000 all_bb_pairs.txt > data/test_100k.txt
python train.py --bb_pairs_file data/test_100k.txt --vocab_file ../pre-trained_model/palmtree/vocab

# Option B: Full training with streaming (edit train.py first)
python train.py --bb_pairs_file ../all_bb_pairs.txt --vocab_file ../pre-trained_model/palmtree/vocab --streaming
```

---

## Which Should You Use?

- **Debugging model**: Subset (100K-1M lines)
- **Proof of concept**: Subset (1-5M lines)  
- **Production training**: Streaming (full 21GB)
- **Maximum performance**: Binary preprocessing

For now, I recommend **starting with Strategy 2 (subset)** to validate the training tasks work, then switch to **Strategy 1 (streaming)** for full training!
