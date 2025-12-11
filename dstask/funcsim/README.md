# Function Similarity Fine-tuning

This directory contains code for fine-tuning AddressAwareBERT for function similarity tasks using contrastive learning.

## Quick Start

### 1. Train and Evaluate (Full Pipeline)

```bash
# Run full pipeline (train + evaluate)
cd /home/kun/Document/PalmTree/dstask/funcsim
./run_funcsim_pipeline.sh

# Or skip training (use existing model)
./run_funcsim_pipeline.sh --skip-train

# Or only train
./run_funcsim_pipeline.sh --skip-eval
```

### 2. Individual Steps

```bash
# Train only
./run_funcsim_train.sh

# Evaluate only (after training)
./run_eval.sh
```

## Configuration

**Pre-trained Model:** `output/mlm/best_bert.pt` (MLM pre-trained BERT)
- This model was trained on masked language modeling task
- Contains learned instruction embeddings

**Data Files:**
- `/data/kun/funcsim_match/function_blocks.json` (9.4GB)
- `/data/kun/funcsim_match/funcsim_pairs.json` (220MB)
- Vocabulary: `strupos/vocab.pkl`

**Model Hyperparameters:**
- Hidden size: 128
- Layers: 12
- Attention heads: 8
- Function embedding dim: 256

**Training Hyperparameters:**
- Batch size: 32
- Epochs: 20
- Learning rate: 1e-4
- Contrastive loss margin: 1.0
- Train/Val/Test split: 72%/8%/20%

## Overview

The function similarity task trains the model to:
- **Recognize** that the same function compiled at different optimization levels (O0, O1, O2, O3) are similar
- **Distinguish** between different functions

This is useful for:
- Binary code similarity analysis
- Vulnerability detection across compiler versions
- Code clone detection in binaries

## Data Format

The training uses two JSON files generated from the funcsim dataset:

### 1. function_blocks.json
Maps function block IDs to instruction sequences:
```json
{
  "1": ["mov(...) rax rbx", "add(...) rax 0x10", ...],
  "2": [...],
  "3": [...],
  "4": [...]
}
```

### 2. funcsim_pairs.json
Ground truth similarity pairs:
```json
{
  "1": {
    "function_id": "1",
    "opt_level": "O0",
    "ground_truth": ["2", "3", "4"]  // Same function at O1, O2, O3
  },
  "2": {
    "function_id": "2",
    "opt_level": "O1",
    "ground_truth": ["1", "3", "4"]  // Same function at O0, O2, O3
  }
}
```

## Model Architecture

```
Input: Function Instructions
    ↓
AddressAwareBERT Encoder (pre-trained)
    ↓
Mean Pooling (over sequence)
    ↓
Projection Layer (768 → 256)
    ↓
L2 Normalization
    ↓
Function Embedding (256-dim)
```

## Training

### 1. Prepare Data
First, generate the function blocks and pairs from funcsim dataset:
```bash
cd ../../src/data_generator
./combine_funcsim.sh      # Deduplicate and combine funcsim data
./generate_pairs.sh        # Generate function blocks and pairs
```

This creates:
- `/data/kun/funcsim_match/combined_deduplicated.json`
- `/data/kun/funcsim_match/function_blocks.json`
- `/data/kun/funcsim_match/funcsim_pairs.json`

### 2. Train Function Similarity Model
```bash
cd /home/kun/Document/PalmTree/dstask/funcsim
./run_funcsim_train.sh
```

The training script will:
- Split data: **80% for training, 20% for testing** (held-out)
- Further split training: **90% train, 10% validation**
- Save test indices to `output/funcsim/test_indices.json`
- Train model and save best checkpoint

### 3. Evaluate on Test Set
After training, evaluate on the held-out 20% test set:
```bash
./run_eval.sh
```

This will:
- Load the best trained model
- Evaluate on the 20% held-out test set
- Report accuracy and similarity metrics
- Save results to `output/funcsim/test_results.json`

### Configuration
Edit `run_funcsim_train.sh` to customize:
- `PRETRAINED_BERT`: Path to pre-trained BERT checkpoint
- `BATCH_SIZE`: Batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 20)
- `LR`: Learning rate (default: 1e-4)
- `NEGATIVE_SAMPLES`: Negative samples per positive (default: 3)
- `EMBEDDING_DIM`: Function embedding dimension (default: 256)
- `TRAIN_SPLIT`: Training split ratio (default: 0.8 = 80%)
- `VAL_SPLIT`: Validation split from training set (default: 0.1 = 10%)

### Data Split
- **Training**: 72% of total data (80% × 90%)
- **Validation**: 8% of total data (80% × 10%)
- **Test**: 20% of total data (held-out for final evaluation)

## Training Strategy

### Contrastive Learning
- **Positive pairs**: Same function at different optimization levels
- **Negative pairs**: Different functions (randomly sampled)
- **Loss**: Contrastive loss with margin
  - Pulls positive pairs closer
  - Pushes negative pairs apart (up to margin)

### Example
For function `func_1` with optimization levels O0, O1, O2, O3:
- Positive pairs: (O0, O1), (O0, O2), (O0, O3), (O1, O2), etc.
- Negative pairs: (O0, other_function), (O1, other_function), etc.

## Output

Training produces:
- `../../output/funcsim/best_model.pt` - Best model checkpoint
- `../../output/funcsim/checkpoint_latest.pt` - Latest checkpoint
- `../../output/funcsim/args.json` - Training arguments
- `../../log/funcsim/train_*.log` - Training logs

## Evaluation Metrics

### Pairwise Metrics (from training pairs)
- **Accuracy**: Percentage of correct similarity predictions
  - Positive pairs should have similarity > 0.5
  - Negative pairs should have similarity < 0.5
- **Avg Positive Similarity**: Average cosine similarity for positive pairs
- **Avg Negative Similarity**: Average cosine similarity for negative pairs
- **Similarity Gap**: Difference between positive and negative similarities

### Retrieval Metrics (ranking-based)
- **Recall@K**: Percentage of queries where at least one ground truth appears in top K results
  - Recall@1: Top-1 accuracy
  - Recall@5: Top-5 accuracy
  - Recall@10: Top-10 accuracy
  - Recall@20: Top-20 accuracy
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank where rank is the position of the first ground truth
  - Higher is better (max = 1.0)
  - Measures how high the first correct match appears in rankings

### Example Interpretation
```
Recall@1: 0.75   → 75% of queries have ground truth as top-1 result
Recall@5: 0.92   → 92% of queries have ground truth in top-5
MRR: 0.85        → On average, first ground truth appears at rank ~1.18
```

## Files

- `model.py` - FunctionSimilarityModel, ContrastiveLoss, TripletLoss
- `dataloader.py` - FunctionSimilarityDataset for loading pairs
- `train.py` - Training script
- `evaluate.py` - Evaluation script for test set
- `run_funcsim_train.sh` - Convenient training launcher
- `run_eval.sh` - Convenient evaluation launcher

## Workflow

```bash
# 1. Prepare data
cd ../../src/data_generator
./combine_funcsim.sh && ./generate_pairs.sh

# 2. Train model (uses 80% of data)
cd ../../dstask/funcsim
./run_funcsim_train.sh

# 3. Evaluate on test set (20% held-out)
./run_eval.sh
```

## Future Work

- [ ] Add triplet loss support
- [ ] Add evaluation on test set
- [ ] Add retrieval metrics (Recall@K, MRR)
- [ ] Support for cross-architecture similarity
- [ ] Online hard negative mining
