# strupos - Training from Scratch

**Same architecture as addressaware, but train ALL components from scratch (no PalmTree pre-training).**

## What Changed?

### addressaware (uses pre-trained PalmTree)
```python
# Load PalmTree weights
bert = AddressAwareBERT(
    vocab_size=len(vocab),
    pretrained_token_emb=palmtree_weights['token'],      # ← FROZEN
    pretrained_position_emb=palmtree_weights['position'], # ← FROZEN
    pretrained_transformer=palmtree_weights['transformer'] # ← FROZEN
)
```

### strupos (train from scratch)
```python
# NO pre-trained weights
bert = AddressAwareBERT(
    vocab_size=len(vocab),
    # All components initialized randomly and trained
)
```

## Files

All files copied from `addressaware/` with pre-trained loading removed:

- `model.py` - AddressAwareBERT (no pretrained args)
- `address_embedding.py` - AddressAwareBERTEmbedding (no pretrained args)  
- `dataloader_paired.py` - Original addressaware dataloader (splits line in half for NSP)
- `dataloader_consecutive.py` - **NEW**: Consecutive pair NSP (1-2, 2-3, ..., 7-8)
- `dataloader_scope.py` - Same as addressaware
- `train_from_scratch.py` - Training script (no --palmtree_checkpoint)
- `config.py` - Simple config file
- `run_training.sh` - Training launcher (original half-split NSP)
- `run_consecutive.sh` - **NEW**: Training with consecutive pair NSP

## Training Strategy

### NSP (Next Sequence Prediction) Approach

Each CFG/DFG file has lines with **8 consecutive instructions** (separated by tabs).

**Two NSP strategies available:**

#### 1. Half-Split NSP (original addressaware approach)
- MLM: All 8 instructions
- NSP: First 4 instructions vs. Last 4 instructions
- Use: `./run_training.sh`

#### 2. Consecutive Pair NSP (NEW - recommended)
- MLM: All 8 instructions  
- NSP: Random consecutive pair from (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8)
- Use: `./run_consecutive.sh` ← **This is what you want!**

## Quick Start

**For consecutive pair NSP (recommended):**
```bash
chmod +x run_consecutive.sh
./run_consecutive.sh
```

**Or original half-split NSP:**
```bash
./run_training.sh
```

**That's it!** Same code as addressaware, just:
1. Removed the pre-trained PalmTree loading
2. Added consecutive pair NSP option
