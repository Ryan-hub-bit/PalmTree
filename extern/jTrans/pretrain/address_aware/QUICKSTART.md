# Address-Aware jTrans - Quick Start Guide

## What Was Created

### 1. Data Generation (`/extern/jTrans/datautils_addressaware/`)
- **convert_pkl_to_addressaware.py** - Converts pickles to address-aware format
- **generate_addressaware_data.sh** - One-command data generation
- **README.md** - Detailed documentation

### 2. Model Training (`/extern/jTrans/pretrain/address_aware/`)
- **vocab.py** - Vocabulary builder (from strupos)
- **address_embedding.py** - Hierarchical positional embeddings (from strupos)
- **dataloader_addressaware.py** - Dataset loader (simplified for jTrans format)
- **model_addressaware.py** - BERT model with address embeddings
- **train_addressaware.py** - Training script
- **run_addressaware_pretrain.sh** - Shell script to run training

## Complete Pipeline

### Step 1: Generate Address-Aware Data

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_addressaware
bash generate_addressaware_data.sh
```

This will:
- Read pickles from `/data/kun/jtransdata/extract/`
- Generate `/data/kun/jtransdata/addressaware_train.txt`
- Generate `/data/kun/jtransdata/addressaware_test.txt`

### Step 2: Create Vocabulary

```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware

python << 'EOFPY'
from vocab import WordVocab

vocab = WordVocab.create_vocab(
    corpus_path='/data/kun/jtransdata/addressaware_train.txt',
    output_path='vocab.pkl',
    min_freq=1
)
print(f"✓ Created vocab with {len(vocab)} tokens")
EOFPY
```

### Step 3: Train Model

```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
bash run_addressaware_pretrain.sh
```

Model will be saved to:
- `/home/kun/Document/AAE/output/addressaware_pretrain/checkpoint_epoch_X/`
- `/home/kun/Document/AAE/output/addressaware_pretrain/best_model/`

## Data Format Explained

### Baseline Format (simple)
```
push rbp mov rbp rsp lea JUMP_ADDR_7 call GLOBAL_VAR
```

### Address-Aware Format (hierarchical)
```
push(0x401000:0.1:0.05:0.0) rbp mov(0x401001:0.11:0.1:0.01) rbp rsp lea(0x401005:0.15:0.2:0.05) rbp address(0x40100c:0.2:0.3:0.1) call(0x401010:0.25:0.35:0.15) daddr(0x405000:0.3:0.4:0.2)
```

**Token types:**
- `opcode(0xADDR:bnorm:fnorm:bbnorm)` - Opcode with hierarchical positions
- `address(0xADDR:bnorm:fnorm:bbnorm)` - Jump target
- `daddr(0xADDR:bnorm:fnorm:bbnorm)` - Data address
- `var(0xOFFSET)` - Stack variable (e.g., `var(0x10)` for [rbp-0x10])
- `imm` - Immediate value (just the token, no brackets)
- `rbp`, `rax`, etc. - Regular operands

**Position values:**
- `bnorm`: [0, 1] position within entire binary
- `fnorm`: [0, 1] position within function
- `bbnorm`: [0, 1] position within basic block

## Key Differences from Baseline

| Aspect | Baseline | Address-Aware |
|--------|----------|---------------|
| **Embedding** | position = word embedding | Hierarchical sin/cos + MLP |
| **Address Info** | Implicit (sequence position) | Explicit (binary/function/BB) |
| **Jump Targets** | JUMP_ADDR_X (index) | address(0xADDR:positions) |
| **Data Access** | GLOBAL_VAR | daddr(0xADDR:positions) |
| **Stack Vars** | LOCAL_VAR | var(0xOFFSET) |
| **Immediates** | imm | imm |

## Expected Results

After training, compare with baseline on:
1. **Pretraining metrics**: MLM accuracy, loss convergence
2. **Downstream tasks**: Function similarity, search, vulnerability detection
3. **Embedding quality**: t-SNE visualization, clustering

**Hypothesis**: Address-aware should better capture:
- Cross-function relationships (binary-level positions)
- Function-internal structure (function-level positions)  
- Control flow patterns (BB-level positions)
- Memory access patterns (daddr vs code addresses)

## Files Overview

```
jTrans/
├── datautils_addressaware/              # Data generation
│   ├── convert_pkl_to_addressaware.py  # Main converter
│   ├── generate_addressaware_data.sh   # One-command script
│   └── README.md                        # Detailed docs
│
└── pretrain/
    ├── baseline/                        # Baseline (position=word)
    │   └── ...
    │
    └── address_aware/                   # Your approach
        ├── vocab.py                     # From strupos
        ├── address_embedding.py         # From strupos
        ├── dataloader_addressaware.py   # Adapted for jTrans
        ├── model_addressaware.py        # BERT + address embeddings
        ├── train_addressaware.py        # Training script
        ├── run_addressaware_pretrain.sh # Shell script
        ├── vocab.pkl                    # Generated in step 2
        ├── README.md                    # Overview
        ├── COMPARISON.md                # Detailed comparison
        └── QUICKSTART.md                # This file
```

## Troubleshooting

### Issue: No pickle files found
**Solution**: Run jTrans extraction first:
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils
python run.py
```

### Issue: Import errors in converter
**Solution**: The script adds parent directory to path automatically

### Issue: Vocabulary loading error
**Solution**: Make sure vocab.pkl exists:
```bash
ls -lh /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.pkl
```

### Issue: CUDA OOM during training
**Solution**: Reduce batch size in `run_addressaware_pretrain.sh`:
```bash
--batch_size 16  # Instead of 32
```

### Issue: GPU not being used
**Solution**: Check CUDA_VISIBLE_DEVICES:
```bash
export CUDA_VISIBLE_DEVICES=1
nvidia-smi  # Verify GPU 1 has free memory
```

## Next Steps

1. **Generate data** (Step 1 above)
2. **Create vocab** (Step 2 above)
3. **Start training** (Step 3 above)
4. **Monitor progress**:
   ```bash
   tail -f /home/kun/Document/AAE/output/addressaware_pretrain/*.log
   ```
5. **Compare with baseline** after training completes

## Questions?

- Check `/extern/jTrans/datautils_addressaware/README.md` for data generation details
- Check `/extern/jTrans/pretrain/address_aware/COMPARISON.md` for architecture comparison
- Check `/home/kun/Document/AAE/strupos/` for original address-aware implementation
- Check `/home/kun/Document/AAE/extern/PalmTree/src/address_aware/` for PalmTree's approach
