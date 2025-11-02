# Ready to Train! ✅

## What We've Built

### 3-Level Embedding System
1. **Level 1: Semantic** - Pre-trained PalmTree token embeddings for instructions
2. **Level 2: Address** - Address type embeddings + sinusoidal value encodings  
3. **Level 3: Positional** - Position in sequence

### Dual Vocabulary System
- **PalmTree Vocab** (6,631 tokens): Pre-trained instruction tokens (mov, rax, etc.)
- **Address Vocab** (9 tokens): Address-specific tokens kept separate
  - `<addr_start>`, `<addr_end>`
  - `<addr_code>`, `<addr_data>`  
  - `<seq>` (instruction separator)
  - etc.

### Key Features
- ✅ Commas removed from assembly (via updated `bb_flow.py`)
- ✅ `<seq>` separators between instructions
- ✅ Address tokens treated separately from pre-trained vocab
- ✅ Multi-task learning (next BB prediction, address type, edge type)

## How to Train

```bash
cd /home/louie/PalmTree/bb_pretrain
conda activate palmtree

# Full training
python train.py \
    --bb_pairs_file /home/louie/PalmTree/all_bb_pairs.txt \
    --vocab_file /home/louie/PalmTree/pre-trained_model/palmtree/vocab
```

## Training Configuration

From `config.py`:
- Batch size: 32
- Learning rate: 2e-5
- Epochs: 10
- Max sequence length: 512
- Hidden size: 768
- Address encoding dim: 128

## Outputs

Training artifacts saved to `bb_pretrain/output/`:
- `best_model.pt` - Best model checkpoint
- `logs/` - TensorBoard logs

## Monitor Training

```bash
tensorboard --logdir bb_pretrain/output/logs
```

## Architecture Flow

```
Input Token: "mov rax qword [rbp-0x10]"
     ↓
[DUAL VOCABULARY TOKENIZATION]
     ├─→ PalmTree ID (for semantic embedding)
     └─→ Address ID (to identify if address token)
     ↓
[3-LEVEL EMBEDDINGS]
     ├─→ Level 1: Semantic (from PalmTree)
     ├─→ Level 2: Address type + sin/cos value
     └─→ Level 3: Position in sequence
     ↓
[FUSION NETWORK]
     Concat → Linear → LayerNorm → GELU → Linear
     ↓
[MULTI-TASK HEADS]
     ├─→ Next BB prediction
     ├─→ Address type classification
     └─→ Edge type classification
```

## Next Steps After Training

1. Evaluate on test set
2. Visualize embeddings
3. Test on downstream tasks (function similarity, malware detection, etc.)
