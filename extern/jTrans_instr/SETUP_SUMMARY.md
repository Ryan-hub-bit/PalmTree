# jTrans_instr Complete Setup

✅ **STATUS**: Ready for pretraining

## Quick Start

```bash
# 1. Generate data (from binaries)
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
bash generate_baseline.sh  # IDA Pro extraction
bash generate_text.sh      # Convert to text
bash build_vocab.sh        # Build vocabulary

# 2. Train model
cd /home/kun/Document/AAE/extern/jTrans_instr/pretrain
# Edit run_instr_pretrain.sh to set GPU and paths
bash run_instr_pretrain.sh
```

## Files Created

### Data Pipeline (`datautils/`)
✅ `data.py` - Modified for instruction-level addressing
✅ `convert_pkl_to_text.py` - Converts to `instr_addr_{i}` format
✅ `build_vocab.py` - Builds vocabulary from data
✅ `generate_baseline.sh` - IDA Pro extraction script
✅ `generate_text.sh` - Text conversion script
✅ `build_vocab.sh` - Vocabulary building script

### Model (`pretrain/`)
✅ `model_instr.py` - Model with instruction embeddings + special instr_addr handling
✅ `dataloader_instr.py` - Dataloader that generates instruction_ids
✅ `train_instr.py` - Training script (MLM + JTP)
✅ `run_instr_pretrain.sh` - Shell script to run training
✅ `README.md` - Pretraining documentation

### Documentation
✅ `ARCHITECTURE.md` - Detailed architecture explanation
✅ `SETUP_SUMMARY.md` - This file

## Architecture Highlights

### Embeddings
```
Final = Token Embedding + Position Embedding + Instruction Embedding
```

**Special Feature**: `instr_addr_{i}` tokens use `instruction_embedding[i]` directly!

### Why This Matters
```assembly
# Instruction 5:
mov eax, ebx    <- Tokens use instruction_embedding[5]

# Instruction 10:
jmp instr_addr_5  <- This token ALSO uses instruction_embedding[5]
```

Result: Jump targets semantically link to actual instructions! 🎯

## Training Tasks

1. **MLM (Masked Language Modeling)**
   - Mask 15% of tokens
   - Predict from vocabulary (~2400 tokens)

2. **JTP (Jump Target Prediction)**
   - Mask 20% of `instr_addr` tokens
   - Predict instruction index (0-200)

## Model Usage

```python
from model_instr import create_instr_model
from transformers import BertTokenizer

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("jtrans_tokenizer")

# Create model
model = create_instr_model(
    vocab_size=len(tokenizer),
    max_instructions=201
)

# CRITICAL: Set instr_addr mappings
model.set_instr_addr_token_ids(tokenizer)

# Forward pass
mlm_logits, jtp_logits = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    instruction_ids=instruction_ids  # Maps tokens to instructions
)
```

## Key Implementation Details

### 1. Instruction Boundary Detection (`dataloader_instr.py`)
- Detects assembly mnemonics (mov, add, jmp, etc.)
- Each new mnemonic starts a new instruction
- Generates `instruction_ids`: `[0,0,0,1,1,2,2,2,...]`

### 2. Special Token Handling (`model_instr.py`)
```python
# After getting token embeddings:
for i in range(max_instructions):
    mask = (input_ids == token_id_of_instr_addr_i)
    token_emb[mask] = instruction_embeddings[i]  # Replace!
```

### 3. Data Format (`instr_pretrain.txt`)
```
[CLS] push ebp mov ebp esp call instr_addr_5 ... [SEP]
```

## Differences from Baseline jTrans

| Aspect | Baseline | jTrans_instr |
|--------|----------|--------------|
| Position emb | position=word | Standard BERT |
| Jump format | JUMP_ADDR_X | instr_addr_{i} |
| Jump target | Token position | Instruction index |
| Instruction info | Implicit | Explicit embeddings |
| Token type | Yes | No (removed) |

## Next Steps

### Immediate
1. Test data generation on small dataset
2. Verify vocabulary generation
3. Test model forward pass

### Short-term
1. Run full pretraining
2. Monitor MLM and JTP accuracy
3. Save checkpoints

### Long-term
1. Fine-tune on downstream tasks
2. Compare with baseline jTrans
3. Analyze instruction embeddings
4. Publish results

## Troubleshooting

**Q**: How do I know if `instr_addr` tokens are working?
**A**: Check JTP accuracy during training. Should be > random (1/201 = 0.5%)

**Q**: What if I get import errors?
**A**: Run from pretrain directory: `cd pretrain && python train_instr.py ...`

**Q**: GPU out of memory?
**A**: Reduce `batch_size` or `max_len` in `run_instr_pretrain.sh`

## Contact

See `ARCHITECTURE.md` for detailed architecture explanation.
See `pretrain/README.md` for pretraining guide.
