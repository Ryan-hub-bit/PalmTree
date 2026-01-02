# Baseline vs Address-Aware Comparison

## Architecture Comparison

### Baseline jTrans
```python
# Uses simple tokenization
tokens = ['push', 'rbp', 'mov', 'rsp', 'JUMP_ADDR_7']

# Position embedding = word embedding (PalmTree trick)
self.embeddings.position_embeddings = self.embeddings.word_embeddings

# Vocabulary: ~2,902 tokens
# - Regular opcodes (push, mov, etc.)
# - JUMP_ADDR_X tokens (X = target position)
# - GLOBAL_VAR, LOCAL_VAR tokens
```

### Address-Aware jTrans (This Implementation)
```python
# Uses hierarchical address tokenization
tokens = [
    'push(0x1000:0.1:0.05:0.0)',    # opcode with positions
    'rbp',                           # register
    'mov(0x1001:0.11:0.1:0.01)',    # opcode with positions
    'rsp',                           # register
    'address(0x100c:0.2:0.3:0.1)',  # jump target with positions
    'daddr(0x5000:0.3:0.4:0.2)',    # data address
    'var(0x10)',                     # stack variable
    'imm'                      # immediate value
]

# Hierarchical address embedding
class AddressPositionalEmbedding:
    def forward(binary_pos, function_pos, bb_pos):
        # Sin/cos encoding on 3 levels
        binary_enc = sinusoidal(binary_pos, dim)
        function_enc = sinusoidal(function_pos, dim)
        bb_enc = sinusoidal(bb_pos, dim)
        
        # MLP projection
        concat = [binary_enc, function_enc, bb_enc]
        return MLP(concat)  # → d_model dimensions

# Vocabulary: Larger (~3,000-5,000 tokens)
# - Regular opcodes
# - daddr tokens (data addresses)
# - var(0xXX) tokens (stack variables with offsets)
# - imm values (immediate values preserved)
# - address tokens (code addresses with hierarchical positions)
```

## Key Differences

| Feature | Baseline | Address-Aware |
|---------|----------|---------------|
| **Position Embedding** | Tied to word embedding | Hierarchical sin/cos encoding |
| **Address Representation** | JUMP_ADDR_X (simple index) | address(hex:bnorm:fnorm:bbnorm) |
| **Data Addresses** | GLOBAL_VAR | daddr(hex:bnorm:fnorm:bbnorm) |
| **Stack Variables** | LOCAL_VAR | var(0xXX) with offset embedding |
| **Immediate Values** 
| **Hierarchical Info** | None | Binary/Function/BB structure |
| **Vocabulary Size** | ~2,902 | ~3,000-5,000 |
| **Training Data** | Simple text format | Address-annotated format |

## Hypothesis

**Baseline relies on** position=word embedding trick to capture structure.

**Address-aware approach** explicitly models:
1. Hierarchical position (binary → function → basic block)
2. Semantic differences (code addr vs data addr)
3. Stack frame structure (var offsets)
4. Immediate

**Expected outcome**: Address-aware should better capture:
- Cross-function relationships (binary-level positions)
- Function-internal structure (function-level positions)
- Control flow patterns (BB-level positions)
- Memory access patterns (daddr separate from code addresses)

## Data Requirements

### Baseline Data
```
push rbp mov rbp rsp lea JUMP_ADDR_7 call GLOBAL_VAR
```

### Address-Aware Data
```
push(0x401000:0.1:0.05:0.0) rbp mov(0x401001:0.11:0.1:0.01) rbp rsp lea(0x401005:0.15:0.2:0.05) rbp address(0x40100c:0.2:0.3:0.1) call(0x401010:0.25:0.35:0.15) daddr(0x405000:0.3:0.4:0.2)
```

Format: `token(hex_addr:binary_norm:function_norm:bb_norm)`
- hex_addr: Original address (for debugging)
- binary_norm: [0, 1] position in binary
- function_norm: [0, 1] position in function
- bb_norm: [0, 1] position in basic block

## Next Steps

1. **Generate address-aware data**:
   - Modify `convert_pkl_to_text.py` to output address-annotated format
   - Include binary/function/BB normalization
   - Preserve daddr, var, imm vocabularies

2. **Create vocab**:
   ```bash
   cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
   python -c "from vocab import WordVocab; WordVocab.create_vocab('/data/kun/jtransdata/addressaware_train.txt', 'vocab.pkl')"
   ```

3. **Run training**:
   ```bash
   bash run_addressaware_pretrain.sh
   ```

4. **Compare results**:
   - MLM/JTP accuracy
   - Downstream task performance
   - Embedding quality (t-SNE visualization)

## File Structure

```
pretrain/
├── baseline/                    # Simple JUMP_ADDR_X + position=word
│   ├── train_baseline.py
│   ├── dataloader_baseline.py
│   ├── model_baseline.py
│   └── vocab.txt               # ~2,902 tokens
│
└── address_aware/              # Hierarchical address + daddr/var/imm
    ├── train_addressaware.py
    ├── dataloader_addressaware.py
    ├── model_addressaware.py
    ├── address_embedding.py    # Hierarchical positional encoding
    ├── vocab.py                # Vocabulary builder
    └── vocab.pkl              # ~3,000-5,000 tokens (to be generated)
```
