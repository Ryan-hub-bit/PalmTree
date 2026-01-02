# Address-Aware jTrans Pretraining

This directory implements address-aware pretraining using the strupos tokenization approach.

## Key Differences from Baseline

### Baseline (position_embeddings = word_embeddings)
- Simple JUMP_ADDR_X tokens
- No hierarchical address structure
- Position embeddings tied to word embeddings (jTrans trick)

### Address-Aware (this approach)
- **Hierarchical address positions**: binary/function/basic-block normalized positions
- **daddr vocabulary**: Separate tokens for data addresses
- **imm vocabulary**: Immediate values preserved
- **var(0xXX) tokens**: Stack variable offsets tracked
- **NO position=word embedding trick**: Uses proper address-aware positional embeddings

## Tokenization Format

```
# Baseline format:
push rbp mov rsp lea JUMP_ADDR_7 call GLOBAL_VAR

# Address-aware format:
push(0x1000:0.1:0.05:0.0) rbp mov(0x1001:0.11:0.1:0.01) rsp lea(0x1005:0.15:0.2:0.05) address(0x100c:0.2:0.3:0.1) call(0x1010:0.25:0.35:0.15) daddr(0x5000:0.3:0.4:0.2)
```

Each token with address has format: `token(hex_addr:binary_norm:function_norm:bb_norm)`
- hex_addr: Original address (for debugging)
- binary_norm: [0,1] position within entire binary
- function_norm: [0,1] position within function
- bb_norm: [0,1] position within basic block

## Comparison Goal

Test if hierarchical address structure + explicit address vocabularies outperforms
the simple position=word embedding trick used in baseline jTrans.
