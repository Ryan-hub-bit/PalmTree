# Input Consistency: PalmTree vs AddressAware

## Overview

Both models receive **IDENTICAL token sequences** to ensure fair comparison. The only difference is that AddressAware receives additional address positional information.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Inline Address Format (from data generator)                     │
│ mov(0x400000:0.12345678:0.123456:0.1234) rax                   │
│ addr_code(0x401000:0.13456789:0.134567:0.1345)                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Paired Dataloader (dataloader_paired.py)                        │
│                                                                  │
│ Parses and extracts:                                            │
│  1. Tokens: [mov, rax, addr_code]                              │
│  2. Address positions:                                          │
│     - Binary:   [0.12345678, 0.0, 0.13456789]                  │
│     - Function: [0.123456, 0.0, 0.134567]                      │
│     - BB:       [0.1234, 0.0, 0.1345]                          │
│  3. Segment labels: [0, 0, 1]                                  │
│  4. MLM labels (CFG only): [mov, -1, addr_code]                │
│  5. NSP labels: 0 or 1                                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌──────────────────────┐              ┌──────────────────────┐
│ PalmTree Model       │              │ AddressAware Model   │
├──────────────────────┤              ├──────────────────────┤
│ Input:               │              │ Input:               │
│  ✓ Token IDs         │              │  ✓ Token IDs         │
│  ✓ Segment labels    │              │  ✓ Segment labels    │
│  ✗ Address positions │              │  ✓ Binary positions  │
│    (ignored)         │              │  ✓ Function positions│
│                      │              │  ✓ BB positions      │
│                      │              │                      │
│ Embedding:           │              │ Embedding:           │
│  token_emb +         │              │  token_emb +         │
│  position_emb +      │              │  position_emb +      │
│  segment_emb         │              │  segment_emb +       │
│                      │              │  address_emb         │
│                      │              │                      │
│ (128-dim)            │              │ (128-dim after MLP)  │
└──────────────────────┘              └──────────────────────┘
```

## Key Points

### ✓ Identical Inputs
- **Token sequences**: Exactly the same for both models
- **Vocabulary**: All tokens from PalmTree's vocabulary (6,631 tokens)
- **Segment labels**: Same for both models
- **MLM/NSP pairs**: Same masking and pairing strategy

### ✓ Address Position Usage
| Component | PalmTree | AddressAware |
|-----------|----------|--------------|
| Token embedding | ✓ Used | ✓ Used (frozen from PalmTree) |
| Sequence position | ✓ Used | ✓ Used (frozen from PalmTree) |
| Segment embedding | ✓ Used | ✓ Used (frozen from PalmTree) |
| Binary position | ✗ Not used | ✓ Used (NEW, trainable) |
| Function position | ✗ Not used | ✓ Used (NEW, trainable) |
| Basic block position | ✗ Not used | ✓ Used (NEW, trainable) |

### ✓ Fair Comparison
1. **Same data source**: Both read from `all_cfg_combined.txt` and `all_dfg_combined.txt`
2. **Same preprocessing**: Inline format parsed identically
3. **Same masking**: 15% MLM masking for CFG, none for DFG
4. **Same NSP strategy**: 50% positive, 50% negative pairs
5. **Same vocabulary**: No new tokens introduced

### ✓ What Makes AddressAware Different?
Only the **additional positional information**:
- Hierarchical address embeddings (binary/function/BB level)
- MLP projection to match embedding dimension
- Concatenated with frozen PalmTree embeddings

## Example: Token Processing

### Input Sequence
```
mov(0x400000:0.12:0.13:0.14) rax addr_code(0x401000:0.15:0.16:0.17)
```

### After Parsing
```
Tokens:         [mov,     rax,  addr_code]
Token IDs:      [1234,    567,  890]
Binary pos:     [0.12,    0.0,  0.15]
Function pos:   [0.13,    0.0,  0.16]
BB pos:         [0.14,    0.0,  0.17]
Segment:        [0,       0,    1]
```

### PalmTree Processing
```python
# Forward pass
x = token_emb[1234, 567, 890]      # From vocab
x += position_emb[0, 1, 2]         # Sequence positions
x += segment_emb[0, 0, 1]          # Segment labels
# Address positions [0.12, 0.0, 0.15] etc. are IGNORED
```

### AddressAware Processing
```python
# Forward pass
x = token_emb[1234, 567, 890]      # From vocab (FROZEN from PalmTree)
x += position_emb[0, 1, 2]         # Sequence positions (FROZEN from PalmTree)
x += segment_emb[0, 0, 1]          # Segment labels (FROZEN from PalmTree)
x += address_emb([0.12, 0.13, 0.14],  # NEW: Binary/Function/BB positions
                 [0.0, 0.0, 0.0],       # (TRAINABLE)
                 [0.15, 0.16, 0.17])
```

## Verification

Run the verification script to check input consistency:
```bash
cd /home/kun/Document/PalmTree/addressaware/test_comparison
python verify_inputs.py
```

This will:
- Show actual token sequences from the dataloader
- Verify all tokens are in PalmTree's vocabulary
- Display address position distributions
- Confirm both models see identical tokens
