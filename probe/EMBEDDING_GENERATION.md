# BB Bucket Probe: Embedding Generation

## Overview

The probe encodes instructions using the same pattern as `UsableTransformer` but extracts and uses address position information for the address-aware model.

## Address-Aware Model Architecture

The address-aware model integrates address information **directly into the token embeddings**:

```
Token Embedding = Token + Sequence_Position + Address_Position + Segment
                    ↓           ↓                    ↓              ↓
                  frozen      frozen            TRAINABLE        trainable
                (PalmTree)  (PalmTree)         (3-level sin/cos)  (NSP)
```

### Address Position Embedding

The address position embedding uses **sin/cos encoding on three levels**:

1. **Binary position (bnorm)**: Position within entire binary [0, 1]
2. **Function position (fnorm)**: Position within function [0, 1]  
3. **Basic block position (bbnorm)**: Position within basic block [0, 1]

Each level gets its own sinusoidal encoding, then they are:
- Weighted by learnable parameters
- Concatenated: `[binary_enc | function_enc | bb_enc]`
- Projected through MLP to embedding dimension

## Data Flow

### Input Instruction
```
"mov(0x202146:0.41853381:0.162162:0.1935) r9 rdx"
     ↓
Parse address info:
  - bnorm = 0.41853381  (position in binary)
  - fnorm = 0.162162    (position in function)
  - bbnorm = 0.1935     (position in basic block)
     ↓
Clean for tokenization: "mov r9 rdx"
```

### Tokenization (like UsableTransformer)
```
"mov r9 rdx"
    ↓
[vocab.to_seq()]
    ↓
[15, 42, 73]  # Token IDs
    ↓
[Add Special Tokens]
    ↓
[3, 15, 42, 73, 2]  # [CLS] mov r9 rdx [SEP]
    ↓
Segment labels: [1, 1, 1, 1, 1]  # All 1s
```

### Position Arrays
```
For each token, we track 3 position values:

Example: "mov(0x202146:0.41853381:0.162162:0.1935) r9 address(0x203c50:0.75864780:0.000000:0)"

Token:       [CLS]  mov         r9          address     [SEP]
Token ID:    3      15          42          73          2
Segment:     1      1           1           1           1
Bnorm:       0.0    0.41853381  0.0         0.75864780  0.0
Fnorm:       0.0    0.162162    0.0         0.000000    0.0
BBnorm:      0.0    0.1935      0.0         0.0000      0.0
             ↑      ↑           ↑           ↑           ↑
         (special) (opcode)  (register)  (address)   (special)
                   inst pos   no pos     target pos   no pos
```

**Position assignment:**
- **Opcode token**: Gets the instruction's address position (bnorm, fnorm, bbnorm)
- **Address operand**: Gets the target address position from `address(0xADDR:bnorm:fnorm:bbnorm)`
- **Register/immediate operands**: Get default (0.0, 0.0, 0.0) - no specific position
- **Special tokens** ([CLS], [SEP]): Get default (0.0, 0.0, 0.0)

### Forward Pass (Address-Aware Model)

```python
# Embedding layer combines 4 components:
token_emb = token_embedding(token_ids)          # [batch, seq, hidden]
seq_pos_emb = sequence_position(position_ids)   # [batch, seq, hidden]
addr_pos_emb = address_position(bnorm, fnorm, bbnorm)  # [batch, seq, hidden]
segment_emb = segment_embedding(segment_labels) # [batch, seq, hidden]

# Combined embedding (element-wise addition)
embedding = token_emb + seq_pos_emb + addr_pos_emb + segment_emb

# Pass through transformer
encoded = transformer(embedding)  # [batch, seq, hidden]

# Mean pooling (like UsableTransformer)
result = torch.mean(encoded, dim=1)  # [batch, hidden]
```

### Forward Pass (Baseline Model)

```python
# Standard BERT - no address positions
token_emb = token_embedding(token_ids)
pos_emb = position_embedding(position_ids) 
segment_emb = segment_embedding(segment_labels)

embedding = token_emb + pos_emb + segment_emb
encoded = transformer(embedding)
result = torch.mean(encoded, dim=1)
```

## Testing

Run the probe:
```bash
cd /home/kun/Document/PalmTree/probe
./test_probe.sh
```

This will:
1. Load 473 instructions from the test binary
2. Encode them using both models
3. Train logistic regression probes
4. Compare bucket prediction accuracy
5. Save results to `probe_results/`

Expected output:
- Address-aware model should have **higher accuracy** if it learned better positional encodings
- Results show per-bucket performance (which positions are easier/harder to identify)
