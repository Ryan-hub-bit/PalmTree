# Real Example: 3-Level Embeddings from Your Data

## Original Line from bb_pairs_output/atilibusb.so_bb_pairs.txt

```
<addr_start:0x4020f0:0.267785:0.000000> lea rdi [ rel <addr_data:0x405150:0.661077:-3.000000> ] lea rax [ rel <addr_data:0x405150:0.661077:-3.000000> ] cmp rax rdi je <addr_code:0x402118:0.269055:1.000000> <addr_end:0x402103:0.268388:0.558824> <addr_start:0x402118:0.269055:1.000000> retn <addr_end:0x402119:0.269087:1.000000>
```

## Split into Source and Target Basic Blocks

**Source BB:**
```
<addr_start:0x4020f0:0.267785:0.000000> lea rdi [ rel <addr_data:0x405150:0.661077:-3.000000> ] lea rax [ rel <addr_data:0x405150:0.661077:-3.000000> ] cmp rax rdi je <addr_code:0x402118:0.269055:1.000000> <addr_end:0x402103:0.268388:0.558824>
```

**Target BB:**
```
<addr_start:0x402118:0.269055:1.000000> retn <addr_end:0x402119:0.269087:1.000000>
```

## 3-Level Embeddings Breakdown

After processing through the data loader, here's what each token gets:

**IMPORTANT**: Only `[ADDR]` tokens (representing `addr_*`) have Level 2 position embeddings!
Regular instruction tokens get (0.0, 0.0) for address positions.

### Tokenized Sequence: [CLS] source_tokens [SEP] target_tokens [SEP]

| Position | Token | Token ID | Binary Pos | Function Pos | Seq Pos | Notes |
|----------|-------|----------|------------|--------------|---------|-------|
| 0 | [CLS] | 3 | 0.0000 | 0.0000 | 0 | Special token |
| 1 | [ADDR] | 100 | **0.267785** | **0.000000** | 1 | ✓ addr_start - has position |
| 2 | lea | 5234 | 0.0000 | 0.0000 | 2 | Regular token - NO position |
| 3 | rdi | 1234 | 0.0000 | 0.0000 | 3 | Regular token - NO position |
| 4 | [ | 89 | 0.0000 | 0.0000 | 4 | Regular token - NO position |
| 5 | rel | 90 | 0.0000 | 0.0000 | 5 | Regular token - NO position |
| 6 | [ADDR] | 100 | **0.661077** | **-3.000000** | 6 | ✓ addr_data - has position |
| 7 | ] | 91 | 0.0000 | 0.0000 | 7 | Regular token - NO position |
| 8 | lea | 5234 | 0.0000 | 0.0000 | 8 | Regular token - NO position |
| 9 | rax | 1245 | 0.0000 | 0.0000 | 9 | Regular token - NO position |
| 10 | [ | 89 | 0.0000 | 0.0000 | 10 | Regular token - NO position |
| 11 | rel | 90 | 0.0000 | 0.0000 | 11 | Regular token - NO position |
| 12 | [ADDR] | 100 | **0.661077** | **-3.000000** | 12 | ✓ addr_data - has position |
| 13 | ] | 91 | 0.0000 | 0.0000 | 13 | Regular token - NO position |
| 14 | cmp | 5456 | 0.0000 | 0.0000 | 14 | Regular token - NO position |
| 15 | rax | 1245 | 0.0000 | 0.0000 | 15 | Regular token - NO position |
| 16 | rdi | 1234 | 0.0000 | 0.0000 | 16 | Regular token - NO position |
| 17 | je | 5678 | 0.0000 | 0.0000 | 17 | Regular token - NO position |
| 18 | [ADDR] | 100 | **0.269055** | **1.000000** | 18 | ✓ addr_code - has position |
| 19 | [ADDR] | 100 | **0.268388** | **0.558824** | 19 | ✓ addr_end - has position |
| 20 | [SEP] | 2 | 0.0000 | 0.0000 | 20 | Separator |
| 21 | [ADDR] | 100 | **0.269055** | **1.000000** | 21 | ✓ Target addr_start - has position |
| 22 | retn | 6789 | 0.0000 | 0.0000 | 22 | Regular token - NO position |
| 23 | [ADDR] | 100 | **0.269087** | **1.000000** | 23 | ✓ Target addr_end - has position |
| 24 | [SEP] | 2 | 0.0000 | 0.0000 | 24 | End separator |

## What Your Model Receives

```python
batch = {
    # Level 1: Semantic Embedding
    'input_ids': [3, 100, 5234, 1234, 89, 90, 100, 91, 5234, 1245, ...],
    
    # Level 2: Address Position Embedding (ONLY for [ADDR] tokens!)
    'binary_positions': [0.0,        # [CLS] - no position
                         0.267785,   # [ADDR] addr_start ✓
                         0.0,        # lea - no position
                         0.0,        # rdi - no position
                         0.0,        # [ - no position
                         0.0,        # rel - no position
                         0.661077,   # [ADDR] addr_data ✓
                         0.0,        # ] - no position
                         ...],
    
    'function_positions': [0.0,      # [CLS]
                          0.0,       # [ADDR] addr_start ✓
                          0.0,       # lea - no position
                          0.0,       # rdi - no position
                          0.0,       # [ - no position
                          0.0,       # rel - no position
                          -3.0,      # [ADDR] addr_data ✓
                          0.0,       # ] - no position
                          ...],
    
    # Level 3: Sequence Position Embedding (ALL tokens)
    'sequence_positions': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
}
```

## How to Use in Your Model

```python
# Your model forward pass
def forward(self, batch):
    # Level 1: Semantic embedding
    semantic_emb = self.token_embedding(batch['input_ids'])  # [batch, seq, 768]
    
    # Level 2: Address position embeddings
    binary_emb = self.binary_pos_embedding(batch['binary_positions'])  # [batch, seq, 768]
    function_emb = self.function_pos_embedding(batch['function_positions'])  # [batch, seq, 768]
    
    # Level 3: Sequence position embedding (standard transformer)
    sequence_emb = self.sequence_pos_embedding(batch['sequence_positions'])  # [batch, seq, 768]
    
    # Combine all three levels
    final_emb = semantic_emb + binary_emb + function_emb + sequence_emb
    
    # Feed to transformer
    output = self.transformer(final_emb, batch['attention_mask'])
    return output
```

## Key Observations

1. **Token "lea" at position 2:**
   - Semantic: ID 5234 (instruction meaning)
   - Binary position: 0.267785 (26.8% through binary)
   - Function position: 0.000000 (start of function)
   - Sequence position: 2 (second token in sequence)

2. **Token "lea" at position 8:**
   - Same semantic ID 5234
   - Different address context: 0.661077, -3.0 (data reference)
   - Different sequence position: 8

3. **Special marker -3.0:**
   - Indicates data reference, not code
   - Your model learns this is semantically different from code positions

This 3-level design lets your model understand:
- **What** the instruction is (semantic)
- **Where** it's located (address position)  
- **When** it appears (sequence order)
