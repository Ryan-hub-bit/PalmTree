# 3-Level Embedding Architecture for Address-Aware PalmTree

## Overview
The model now uses **three levels of embeddings** that are fused together to create rich representations of basic blocks with address information.

## The 3 Levels

### Level 1: Semantic Embeddings
- **What**: Token embeddings from the pre-trained PalmTree model
- **Purpose**: Capture the semantic meaning of instructions, registers, and operands
- **Example**: `mov`, `rax`, `[rbp-0x10]` each have learned semantic representations
- **Dimension**: `[batch_size, seq_len, hidden_size]`

### Level 2: Address-Level Embeddings
This level has TWO components:

#### 2a. Address Type Embeddings
- **What**: Learned embeddings for different address types
- **Purpose**: Distinguish between code addresses, data addresses, BB start/end markers, etc.
- **Types**:
  - `code`: Code addresses (function calls, jumps)
  - `data`: Data addresses (memory operands)
  - `start`: Basic block start marker
  - `end`: Basic block end marker
  - `tgt`: Target address
  - `unknown`: Unknown/regular tokens
- **Implementation**: `nn.Embedding(6, hidden_size)`
- **Dimension**: `[batch_size, seq_len, hidden_size]`

#### 2b. Address Value Encodings
- **What**: Sinusoidal encoding of actual address values
- **Purpose**: Encode the numerical address value (like positional encoding but for addresses)
- **Formula**: Similar to Transformer positional encoding
  ```python
  sin(address / 10000^(i/d)) for even i
  cos(address / 10000^(i/d)) for odd i
  ```
- **Implementation**: Projected to hidden_size via `nn.Linear(128, hidden_size)`
- **Dimension**: `[batch_size, seq_len, hidden_size]`

**Combined Address Level**: `addr_type_embeds + addr_value_embeds`

### Level 3: Positional Embeddings
- **What**: Position of token in the sequence
- **Purpose**: Capture where in the basic block sequence this token appears
- **Implementation**: `nn.Embedding(max_seq_length, hidden_size)`
- **Dimension**: `[batch_size, seq_len, hidden_size]`
- **Note**: This is ADDITIONAL to PalmTree's internal positional encoding

## Fusion Strategy

The three levels are combined using a learned fusion network:

```
Input: Concat([semantic, address, positional]) -> [B, L, 3*H]
   ↓
Linear(3*H -> 2*H) + LayerNorm + GELU + Dropout
   ↓
Linear(2*H -> H) + LayerNorm
   ↓
Output: Fused features [B, L, H]
```

This allows the model to learn how to optimally combine the three information sources.

## Data Flow Example

For instruction: `mov rax qword [ rel addr_data:0x4040c0 ]`

At position 5 in the sequence, for token `addr_data:0x4040c0`:

1. **Level 1 (Semantic)**: 
   - Token ID: vocab['<addr_data>'] 
   - Embedding: PalmTree token embedding vector

2. **Level 2a (Address Type)**:
   - Type ID: ADDR_TYPE_LABELS['data'] = 1
   - Embedding: addr_type_embedding(1)

3. **Level 2b (Address Value)**:
   - Value: 0x4040c0 = 4210880
   - Sinusoidal encoding: [sin/cos pattern of length 128]
   - Projected: Linear(encoding) -> hidden_size vector

4. **Level 3 (Position)**:
   - Position: 5
   - Embedding: position_embedding(5)

5. **Fusion**:
   - Concatenate all 3 levels: [semantic | address | positional]
   - Pass through fusion network
   - Result: Rich contextual representation

## Benefits

1. **Semantic Understanding**: Learns what instructions mean
2. **Address Awareness**: Knows where code/data is located in memory
3. **Sequence Awareness**: Understands position in the control flow
4. **Flexible Fusion**: Model learns optimal combination of all three

## Training

The model is trained on multiple tasks:
- **Next BB Prediction**: Predict the next basic block
- **Address Type Classification**: Classify address types
- **Edge Type Classification**: Classify control flow edge types

All tasks benefit from the rich 3-level embeddings.
