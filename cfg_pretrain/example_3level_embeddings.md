# 3-Level Embeddings Example

## Example Basic Block Pair

```assembly
Source BB:
<addr_start:0x402118:0.269:0.5> mov rax rbx call <addr_code:0x402cc0:0.364:2.0> <addr_end:0x402130:0.271:0.8>

Target BB:
<addr_start:0x402cc0:0.364:0.0> push rbp mov rbp rsp <addr_end:0x402cc8:0.365:0.3>
```

## 3-Level Embedding Breakdown

### Level 1: Semantic Embedding (Token IDs)
Maps instruction tokens to learned vectors:

| Token | Token ID | Semantic Vector (learned) |
|-------|----------|---------------------------|
| [CLS] | 3        | [0.1, -0.3, 0.5, ...] (768-dim) |
| [ADDR] | 100      | [0.2, 0.1, -0.2, ...] |
| mov | 5234     | [-0.1, 0.4, 0.2, ...] |
| rax | 1234     | [0.3, -0.1, 0.1, ...] |
| rbx | 1245     | [0.3, -0.1, 0.1, ...] |
| call | 5456     | [0.5, 0.2, -0.3, ...] |
| [ADDR] | 100      | [0.2, 0.1, -0.2, ...] |
| [SEP] | 2        | [0.0, 0.1, -0.1, ...] |

### Level 2: Address Position Embedding (Global + Local)

Each token gets TWO positional values from the address context:

| Token | Binary Pos | Function Pos | Meaning |
|-------|------------|--------------|---------|
| [CLS] | 0.0        | 0.0          | Special token (no position) |
| [ADDR] (addr_start) | 0.269 | 0.5 | Start at 26.9% through binary, 50% through function |
| mov | 0.269      | 0.5          | Same context as addr_start |
| rax | 0.269      | 0.5          | Same context |
| rbx | 0.269      | 0.5          | Same context |
| call | 0.269      | 0.5          | Same context |
| [ADDR] (addr_code) | 0.364 | 2.0 | Target at 36.4% through binary, external forward call |
| [ADDR] (addr_end) | 0.271 | 0.8 | End at 27.1% through binary, 80% through function |
| [SEP] | 0.0        | 0.0          | Special token |

**Encoding**: Can use sinusoidal or learned:
- Binary position → sin/cos(0.269 * freq)
- Function position → sin/cos(0.5 * freq)

### Level 3: Sequence Position Embedding

Standard transformer positional encoding based on token order:

| Token | Seq Pos | Encoding |
|-------|---------|----------|
| [CLS] | 0       | PE(0) = sin/cos based on position 0 |
| [ADDR] | 1       | PE(1) |
| mov | 2       | PE(2) |
| rax | 3       | PE(3) |
| rbx | 4       | PE(4) |
| call | 5       | PE(5) |
| [ADDR] | 6       | PE(6) |
| [ADDR] | 7       | PE(7) |
| [SEP] | 8       | PE(8) |

## Final Combined Embedding

For each token:

```
Final_Embedding(token_i) = Semantic_Embedding(token_id_i)
                         + Binary_Position_Embedding(bin_pos_i)  
                         + Function_Position_Embedding(func_pos_i)
                         + Sequence_Position_Embedding(seq_pos_i)
```

Example for token "mov" at position 2:
```
E_mov = E_semantic(5234)           # Learned from vocab
      + E_binary(0.269)            # Position in binary
      + E_function(0.5)            # Position in function  
      + E_sequence(2)              # Position in sequence
```

## Why 3 Levels?

1. **Semantic** - What does the instruction mean?
2. **Address Position** - Where is it located globally and locally?
3. **Sequence Position** - What's the order in this particular sequence?

This allows the model to understand:
- Instruction semantics (Level 1)
- Code location context (Level 2)
- Sequential dependencies (Level 3)

All three are crucial for understanding assembly code behavior!
