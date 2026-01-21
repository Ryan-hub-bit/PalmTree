# jTrans_instr Architecture

## Overview

jTrans_instr uses standard BERT architecture with **instruction-level embeddings** to understand assembly code structure at the instruction level.

## Embedding Architecture

The model combines **three types of embeddings**:

```
Final Embedding = Token Embedding + Position Embedding + Instruction Embedding
```

### 1. Token Embedding (word_embeddings)
- Standard vocabulary embeddings like BERT
- Maps each token to its semantic representation
- Size: `[vocab_size, hidden_size]`
- Covers: assembly instructions, registers, immediates, instr_addr tokens, special tokens

### 2. Position Embedding (position_embeddings)
- Standard absolute position embeddings like BERT
- Maps token positions (0-511) to position representations
- Size: `[512, hidden_size]`
- Token at position 0 gets position_embeddings[0], position 1 gets position_embeddings[1], etc.

### 3. Instruction Embedding (instruction_embeddings) **[NEW]**
- Maps instruction indices (0-200) to instruction-level representations
- Size: `[201, hidden_size]`
- Multiple tokens can share the same instruction embedding
- Example: `mov eax, 0x100` might be 3 tokens but all belong to instruction #5

## Special Handling: instr_addr Tokens

**Key Innovation:** `instr_addr_{i}` tokens directly use instruction embedding `i`.

### Why This Matters

When you have a jump instruction like:
```
jmp instr_addr_5
```

The token `instr_addr_5` will use `instruction_embeddings[5]` **instead of** its token embedding.

Meanwhile, if instruction #5 is `mov eax, ebx`, all its tokens have `instruction_ids=5` and also use `instruction_embeddings[5]`.

**Result:** The jump target and the actual instruction share the same instruction-level representation!

### Example Flow

```
Assembly code:
  instr_0: push ebp
  instr_1: mov ebp, esp
  ...
  instr_5: mov eax, ebx       <- Target instruction
  ...
  instr_10: jmp instr_addr_5  <- Jump to instruction 5
```

**Token embeddings:**
```
Instruction 5: [mov, eax, ebx]
- Token "mov" → token_embedding[mov] + position_embedding[25] + instruction_embedding[5]
- Token "eax" → token_embedding[eax] + position_embedding[26] + instruction_embedding[5]
- Token "ebx" → token_embedding[ebx] + position_embedding[27] + instruction_embedding[5]

Instruction 10: [jmp, instr_addr_5]
- Token "jmp" → token_embedding[jmp] + position_embedding[50] + instruction_embedding[10]
- Token "instr_addr_5" → instruction_embedding[5] + position_embedding[51] + instruction_embedding[10]
                          ^^^^^^^^^^^^^^^^^^^^^^
                          Uses instruction embedding 5 directly!
```

**Semantic Link:**
- Tokens in instruction #5 use `instruction_embedding[5]` (via instruction_ids)
- Token `instr_addr_5` uses `instruction_embedding[5]` (direct replacement)
- Model learns: jump targets reference actual instructions

## Implementation Details

### Setting Up instr_addr Mapping

After loading the tokenizer, call:
```python
model.set_instr_addr_token_ids(tokenizer)
```

This maps each `instr_addr_{i}` token ID to instruction embedding `i`.

### Forward Pass

```python
model(
    input_ids=input_ids,              # [batch, seq_len] - Token IDs
    attention_mask=attention_mask,    # [batch, seq_len] - Attention mask
    instruction_ids=instruction_ids   # [batch, seq_len] - Instruction index for each token
)
```

**instruction_ids format:**
```
[0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, ...]
 |instr0| |i1| |instr2   | |i3|
```
Each token maps to its parent instruction index (0-200).

### Embedding Computation

```python
# Step 1: Get token embeddings
token_emb = word_embeddings(input_ids)

# Step 2: Replace instr_addr tokens with instruction embeddings
for i in range(max_instructions):
    mask = (input_ids == token_id_of_instr_addr_i)
    token_emb[mask] = instruction_embeddings[i]

# Step 3: Add position embeddings
position_emb = position_embeddings(position_ids)
embeddings = token_emb + position_emb

# Step 4: Add instruction embeddings
instruction_emb = instruction_embeddings(instruction_ids)
embeddings = embeddings + instruction_emb

# Step 5: Layer norm + dropout
embeddings = LayerNorm(embeddings)
embeddings = dropout(embeddings)
```

## Comparison with Baseline jTrans

| Feature | Baseline jTrans | jTrans_instr |
|---------|----------------|--------------|
| Position embeddings | Uses word_embeddings (position=word trick) | Standard BERT absolute positions |
| Instruction-level understanding | Implicit through position trick | Explicit instruction_embeddings |
| Jump addresses | JUMP_ADDR_X (token positions) | instr_addr_{i} (instruction indices) |
| Jump-instruction link | No direct link | instr_addr_{i} uses instruction_embedding[i] |
| Token type embeddings | Has token_type_embeddings | Removed (not needed) |

## Training Tasks

### 1. Masked Language Modeling (MLM)
- Randomly mask tokens
- Predict original token from vocabulary
- Output: `[batch, seq_len, vocab_size]`

### 2. Jump Target Prediction (JTP)
- Mask jump target addresses (instr_addr tokens)
- Predict which instruction is the jump target
- Output: `[batch, seq_len, max_instructions]`
- Range: 0-200 (instruction indices)

## Benefits

1. **Explicit instruction structure:** Model learns instruction boundaries through instruction_embeddings
2. **Semantic jump targets:** Jump addresses directly reference instruction representations
3. **Standard BERT compatibility:** Uses standard position embeddings, easier to initialize from pretrained BERT
4. **Cleaner architecture:** No position=word trick, no unnecessary token_type embeddings
5. **Instruction-level reasoning:** Model can learn patterns at instruction level, not just token level
