# Address-Aware Embedding & ROPE Integration Plan

This document outlines the pipeline for extending PalmTree embeddings with **token + src/tgt address slots**, applying **Rotary Position Embeddings (ROPE)**, and integrating into a Transformer/BERT workflow.

---

## 1. Data Collection

- Extract per-instruction features:
  - `inst_id`, `func_id`, `bb_id`
  - `opcode`, `operands`
  - `src_addr`, `tgt_addr`
  - Control-flow type (ret/jumptable/tailcall/etc.)

- Normalize addresses:
  - Keep raw addresses for logs.
  - Derive position proxies (instruction index, CFG order, or data-flow distance).

---

## 2. Reproduce PalmTree

- Tokenize opcode + operands as in PalmTree.
- Train baseline model to verify:
  - Opcode/operand outlier detection
  - Basic block matching
  - Indirect edge prediction

---

## 3. Construct Embeddings (3×128 per instruction)

### 3.1 Token embedding + sequence-length position
```python
token_emb = embed_token(token_id)      # [128]
seq_emb   = embed_seqlen(seq_idx)      # [128]

x_token   = token_emb + seq_emb
x_src = sin/cos(addr)
x_des = sin/cos(des)
x_i = [x_token, x_src, x_des] [3 * 128]

```

### 3.2  Transfomer encoder

x_i -> make 3 * [128]  to [3 * 128] to bert

-> make 3 * [128] to 128 then input to bert 


## 5. Address Out-of-Window Issue
Attention is within a sequence.

0x400800 in one binary != 0x400800 in another binary.  