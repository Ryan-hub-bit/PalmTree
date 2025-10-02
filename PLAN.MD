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
```


### 3.2  ROPE

q = W_q(x_token)            # [128]
k = W_k(x_token)            # [128]

q_rope = apply_rope(q, pos_idx=seq_idx)
k_rope = apply_rope(k, pos_idx=seq_idx)


## 4. Input to BERT

## 5. Address Out-of-Window Issue