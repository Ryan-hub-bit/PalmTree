================================================================================
jTrans Address-Aware Complete Pipeline
================================================================================

From Data Generation → Pretraining → Finetuning → Evaluation

This document explains every step of the address-aware jTrans pipeline.

================================================================================
OVERVIEW
================================================================================

Pipeline stages:
1. Data Generation: Create address-aware tokenized corpus
2. Vocabulary Creation: Build vocab with special tokens
3. Pretraining: MLM + JTP tasks to learn representations
4. Finetuning: Triplet learning for function similarity
5. Evaluation: Pool-based retrieval metrics

Key difference from baseline: Address embeddings at every stage


================================================================================
STAGE 1: DATA GENERATION
================================================================================

Goal: Generate assembly code with address annotations
------------------------------------------------------

Input format:
    Raw assembly: push rbp / mov rsp, rbp

Output format:
    Address-aware: push(0x1000:0.1:0.5:0.2) rbp mov(0x1001:0.15:0.55:0.25) rsp rbp
    
    Where each instruction has: opcode(0xADDR:binary_norm:function_norm:bb_norm)
    - 0xADDR: Actual address (for reference)
    - binary_norm: Normalized position in binary [0, 1]
    - function_norm: Normalized position in function [0, 1]
    - bb_norm: Normalized position in basic block [0, 1]




================================================================================
STAGE 2: VOCABULARY CREATION
================================================================================

Goal: Build vocabulary from address-aware corpus
-------------------------------------------------

Location: /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/

Key script: create_vocab.py (or similar)

Process:
1. Parse address-aware corpus
2. Extract unique tokens:
   - Opcodes: push, mov, add, call, jmp, etc.
   - Registers: rax, rbx, rsp, rbp, etc.
   - Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
   - Address tokens: address, daddr (if used as standalone)
   - Immediate values: imm(0xNN)
   - Variables: var(0xNN)

Output files:
    pretrain/address_aware/vocab.txt          - Token list (one per line)
    pretrain/address_aware/vocab_addr.txt     - Extended vocab
    pretrain/address_aware/vocab.pkl          - Python pickle format

Vocabulary size: ~2,000-3,000 tokens

Note: 'address' and 'daddr' may or may not be in vocab depending on
      whether they appear as standalone tokens vs only in patterns


================================================================================
STAGE 3: PRETRAINING
================================================================================

Goal: Learn general binary code representations
------------------------------------------------

Location: /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/

Key files:
- model_addressaware.py: Define AddressAwareJTransForMLM model
- address_embedding.py: Define AddressAwareBERTEmbedding
- dataloader_addressaware.py: Load and process training data
- train_addressaware.py: Training loop
- train_addressaware.sh: Launch script


3.1 Model Architecture
-----------------------
File: pretrain/address_aware/model_addressaware.py

Components:
1. AddressAwareBERTEmbedding:
   - Token embedding (learned)
   - Sequence position embedding (sinusoidal)
   - Segment embedding (learned)
   - Address position embedding (hierarchical sinusoidal + MLP)
   - Var offset embedding (hybrid lookup + MLP)

2. BERT Encoder:
   - 12 transformer layers
   - 12 attention heads
   - 768 hidden dimensions
   - 3072 feed-forward dimensions

3. Task heads:
   - MLM head: Predict masked tokens (vocab_size output)
   - JTP head: Predict jump target positions (max_len output)


3.2 Address Embedding Details
-------------------------------
File: pretrain/address_aware/address_embedding.py

Class: AddressPositionalEmbedding

Process for each token:
1. Check if token has valid address (binary_pos >= 0, function_pos >= 0, bb_pos >= 0)
2. If valid:
   a) Apply sinusoidal encoding to each hierarchical level:
      - binary_sincos = sin_cos(binary_pos)  [16 dims]
      - function_sincos = sin_cos(function_pos)  [16 dims]
      - bb_sincos = sin_cos(bb_pos)  [16 dims]
   b) Concatenate: [binary_sincos, function_sincos, bb_sincos]  [48 dims]
   c) Project through MLP to 768 dims:
      - If token is 'address' ID: use code_address_projection
      - If token is 'daddr' ID: use data_address_projection
      - Otherwise: use code_address_projection (default)
3. If invalid: return zero vector

Final embedding = token_emb + position_emb + segment_emb + address_emb + var_emb


3.3 Training Tasks
-------------------

Task 1: MLM (Masked Language Modeling)
----------------------------------------
- Randomly mask 15% of tokens
- Model predicts original tokens
- Loss: CrossEntropyLoss on masked positions
- Purpose: Learn token semantics and context

Task 2: JTP (Jump Target Prediction)
--------------------------------------
- For jump instructions (jmp, je, jne, call, etc.)
- Replace jump target with JUMP_ADDR_N token
- Model predicts target position in sequence
- Loss: CrossEntropyLoss on jump positions
- Purpose: Learn control flow structure


3.4 Dataloader
---------------
File: pretrain/address_aware/dataloader_addressaware.py

Class: AddressAwarePretrainingDataset

Process for each sample:
1. Load function text from JSON
2. Parse address-aware tokens:
   - push(0x1000:0.1:0.5:0.2) → token='push', pos=(0.1, 0.5, 0.2)
   - address(0x2000:0.2:0.6:0.3) → token='address', pos=(0.2, 0.6, 0.3)
   - daddr(0x3000:0.3:0.7:0.4) → token='daddr', pos=(0.3, 0.7, 0.4)
   - rbp → token='rbp', pos=(-1, -1, -1)

3. Tokenize to IDs using vocab

4. Create MLM mask:
   - Select 15% of tokens
   - 80% replace with [MASK]
   - 10% replace with random token
   - 10% keep original

5. Create JTP labels:
   - Identify jump instructions
   - Store target position

6. Pad to max_len (512)

7. Return batch:
   ```python
   {
     'input_ids': [batch_size, seq_len],
     'attention_mask': [batch_size, seq_len],
     'token_type_ids': [batch_size, seq_len],
     'mlm_labels': [batch_size, seq_len],  # -100 for non-masked
     'jtp_labels': [batch_size, seq_len],  # -100 for non-jumps
     'binary_pos': [batch_size, seq_len],
     'function_pos': [batch_size, seq_len],
     'bb_pos': [batch_size, seq_len],
     'var_offsets': [batch_size, seq_len]
   }
   ```


3.5 Training Script
--------------------
File: pretrain/address_aware/train_addressaware.sh

Command:
```bash
python train_addressaware.py \
    --corpus_path ../PalmTree/corpus/Ox/train/ \
    --vocab_path vocab_addr.txt \
    --output_dir ./addressaware_pretrained/ \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --max_len 512 \
    --num_workers 4
```

Training loop (train_addressaware.py):
1. Load dataset
2. For each epoch:
   a) For each batch:
      - Forward pass through model
      - Compute MLM loss
      - Compute JTP loss
      - Total loss = MLM_loss + JTP_loss
      - Backward pass
      - Update weights
   b) Validation
   c) Save checkpoint

Output:
    pretrain/address_aware/addressaware_pretrained/
    ├── checkpoint_epoch_1.pt
    ├── checkpoint_epoch_2.pt
    ├── ...
    ├── best_model.pt
    └── config.json


================================================================================
STAGE 4: FINETUNING
================================================================================

Goal: Adapt pretrained model for function similarity
-----------------------------------------------------

Location: /home/kun/Document/AAE/extern/jTrans/

Key files:
- finetune.py: Finetuning script with triplet learning
- data_json.py: FunctionDataset_CL_AddressAware_JSON class
- run_finetune_addressaware.sh: Launch script


4.1 Model Architecture for Finetuning
---------------------------------------
File: finetune.py

Class: AddressAwareBertWrapper

Components:
1. Load pretrained BERT:
   - From pretrain/address_aware/addressaware_pretrained/
   - Keep all weights including address embeddings

2. Add projection head:
   ```python
   projection = nn.Sequential(
       nn.Linear(768, 768),
       nn.GELU(),
       nn.Linear(768, 256)  # Embedding dimension for similarity
   )
   ```

3. L2 normalization on output


4.2 Triplet Learning
---------------------
File: data_json.py

Class: FunctionDataset_CL_AddressAware_JSON

Triplet structure:
- Anchor: Function at O0/O1/O2
- Positive: Same function at O3
- Negative: Different function at O3

Sampling strategy:
```python
# For each anchor function
anchor = random_function_at(query_opt)  # O0, O1, or O2
positive = same_function_at(O3)
negative = different_function_at(O3)
```

With --target_opt O3:
- Ensures positive and negative both from O3
- Matches evaluation setup (Ox → O3 retrieval)


4.3 Loss Function
------------------
File: finetune.py

Triplet loss with cosine similarity:
```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    sim_pos = cosine_similarity(anchor, positive)
    sim_neg = cosine_similarity(anchor, negative)
    
    # We want: sim_pos > sim_neg + margin
    # Loss = max(0, margin - (sim_pos - sim_neg))
    loss = max(0, margin - sim_pos + sim_neg)
    return loss
```

Hyperparameters:
- Margin: 0.2
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 15


4.4 Data Processing
--------------------
File: data_json.py

Method: _parse_address_aware_function()

Same as pretraining:
1. Parse address-aware tokens
2. Extract positions
3. Tokenize to IDs
4. Truncate/pad to max_len

Key difference: No masking (we want full function representations)

Returns:
```python
{
  'anchor_input': [seq_len],
  'anchor_mask': [seq_len],
  'anchor_segments': [seq_len],
  'anchor_binary_pos': [seq_len],
  'anchor_function_pos': [seq_len],
  'anchor_bb_pos': [seq_len],
  'anchor_var_offsets': [seq_len],
  
  'positive_input': [...],  # Same structure
  'negative_input': [...],  # Same structure
}
```


4.5 Finetuning Script
-----------------------
File: run_finetune_addressaware.sh

Command:
```bash
python finetune.py \
    --model_type addressaware \
    --model_path pretrain/address_aware/addressaware_pretrained/ \
    --tokenizer jtrans_tokenizer/ \
    --data_dir ../PalmTree/corpus/Ox/ \
    --output_dir output/addressaware_finetune/ \
    --target_opt O3 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 15 \
    --triplet_margin 0.2 \
    --embedding_dim 256 \
    --use_projection
```

Training process:
1. Load pretrained model
2. Add projection layer
3. For each epoch:
   a) For each batch of triplets (anchor, positive, negative):
      - Get embeddings: anchor_emb, pos_emb, neg_emb
      - Compute triplet loss
      - Backward pass
      - Update weights (projection + fine-tune BERT)
   b) Save checkpoint

Output:
    output/addressaware_finetune/
    ├── finetune_epoch_1.pt
    ├── finetune_epoch_5.pt
    ├── finetune_epoch_10.pt
    ├── finetune_epoch_15.pt
    └── config.json


================================================================================
STAGE 5: EVALUATION
================================================================================

Goal: Measure function similarity performance
----------------------------------------------

Location: /home/kun/Document/AAE/extern/jTrans/

Key files:
- evaluate_addressaware_pools.py: Evaluation script
- create_filtered_addressaware_pools.py: Pool generation
- Pool files: output/funcsim/*_pool_*.json


5.1 Pool Structure
-------------------

Format: Ox vs O3 retrieval
- Query: Function at O0, O1, or O2
- Pool: 100, 1000, or 10000 functions at O3
- Ground truth: Correct O3 version in pool

9 evaluation pools:
1. O0_vs_O3_pool_100.json
2. O0_vs_O3_pool_1000.json
3. O0_vs_O3_pool_10000.json
4. O1_vs_O3_pool_100.json
5. O1_vs_O3_pool_1000.json
6. O1_vs_O3_pool_10000.json
7. O2_vs_O3_pool_100.json
8. O2_vs_O3_pool_1000.json
9. O2_vs_O3_pool_10000.json

Each pool file:
```json
{
  "pool": [
    {
      "low_id": "binary1_func1_O3",
      "high_id": "binary1_func1_O3",
      "binary": "binary1",
      "function": "func1",
      "text": "push(0x...) ..."
    },
    ...
  ],
  "queries": [
    {
      "query_id": "binary1_func1_O0",
      "gt_id": "binary1_func1_O3",
      "binary": "binary1",
      "function": "func1",
      "text": "push(0x...) ..."
    },
    ...
  ]
}
```


5.2 Evaluation Process
------------------------
File: evaluate_addressaware_pools.py

For each pool:
1. Load finetuned model with projection
2. Extract embeddings:
   a) For each query function:
      - Parse address-aware tokens
      - Get token IDs and positions
      - Forward through model: query_emb = model(input_ids, binary_pos, ...)
   
   b) For each pool function:
      - Same process: pool_emb = model(input_ids, binary_pos, ...)

3. Compute similarities:
   ```python
   for query in queries:
       sims = cosine_similarity(query_emb, pool_embeddings)
       ranked = argsort(sims, descending=True)
   ```

4. Calculate metrics:
   - MRR (Mean Reciprocal Rank):
     ```python
     rank = position_of_ground_truth + 1
     reciprocal_rank = 1.0 / rank
     MRR = mean(reciprocal_ranks)
     ```
   
   - Recall@K:
     ```python
     Recall@1 = % of queries with GT in top 1
     Recall@5 = % of queries with GT in top 5
     Recall@10 = % of queries with GT in top 10
     ```

5. Save results:
   ```json
   {
     "pool_file": "O0_vs_O3_pool_1000.json",
     "query_count": 100,
     "pool_size": 1000,
     "results": {
       "MRR": 0.45,
       "Recall@1": 0.30,
       "Recall@5": 0.60,
       "Recall@10": 0.75
     }
   }
   ```


5.3 Evaluation Script
-----------------------
File: run_addressaware_pool_evaluation.sh

Command:
```bash
python evaluate_addressaware_pools.py \
    --model_path output/addressaware_finetune/finetune_epoch_15.pt \
    --tokenizer jtrans_tokenizer/ \
    --pool_dir output/funcsim/ \
    --output_file output/jtrans/addressaware_eval_results.json \
    --embedding_dim 256 \
    --use_projection
```


================================================================================
CRITICAL: ADDRESS EMBEDDING APPLICATION
================================================================================

Current Issue in address_embedding.py
---------------------------------------
File: pretrain/address_aware/address_embedding.py
Class: AddressPositionalEmbedding
Method: forward()

Current logic (BUGGY):
```python
# Lines 195-210
address_token_id = vocab_stoi.get('address', -1)  # Returns -1 if not in vocab
daddr_token_id = vocab_stoi.get('daddr', -1)      # Returns -1 if not in vocab

is_code_address = (token_ids == address_token_id).float()  # All False if -1!
is_data_address = (token_ids == daddr_token_id).float()    # All False if -1!

code_embedding = self.code_address_projection(hierarchical_features)
data_embedding = self.data_address_projection(hierarchical_features)

embedding = code_embedding * is_code_address + data_embedding * is_data_address
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           ALL ZEROS if 'address' not in vocab  ALL ZEROS if 'daddr' not in vocab

# Result: No address embeddings applied!
```

Problem:
- If 'address' and 'daddr' are NOT in vocab (as separate tokens)
- Then address_token_id = -1 and daddr_token_id = -1
- No token ID equals -1 (real tokens are >= 0)
- Both masks are all False
- Final embedding = 0 + 0 = 0
- **Address information is completely ignored!**


Where Address Info Should Be Applied
--------------------------------------
Address embeddings should be added to these token types:

1. Opcodes with addresses:
   push(0x1000:0.1:0.5:0.2) → token_id=525 (push), positions=(0.1, 0.5, 0.2)
   Should get address embedding ✓

2. 'address' tokens:
   address(0x2000:0.2:0.6:0.3) → token_id=??? (may not be in vocab)
   Should get address embedding ✓

3. 'daddr' tokens:
   daddr(0x3000:0.3:0.7:0.4) → token_id=??? (may not be in vocab)
   Should get address embedding ✓


Correct Logic
--------------
Apply address embeddings to ALL tokens with valid positions:

```python
# Check if positions are valid
address_mask = (binary_pos >= 0) & (function_pos >= 0) & (bb_pos >= 0)

# Apply embedding to all tokens with valid addresses
embedding = self.code_address_projection(hierarchical_features)

# Zero out for tokens without valid positions
embedding = embedding * address_mask.unsqueeze(-1).float()
```

This ensures:
- Opcodes with addresses get embeddings ✓
- 'address' tokens get embeddings ✓
- 'daddr' tokens get embeddings ✓
- Regular tokens (rbp, rsp) don't get embeddings ✓


================================================================================
SUMMARY: COMPLETE PIPELINE
================================================================================

1. Data Generation
   Input: Raw binaries (O0, O1, O2, O3)
   Output: Address-aware JSON corpus
   Files: PalmTree/corpus/Ox/train/*.json

2. Vocabulary
   Input: Address-aware corpus
   Output: vocab.txt, vocab.pkl
   Files: pretrain/address_aware/vocab*.txt

3. Pretraining
   Input: Address-aware corpus + vocab
   Tasks: MLM + JTP
   Output: Pretrained model
   Files: pretrain/address_aware/addressaware_pretrained/
   Script: train_addressaware.sh

4. Finetuning
   Input: Pretrained model + triplet data
   Task: Function similarity (triplet learning)
   Output: Finetuned model
   Files: output/addressaware_finetune/finetune_epoch_*.pt
   Script: run_finetune_addressaware.sh

5. Evaluation
   Input: Finetuned model + evaluation pools
   Metrics: MRR, Recall@1/5/10
   Output: Results JSON
   Files: output/jtrans/addressaware_eval_*.json
   Script: run_addressaware_pool_evaluation.sh


Key Configuration
------------------
Pretraining:
- Batch size: 32
- LR: 1e-4
- Epochs: 10
- Max len: 512

Finetuning:
- Batch size: 16
- LR: 2e-5
- Epochs: 15
- Margin: 0.2
- Embedding dim: 256
- Target opt: O3

Evaluation:
- Pool sizes: 100, 1000, 10000
- Query opts: O0, O1, O2
- Pool opt: O3


Expected Results
-----------------
With properly working address embeddings:
- MRR: 0.40-0.60
- Recall@1: 0.30-0.50
- Recall@10: 0.70-0.85

With broken address embeddings (current):
- MRR: 0.02-0.10 (similar to random)
- Recall@1: 0.00-0.05
- Recall@10: 0.10-0.20


================================================================================
DEBUGGING CHECKLIST
================================================================================

If results are poor, check:

1. ✓ Data has address information
   grep "0x.*:.*:.*:" corpus_file.json

2. ✓ Vocab loaded correctly
   Check vocab_stoi is passed to embeddings

3. ✓ Address embeddings applied
   Add logging: print(embedding.norm()) should be > 0

4. ✓ Training uses --target_opt O3
   Check run_finetune_addressaware.sh

5. ✓ Evaluation pools match training
   Queries from Ox, pool from O3

6. ✓ Projection layer loaded
   Check config.json has use_projection=True

7. ✓ Correct checkpoint evaluated
   Use finetune_epoch_15.pt, not pretrained model

================================================================================
