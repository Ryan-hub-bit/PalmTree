# CFG Pretraining Approach

## Training Philosophy

**Goal**: Improve representations for **address tokens** and **control flow related tokens**, NOT general semantic understanding.

This is different from PalmTree's semantic pretraining. We focus on:
- Address token representations (`addr_start`, `addr_end`, `addr_code`, `addr_data`)
- Control flow understanding (which BB can reach which BB)
- Hierarchical position encoding for address normalization
- **Preserve PalmTree's semantic knowledge** via selective contrastive learning

## Model Architecture

### Three-Level Embeddings

1. **Semantic Embeddings** (from PalmTree, frozen for first 6631 tokens)
   - Pre-trained instruction semantics
   - New address tokens initialized randomly
   
2. **Hierarchical Position Encoding** (2 separate encoders)
   - **Binary Position Encoder** (base=10000): Global position in binary (0-1 normalized)
   - **Function Position Encoder** (base=1000): Local position within function (can be negative)
   
3. **Sequence Position** (learnable)
   - Standard positional embeddings for token order

### Four Prediction Heads

#### Head 1: Masked Language Modeling (MLM)

- **Purpose**: Learn to predict masked tokens
- **Focus**: Especially important for address tokens
- **Loss**: CrossEntropyLoss over full vocab (6636 tokens)
- **Weight**: 1.0

#### Head 2: Control Flow Graph (CFG) Prediction

- **Purpose**: Learn which BBs are reachable from which BBs
- **Input**: Pair of BB representations (source, target)
- **Output**: Binary classification (reachable/not reachable)
- **Loss**: BCEWithLogitsLoss
- **Weight**: 1.0

#### Head 3: Address Type Classification

- **Purpose**: Learn to distinguish address token types
- **Classes**: 4 types (addr_start, addr_end, addr_code, addr_data)
- **Applied**: Only to masked address tokens
- **Loss**: CrossEntropyLoss (4-way classification)
- **Weight**: 0.5

#### Head 4: Contrastive Learning (Selective)

- **Purpose**: Keep NON-ADDRESS embeddings close to PalmTree's semantic space
- **Key Innovation**: Only applies to existing tokens (0-6630), NOT address tokens (6632-6635)
- **Rationale**: 
  - Address tokens are NEW → should learn freely
  - Existing tokens → preserve PalmTree's semantic knowledge
- **Loss**: CosineEmbeddingLoss (only on non-address tokens)
- **Weight**: 0.3

## Why Selective Contrastive Learning?

This is the **key innovation** that allows us to have the best of both worlds:

| Token Type | Contrastive Loss Applied? | Why? |
|------------|---------------------------|------|
| **Regular tokens** (0-6630) | ✅ YES | Preserve PalmTree's semantic understanding |
| **Address tokens** (6632-6635) | ❌ NO | Free to learn position/CFG information |

### Benefits:
1. **No semantic drift**: Instruction semantics stay anchored to PalmTree
2. **Free address learning**: Address tokens can capture position/CFG without constraint
3. **Easy comparison**: Can directly compare with PalmTree on downstream tasks
4. **Best of both worlds**: Semantic understanding + structural information

## Training Data

- **Positive Pairs**: Real CFG edges from basic block pairs
- **Negative Pairs**: Random BB pairs (50% probability)
- **Masking Strategy**: 
  - 15% of tokens masked
  - Address tokens included in masking
  - Both MLM and address type prediction applied to masked tokens

## Task Weights

```python
TASK_WEIGHTS = {
    'mlm': 1.0,                # Full weight for token prediction
    'cfg_prediction': 1.0,     # Full weight for control flow
    'addr_prediction': 0.5,    # Half weight for address type (auxiliary)
    'contrastive': 0.3         # Moderate weight to preserve semantics
}
```

## Expected Outcomes

After pretraining, the model should have:

1. **Better address token representations** that capture:
   - Position information (binary + function level)
   - Type information (start/end/code/data)
   - Context from surrounding instructions

2. **Better control flow understanding**:
   - Which BBs typically follow which
   - Reachability patterns in CFG

3. **Preserved semantic knowledge**:
   - Instruction semantics stay close to PalmTree
   - Easy to compare on downstream tasks
   - No semantic drift from contrastive anchor

4. **Improved downstream performance** on:
   - Binary similarity
   - Function matching
   - CFG-based analysis tasks

## Comparison with PalmTree

Since we use contrastive learning to stay close to PalmTree's semantic space:

### Fair Comparison Setup
- **Freeze both models** → Use as feature extractors
- **Same downstream task** → Binary/function similarity
- **Metric**: Which embeddings perform better?

### What Each Model Should Excel At:

| Model | Strengths | Expected to Win On |
|-------|-----------|-------------------|
| **PalmTree** | Pure semantic understanding | Semantic-heavy tasks (algorithm matching) |
| **Our Model** | Semantic + CFG + Address | Position-dependent, CFG-based tasks |

### Key Advantage:
Because we use contrastive learning, our model **won't drift far from PalmTree**, making comparison meaningful. We add CFG/address information **on top of** semantic understanding, not **instead of** it.

## Key Differences from Semantic Pretraining

| Aspect | Semantic (PalmTree) | CFG Pretraining (Ours) |
|--------|---------------------|------------------------|
| **Focus** | Instruction semantics | Address + control flow + semantics |
| **Data** | Individual instructions | Basic block pairs (CFG) |
| **Position** | Simple sequence | Hierarchical (binary + function) |
| **Tasks** | MLM only | MLM + CFG + Address + Contrastive |
| **Tokens** | Standard vocab (6631) | Extended with addr_* tokens (6636) |
| **Semantic Preservation** | N/A (original) | Contrastive loss to PalmTree |

## Training Configuration

- **Epochs**: 20
- **Batch Size**: 32
- **Learning Rate**: 1e-4 with cosine annealing
- **Optimizer**: AdamW
- **Validation**: 10% of data, evaluated each epoch
- **Best Model**: Saved based on validation loss
- **PalmTree Model**: Loaded and frozen for contrastive loss reference
