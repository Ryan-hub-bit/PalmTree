╔══════════════════════════════════════════════════════════════════════════════╗
║          UPDATED: Address-Aware BERT Architecture with FROZEN PalmTree       ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                            EMBEDDING LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: token_ids, segment_labels, binary_pos, function_pos, bb_pos        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │ 1. Token Embedding (from PalmTree)              [✗ FROZEN]     │        │
│  │    - Learned token representations                              │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                              +                                               │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │ 2. Sequence Positional Encoding (standard)      [✗ FROZEN]     │        │
│  │    - Sin/cos encoding based on token position in sequence       │        │
│  │    - Same as PalmTree's position embedding                      │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                              +                                               │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │ 3. Address Positional Encoding (NEW!)           [✓ TRAINABLE]  │        │
│  │    - Sin/cos(binary_pos)    → 8 decimal places                 │        │
│  │    - Sin/cos(function_pos)  → 6 decimal places                 │        │
│  │    - Sin/cos(bb_pos)        → 4 decimal places                 │        │
│  │    - Learnable weights to balance 3 levels                      │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                              +                                               │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │ 4. Segment Embedding (from PalmTree)            [✗ FROZEN]     │        │
│  │    - For NSP task (segment 0 or 1)                             │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                              ↓                                               │
│                      Layer Norm + Dropout                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRANSFORMER BLOCKS                                    │
│                    (from PalmTree) [✗ FROZEN]                               │
│                                                                              │
│  Multi-Head Self-Attention + Feed-Forward × N layers                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TASK HEADS                                          │
│                                                                              │
│  ┌──────────────────────────┐    ┌──────────────────────────┐             │
│  │   MLM Head               │    │   NSP Head               │             │
│  │   [✓ TRAINABLE]          │    │   [✓ TRAINABLE]          │             │
│  │   Predict masked tokens  │    │   Binary classification  │             │
│  └──────────────────────────┘    └──────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                              TRAINING SUMMARY
═══════════════════════════════════════════════════════════════════════════════

FROZEN Components (from PalmTree):
  ✗ Token Embeddings             - Pre-trained token representations
  ✗ Sequence Positional Encoding - Standard sin/cos for token order
  ✗ Segment Embeddings           - For NSP task
  ✗ Transformer Blocks           - All 12 layers with attention

TRAINABLE Components (NEW):
  ✓ Address Positional Encoding  - Hierarchical position (binary/func/bb)
  ✓ MLM Head                     - Predicts masked tokens
  ✓ NSP Head                     - Next sentence prediction

═══════════════════════════════════════════════════════════════════════════════
                               KEY CHANGES
═══════════════════════════════════════════════════════════════════════════════

✅ Added sequence positional encoding (like PalmTree)
✅ Automatic freezing of ALL PalmTree components
✅ Only address embeddings + task heads are trained
✅ Clean separation: PalmTree (frozen) + Address-aware (new)

Example Forward Pass:
  token_ids=[4, 156, 23, ...]           → Token Embedding (frozen)
  sequence=[0, 1, 2, 3, ...]            → Sequence Position (frozen)
  binary_pos=[0.21, 0.22, 0.23, ...]    → Address Position (trainable)
  function_pos=[0.30, 0.74, 0.96, ...]  → Address Position (trainable)
  bb_pos=[0.44, 0.00, 1.00, ...]        → Address Position (trainable)
  segment=[0, 0, 1, 1, ...]             → Segment Embedding (frozen)
