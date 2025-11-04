# PalmTree CFG Pretraining: Training Pipeline Overview

## Introduction
This document describes the training process for our Control Flow Graph (CFG) pretraining model, which leverages PalmTree's semantic embeddings and hierarchical address position encoding. The goal is to enable robust learning of binary code semantics with explicit modeling of both global (binary-level) and local (function-level) address positions.

## Model Architecture
- **Semantic Embeddings:** We use PalmTree, a BERT-based model pre-trained on binary code, to extract semantic embeddings for each basic block (BB).
- **Hierarchical Address Position Encoding:**
  - Two separate Sin/Cos positional encoders are used:
    - **Global (Binary) Position Encoder:** Encodes the position of a BB within the entire binary.
    - **Local (Function) Position Encoder:** Encodes the position of a BB within its function.
  - Both encoders output full 128-dimensional vectors, ensuring maximum expressiveness.
- **Sequence Position Encoding:** Standard Transformer-style positional encoding for sequence order.
- **ThreeLevelEmbedding:** Combines semantic, global, local, and sequence embeddings for each BB.

## Data Preparation
- **Dataset:** BB pairs are parsed from annotated files, with address tags for both binary and function positions.
- **Vocabulary:** WordVocab is used for tokenization and embedding lookup, ensuring compatibility with PalmTree.
- **Data Loader:** Custom `CFGPretrainDataset` loads BB pairs, address positions, and sequence indices for each training sample.

## Training Loop
1. **Initialization:**
   - Load PalmTree semantic embeddings.
   - Initialize SinCosPositionEncoding modules for global and local positions.
   - Set up optimizer, scheduler, and loss function.
2. **Batch Processing:**
   - For each batch, extract BB semantic embeddings, global and local address positions, and sequence indices.
   - Encode positions using the two SinCos encoders.
   - Combine all embeddings and pass through the model.
   - Compute loss and backpropagate.
3. **Checkpointing:**
   - Save model checkpoints at regular intervals.
   - Log training metrics for analysis.

## Validation & Testing
- **Encoding Validation:** Dedicated test scripts (e.g., `test_separate_encodings.py`) ensure that global and local encodings are distinct and expressive.
- **Empirical Evaluation:** Model performance is validated on downstream CFG tasks and semantic metrics.

## Key Design Decisions
- **Separate Position Encoders:** Using two full-dimension encoders for global and local positions maximizes the model's ability to represent hierarchical address semantics.
- **Compatibility:** All components are aligned for seamless integration with PalmTree and downstream tasks.
- **Expressiveness:** Encoding strategies are empirically validated for distinctiveness and coverage.

## References
- PalmTree: BERT-based semantic embedding model for binary code.
- Transformer-style Sin/Cos positional encoding.
- PyTorch for model, training, and data pipeline.

---
For further details, see `train.py`, `config.py`, `data_loader.py`, and the test scripts in the repository.
