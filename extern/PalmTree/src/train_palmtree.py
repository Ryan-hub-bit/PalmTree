"""
Standard PalmTree Training Script

Uses standard BERT without address-aware features.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, 'src')

from config import *
import palmtree
from palmtree import dataset
from palmtree import trainer
from palmtree.model import BERT

print(palmtree.__file__)
print("\n" + "="*80)
print("Standard PalmTree Training")
print("="*80 + "\n")

# ============================================================================
# Configuration - Using parameters from config.py and additional settings
# ============================================================================
vocab_path = "./vocab"
train_cfg_dataset = "data/training/cdfg_bert_1/cfg_train.txt"
train_dfg_dataset = "data/training/cdfg_bert_1/dfg_train.txt"
test_cfg_dataset = "data/training/cdfg_bert_1/cfg_test.txt"
test_dfg_dataset = "data/training/cdfg_bert_1/dfg_test.txt"
output_path = "cdfg_bert_1/transformer"

# Model hyperparameters (use config.py values where available)
VOCAB_SIZE = VOCAB_SIZE if 'VOCAB_SIZE' in dir() else 13000
MIN_FREQ = 1
SEQ_LEN = MAXLEN if 'MAXLEN' in dir() else 20
BATCH_SIZE = 256
NUM_WORKERS = 10

HIDDEN_SIZE = 128
N_LAYERS = 12
ATTN_HEADS = 8
DROPOUT = 0.1

# Training hyperparameters
LEARNING_RATE = 1e-5
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10000
NUM_EPOCHS = 20

# CUDA settings (use config.py values)
USE_CUDA = USE_CUDA if 'USE_CUDA' in dir() else True
CUDA_DEVICES = DEVICES if 'DEVICES' in dir() else [0]
LOG_FREQ = 100

# ============================================================================
# Build Vocabulary
# ============================================================================
print("Building vocabulary from training data...")
with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
    with open(train_dfg_dataset, "r", encoding="utf-8") as f2:
        vocab = dataset.WordVocab([f1, f2], max_size=VOCAB_SIZE, min_freq=MIN_FREQ)

print(f"VOCAB SIZE: {len(vocab)}")
vocab.save_vocab(vocab_path)

# ============================================================================
# Load Vocabulary
# ============================================================================
print(f"\nLoading Vocab from {vocab_path}")
vocab = dataset.WordVocab.load_vocab(vocab_path)
print(f"Vocab Size: {len(vocab)}")
# print(vocab.itos)

# ============================================================================
# Load Training Dataset (Standard)
# ============================================================================
print("\nLoading Standard Training Dataset...")
print(f"  CFG: {train_cfg_dataset}")
print(f"  DFG: {train_dfg_dataset}")
train_dataset = dataset.BERTDataset(
    dfg_corpus_path=train_dfg_dataset,
    cfg_corpus_path=train_cfg_dataset,
    vocab=vocab,
    seq_len=SEQ_LEN,
    corpus_lines=None,
    on_memory=True
)
print(f"  Loaded {len(train_dataset)} training samples")

# ============================================================================
# Load Test Dataset (Standard) - Optional
# ============================================================================
test_dataset = None
if test_cfg_dataset is not None and test_dfg_dataset is not None:
    try:
        print("\nLoading Standard Test Dataset...")
        print(f"  CFG: {test_cfg_dataset}")
        print(f"  DFG: {test_dfg_dataset}")
        test_dataset = dataset.BERTDataset(
            dfg_corpus_path=test_dfg_dataset,
            cfg_corpus_path=test_cfg_dataset,
            vocab=vocab,
            seq_len=SEQ_LEN,
            corpus_lines=None,
            on_memory=True
        )
        print(f"  Loaded {len(test_dataset)} test samples")
    except FileNotFoundError:
        print("  Test dataset files not found, skipping test evaluation")
        test_dataset = None

# ============================================================================
# Create DataLoaders
# ============================================================================
print("\nCreating DataLoaders...")
train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True
)

test_data_loader = None
if test_dataset is not None:
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

# ============================================================================
# Build Standard BERT Model
# ============================================================================
print("\nBuilding Standard BERT model...")
print(f"  Hidden size: {HIDDEN_SIZE}")
print(f"  Layers: {N_LAYERS}")
print(f"  Attention heads: {ATTN_HEADS}")
print(f"  Dropout: {DROPOUT}")

bert = BERT(
    vocab_size=len(vocab),
    hidden=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    attn_heads=ATTN_HEADS,
    dropout=DROPOUT
)

# ============================================================================
# Create Standard BERT Trainer
# ============================================================================
print("\nCreating Standard BERT Trainer...")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Betas: {BETAS}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  CUDA: {USE_CUDA}")
if USE_CUDA:
    print(f"  CUDA devices: {CUDA_DEVICES}")

trainer_instance = trainer.BERTTrainer(
    bert=bert,
    vocab_size=len(vocab),
    train_dataloader=train_data_loader,
    test_dataloader=test_data_loader,
    lr=LEARNING_RATE,
    betas=BETAS,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    with_cuda=USE_CUDA,
    cuda_devices=CUDA_DEVICES,
    log_freq=LOG_FREQ,
    mode='original'  # KEY: Use standard mode
)

# ============================================================================
# Training Loop
# ============================================================================
print("\n" + "="*80)
print("Training Start")
print("="*80 + "\n")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 80)
    
    # Train
    trainer_instance.train(epoch)
    
    # Save checkpoint
    trainer_instance.save(epoch, output_path)
    print(f"Model saved to {output_path}/bert_trained_{epoch}.model")
    
    # Test (if test dataset available)
    if test_data_loader is not None:
        print("\nRunning test evaluation...")
        trainer_instance.test(epoch)

print("\n" + "="*80)
print("Training Complete!")
print("="*80)     
