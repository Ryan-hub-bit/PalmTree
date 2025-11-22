"""
Test script for scope dataloader
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.dataset.vocab import WordVocab
from dataloader_scope import ScopeDataset
from torch.utils.data import DataLoader

# Load vocab
vocab_path = "../pre-trained_model/palmtree/vocab"
vocab = WordVocab.load_vocab(vocab_path)
print(f"Vocabulary size: {len(vocab)}")

# Create scope dataset
scope_data_path = "../data/scope/all_scope.txt"
dataset = ScopeDataset(
    scope_corpus_path=scope_data_path,
    vocab=vocab,
    seq_len=20,
    on_memory=True,
    data_percentage=0.01,  # Use 1% for testing
    train_split=0.9,
    is_train=True
)

print(f"\nDataset size: {len(dataset)}")

# Create dataloader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get one batch
for batch in loader:
    print("\n=== Sample Batch ===")
    print(f"Input shape: {batch['bert_input'].shape}")
    print(f"Segment shape: {batch['segment_label'].shape}")
    print(f"Scope labels: {batch['scope_label']}")
    print(f"Binary pos shape: {batch['binary_pos'].shape}")
    print(f"Function pos shape: {batch['function_pos'].shape}")
    print(f"BB pos shape: {batch['bb_pos'].shape}")
    
    # Decode first sample
    print("\n=== First Sample ===")
    tokens = [vocab.itos[idx.item()] for idx in batch['bert_input'][0] if idx.item() != vocab.pad_index]
    print(f"Tokens: {' '.join(tokens[:20])}")
    print(f"Scope label: {batch['scope_label'][0].item()}")
    
    break

print("\n✓ Scope dataloader test passed!")
