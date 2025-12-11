import os
from config import *
from torch import nn
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import torch
import numpy as np
import eval_utils as utils


palmtree = utils.UsableTransformer(model_path="./palmtree/transformer.ep19", vocab_path="./palmtree/vocab")

# tokens has to be seperated by spaces.

text1 = ["mov rbp rdi", 
        "mov ebx 0x1", 
        "mov rdx rbx", 
        "call memcpy", 
        "mov rcx rax",
        "mov [ rcx + rbx ] 0x0", 
        "mov [ rax ] 0x2e"]
text2 = ["mov rbp rdi", 
        "mov ebx 0x1", 
        "mov rdx rbx", 
        "mov rcx rax",
        "call memcpy", 
        "mov [ rcx + rbx ] 0x0", 
        "mov [ rax ] 0x2e"]

# it is better to make batches as large as possible.
embeddings1 = palmtree.encode(text1)
embeddings2 = palmtree.encode(text2)

# Check if embeddings are identical
print("="*60)
print("EMBEDDING COMPARISON")
print("="*60)
print(f"embeddings1 shape: {embeddings1.shape}")
print(f"embeddings2 shape: {embeddings2.shape}")

# Check if they are exactly the same (for numpy arrays)
are_identical = np.array_equal(embeddings1, embeddings2)
print(f"\nAre embeddings EXACTLY identical? {are_identical}")

if are_identical:
    print("❌ embeddings1 == embeddings2 (IDENTICAL)")
else:
    print("✅ embeddings1 != embeddings2 (DIFFERENT)")

# Calculate similarity metrics
diff = np.abs(embeddings1 - embeddings2)
mean_diff = diff.mean()
max_diff = diff.max()

print(f"\nDifference metrics:")
print(f"  Mean absolute difference: {mean_diff:.8f}")
print(f"  Max absolute difference: {max_diff:.8f}")

# Check with tolerance
are_close = np.allclose(embeddings1, embeddings2, atol=1e-6)
print(f"  Are close (atol=1e-6)? {are_close}")

# Per-instruction comparison
print(f"\nPer-instruction comparison:")
for i in range(min(len(embeddings1), len(embeddings2))):
    inst_diff = np.abs(embeddings1[i] - embeddings2[i]).mean()
    identical = np.array_equal(embeddings1[i], embeddings2[i])
    status = "IDENTICAL" if identical else f"diff={inst_diff:.6f}"
    print(f"  Instruction {i}: {status}")
    if not identical:
        print(f"    text1[{i}]: {text1[i]}")
        print(f"    text2[{i}]: {text2[i]}")

print("="*60)
