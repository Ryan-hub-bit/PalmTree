"""
Test that token_mask_prob and data_percentage parameters work correctly
"""
import pickle
from palmtree.dataset.dataset_baseline import BaselineDataset

# Load vocab
with open('../../../strupos/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

print("="*80)
print("Testing BaselineDataset with custom parameters")
print("="*80)

# Test 1: Custom masking probability (30%)
print("\nTest 1: token_mask_prob=0.30 (should mask ~30% of tokens)")
dataset1 = BaselineDataset(
    dfg_corpus_path='../dstask/funcsim/scope_O0_O3_8/train.dfg',
    cfg_corpus_path='../dstask/funcsim/scope_O0_O3_8/train.cfg',
    vocab=vocab,
    seq_len=512,
    token_mask_prob=0.30,
    data_percentage=0.01  # Use 1% for quick test
)

masked_count = 0
total_count = 0
for i in range(min(10, len(dataset1))):
    data = dataset1[i]
    # Count masked tokens (mask_index in CFG or DFG)
    cfg_masked = sum(1 for t in data["bert_input_cfg"] if t == vocab.mask_index)
    dfg_masked = sum(1 for t in data["bert_input_dfg"] if t == vocab.mask_index)
    cfg_total = sum(1 for t in data["bert_input_cfg"] if t != vocab.pad_index)
    dfg_total = sum(1 for t in data["bert_input_dfg"] if t != vocab.pad_index)
    
    masked_count += cfg_masked + dfg_masked
    total_count += cfg_total + dfg_total

if total_count > 0:
    print(f"  Masking rate: {masked_count}/{total_count} = {100*masked_count/total_count:.2f}%")
    print(f"  Expected: ~30%, Actual: {100*masked_count/total_count:.2f}%")

# Test 2: Data percentage
print(f"\nTest 2: data_percentage=0.01 (should use 1% of data)")
print(f"  Dataset size: {len(dataset1)} samples")
print(f"  Expected: ~3,709 samples (1% of 370,888)")

# Test 3: Default parameters
print("\nTest 3: Default parameters (15% masking, 100% data)")
dataset2 = BaselineDataset(
    dfg_corpus_path='../dstask/funcsim/scope_O0_O3_8/train.dfg',
    cfg_corpus_path='../dstask/funcsim/scope_O0_O3_8/train.cfg',
    vocab=vocab,
    seq_len=512
)

masked_count = 0
total_count = 0
for i in range(min(10, len(dataset2))):
    data = dataset2[i]
    cfg_masked = sum(1 for t in data["bert_input_cfg"] if t == vocab.mask_index)
    dfg_masked = sum(1 for t in data["bert_input_dfg"] if t == vocab.mask_index)
    cfg_total = sum(1 for t in data["bert_input_cfg"] if t != vocab.pad_index)
    dfg_total = sum(1 for t in data["bert_input_dfg"] if t != vocab.pad_index)
    
    masked_count += cfg_masked + dfg_masked
    total_count += cfg_total + dfg_total

if total_count > 0:
    print(f"  Masking rate: {masked_count}/{total_count} = {100*masked_count/total_count:.2f}%")
    print(f"  Expected: ~15%, Actual: {100*masked_count/total_count:.2f}%")
print(f"  Dataset size: {len(dataset2)} samples")
print(f"  Expected: 370,888 samples (100%)")

print("\n" + "="*80)
print("✅ All tests completed!")
print("="*80)
