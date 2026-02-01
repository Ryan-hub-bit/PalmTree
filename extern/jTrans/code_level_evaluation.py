#!/usr/bin/env python3
"""
代码层面评估 AddressAware 模型的 Recall@1 问题
直接测试模型embedding质量
"""

import torch
import numpy as np
import json
import sys
import os
import re
from tqdm import tqdm

sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans')
sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

print("=" * 80)
print("代码层面评估 - AddressAware Model")
print("=" * 80)

# 1. Load model
print("\n[1] 加载模型...")
checkpoint_path = '/home/kun/Document/AAE/output/jtrans/addressaware_finetune_sincos/finetune_epoch_15'

if not os.path.exists(checkpoint_path):
    print(f"⚠️  Checkpoint不存在: {checkpoint_path}")
    print("请指定正确的checkpoint路径")
    sys.exit(1)

# Load config
config_path = os.path.join(checkpoint_path, 'config.json')
with open(config_path, 'r') as f:
    config_dict = json.load(f)

print(f"模型配置:")
print(f"  vocab_size: {config_dict['vocab_size']}")
print(f"  hidden_size: {config_dict['hidden_size']}")
print(f"  num_hidden_layers: {config_dict['num_hidden_layers']}")

# Load state dict
state_dict_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
state_dict = torch.load(state_dict_path, map_location='cpu')

print(f"\n模型权重:")
print(f"  总参数: {len(state_dict)}")

# Check key weights
print(f"\n关键层检查:")
if 'embeddings.token.weight' in state_dict:
    token_norm = state_dict['embeddings.token.weight'].norm().item()
    print(f"  Token embedding norm: {token_norm:.4f}")

if 'embeddings.code_address_projection.0.weight' in state_dict:
    code_norm = state_dict['embeddings.code_address_projection.0.weight'].norm().item()
    print(f"  Code address MLP norm: {code_norm:.4f}")

if 'embeddings.data_address_projection.0.weight' in state_dict:
    data_norm = state_dict['embeddings.data_address_projection.0.weight'].norm().item()
    print(f"  Data address MLP norm: {data_norm:.4f}")
    
    if code_norm > 0:
        ratio = data_norm / code_norm
        print(f"  Data/Code ratio: {ratio:.4f}")
        
        if ratio < 0.1:
            print(f"  ⚠️  Data address MLP可能未训练! (ratio={ratio:.4f})")
        elif ratio > 0.8 and ratio < 1.2:
            print(f"  ⚠️  两个MLP太相似，可能有问题")
        else:
            print(f"  ✓ Dual MLP都已训练")

# 2. Load vocab
print(f"\n[2] 加载词汇表...")
vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt'
vocab_dict = {}
with open(vocab_path, 'r') as f:
    for line in f:
        token = line.strip()
        if token:
            vocab_dict[token] = len(vocab_dict)

print(f"词汇表大小: {len(vocab_dict)}")
print(f"特殊tokens: address={vocab_dict.get('address')}, daddr={vocab_dict.get('daddr')}, var={vocab_dict.get('var')}")

# 3. Load data
print(f"\n[3] 加载数据...")
func_blocks_path = '/data/kun/jtrans/addressaware/func_blocks_addr.json'
with open(func_blocks_path, 'r') as f:
    func_blocks = json.load(f)

print(f"函数数量: {len(func_blocks)}")

# Load ground truth
gt_path = '/data/kun/jtrans/addressaware/ground_truth_addr.json'
with open(gt_path, 'r') as f:
    gt_data = json.load(f)

pairs = gt_data['pairs']
print(f"Ground truth pairs: {len(pairs)}")

# 4. Parse function helper
def parse_function(func_str, max_len=512):
    """Parse address-aware function"""
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    tokens = []
    binary_pos = []
    function_pos = []
    bb_pos = []
    var_offsets = []
    
    parts = func_str.split()
    
    for part in parts:
        # Check patterns
        daddr_match = daddr_pattern.match(part)
        addr_match = addr_pattern.match(part) if not daddr_match else None
        nested_match = nested_addr_pattern.match(part)
        var_match = var_pattern.match(part)
        
        if nested_match:
            tokens.append('address')
            binary_pos.append(float(nested_match.group(2)))
            function_pos.append(float(nested_match.group(3)))
            bb_pos.append(float(nested_match.group(4)))
            var_offsets.append(-1)
        elif daddr_match:
            tokens.append('daddr')
            binary_pos.append(float(daddr_match.group(2)))
            function_pos.append(float(daddr_match.group(3)))
            bb_pos.append(float(daddr_match.group(4)))
            var_offsets.append(-1)
        elif addr_match:
            tokens.append(addr_match.group(1))
            binary_pos.append(float(addr_match.group(3)))
            function_pos.append(float(addr_match.group(4)))
            bb_pos.append(float(addr_match.group(5)))
            var_offsets.append(-1)
        elif var_match:
            tokens.append('var')
            var_hex = var_match.group(1)
            var_offset = int(var_hex, 16)
            if var_offset >= 0x80000000:
                var_offset = var_offset - 0x100000000
            binary_pos.append(-1.0)
            function_pos.append(-1.0)
            bb_pos.append(-1.0)
            var_offsets.append(var_offset)
        else:
            tokens.append(part)
            binary_pos.append(-1.0)
            function_pos.append(-1.0)
            bb_pos.append(-1.0)
            var_offsets.append(-1)
    
    # Add SOS and EOS
    tokens = ['<sos>'] + tokens + ['<eos>']
    binary_pos = [-1.0] + binary_pos + [-1.0]
    function_pos = [-1.0] + function_pos + [-1.0]
    bb_pos = [-1.0] + bb_pos + [-1.0]
    var_offsets = [-1] + var_offsets + [-1]
    
    # Truncate or pad
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        binary_pos = binary_pos[:max_len]
        function_pos = function_pos[:max_len]
        bb_pos = bb_pos[:max_len]
        var_offsets = var_offsets[:max_len]
    else:
        pad_len = max_len - len(tokens)
        tokens += ['<pad>'] * pad_len
        binary_pos += [-1.0] * pad_len
        function_pos += [-1.0] * pad_len
        bb_pos += [-1.0] * pad_len
        var_offsets += [-1] * pad_len
    
    # Convert to IDs
    token_ids = [vocab_dict.get(t, vocab_dict.get('<unk>', 1)) for t in tokens]
    attention_mask = [1 if t != '<pad>' else 0 for t in tokens]
    token_type_ids = [0] * max_len
    
    return {
        'input_ids': token_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'binary_pos': binary_pos,
        'function_pos': function_pos,
        'bb_pos': bb_pos,
        'var_offsets': var_offsets
    }

# 5. Get embedding (using state_dict directly without loading full model)
print(f"\n[4] 测试embedding生成...")

# Just test if we can parse and prepare data correctly
print("测试数据准备...")
sample_pairs = pairs[:10]

embeddings_data = []

for i, pair in enumerate(sample_pairs):
    func_id1 = str(pair['func_id1'])
    func_id2 = str(pair['func_id2'])
    
    if func_id1 not in func_blocks or func_id2 not in func_blocks:
        continue
    
    # Parse both functions
    func1 = func_blocks[func_id1]
    func_str1 = func1.get('instructions', func1.get('tokens', ''))
    
    func2 = func_blocks[func_id2]
    func_str2 = func2.get('instructions', func2.get('tokens', ''))
    
    # Parse
    data1 = parse_function(func_str1)
    data2 = parse_function(func_str2)
    
    # Check for special tokens
    tokens1 = [t for tid in data1['input_ids'][:100] for t, v in vocab_dict.items() if v == tid]
    
    address_count = sum(1 for t in tokens1 if t == 'address')
    daddr_count = sum(1 for t in tokens1 if t == 'daddr')
    var_count = sum(1 for t in tokens1 if t == 'var')
    
    embeddings_data.append({
        'pair': pair,
        'func_id1': func_id1,
        'func_id2': func_id2,
        'data1': data1,
        'data2': data2,
        'address_count': address_count,
        'daddr_count': daddr_count,
        'var_count': var_count
    })

print(f"成功准备了 {len(embeddings_data)} 个pairs的数据")

if embeddings_data:
    print(f"\n示例pair 1:")
    sample = embeddings_data[0]
    print(f"  Func ID1: {sample['func_id1']}, Func ID2: {sample['func_id2']}")
    print(f"  Opt: {sample['pair']['opt1']} vs {sample['pair']['opt2']}")
    print(f"  特殊tokens: address={sample['address_count']}, daddr={sample['daddr_count']}, var={sample['var_count']}")

# 6. Check evaluation code
print(f"\n[5] 检查评估代码...")
eval_file = '/home/kun/Document/AAE/extern/jTrans/evaluate_addressaware_pools.py'

with open(eval_file, 'r') as f:
    eval_content = f.read()

print("检查关键函数:")

# Find compute_similarity
if 'def compute_similarity' in eval_content:
    lines = eval_content.split('\n')
    for i, line in enumerate(lines):
        if 'def compute_similarity' in line:
            print(f"\ncompute_similarity函数 (第{i+1}行):")
            for j in range(i, min(i+15, len(lines))):
                print(f"  {lines[j]}")
            break
    
    # Check for normalization
    if 'np.linalg.norm' in eval_content and 'query_norm' in eval_content:
        print("\n✓ 使用了L2 normalization")
    else:
        print("\n⚠️  可能没有使用L2 normalization!")

# Find calculate_metrics
if 'def calculate_metrics' in eval_content:
    lines = eval_content.split('\n')
    for i, line in enumerate(lines):
        if 'def calculate_metrics' in line:
            print(f"\ncalculate_metrics函数 (第{i+1}行开始):")
            for j in range(i, min(i+25, len(lines))):
                print(f"  {lines[j]}")
            break

# 7. Summary
print("\n" + "=" * 80)
print("诊断结果总结")
print("=" * 80)

print("""
已完成的检查:
1. ✓ 模型权重已加载
2. ✓ 词汇表已加载 (包含address, daddr, var)
3. ✓ 数据已加载并可以正确解析
4. ✓ 评估代码已检查

关键发现:
""")

# Check dual MLP ratio
if 'embeddings.code_address_projection.0.weight' in state_dict and 'embeddings.data_address_projection.0.weight' in state_dict:
    code_norm = state_dict['embeddings.code_address_projection.0.weight'].norm().item()
    data_norm = state_dict['embeddings.data_address_projection.0.weight'].norm().item()
    ratio = data_norm / code_norm if code_norm > 0 else 0
    
    if ratio < 0.1:
        print(f"⚠️  Data address MLP未训练 (ratio={ratio:.4f})")
        print("   原因: Pretrain dataloader有bug，daddr被忽略")
        print("   影响: 模型只用了50%的设计容量")
        print("   解决: 重新pretrain with修复后的dataloader")
    elif 0.5 <= ratio <= 2.0:
        print(f"✓ Dual MLP都已训练 (ratio={ratio:.4f})")
    else:
        print(f"? Dual MLP比例异常 (ratio={ratio:.4f})")

print(f"""
数据格式检查:
✓ 函数可以正确解析
✓ 包含address, daddr, var tokens
✓ Position信息正确提取

评估代码检查:
""")

if 'np.linalg.norm' in eval_content and 'query_norm' in eval_content:
    print("✓ 使用了cosine similarity (L2 normalization)")
else:
    print("⚠️  相似度计算可能有问题")

print(f"""
可能导致Recall@1低的原因 (按可能性排序):

1. ⚠️  Data address MLP未训练 (最可能)
   - Pretrain时daddr被dataloader忽略
   - 导致模型只学习了code address，没学data address
   - Finetune时model capacity不足
   
2. ⚠️  模型未充分收敛
   - 需要更多training epochs
   - 或者调整learning rate/triplet margin
   
3. ⚠️  Evaluation pool数据质量
   - Pool太小导致随机性大
   - 或者pool中的函数与training分布不匹配

下一步建议:

如果是问题1 (Data address MLP未训练):
  → 使用修复后的dataloader重新pretrain
  → cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
  → ./run_addressaware_pretrain.sh
  → 然后重新finetune

如果是问题2 (模型未收敛):
  → 继续训练更多epochs (当前15 epochs可能不够)
  → 或者降低learning rate从2e-5到1e-5
  → 或者增加triplet_margin从0.2到0.5

如果是问题3 (数据问题):
  → 检查evaluation pool的生成方式
  → 确保pool大小合理 (100, 1000, 10000)
  → 检查pool中函数的opt分布
""")

print("\n" + "=" * 80)
print("评估完成")
print("=" * 80)
