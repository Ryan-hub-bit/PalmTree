#!/usr/bin/env python3
"""
检查 Finetune 时 function block 的分词情况
查看是否正确处理了 address 和 daddr tokens
"""

import json
import sys
import re

print("="*80)
print("Finetune 分词检查")
print("="*80)
print()

# 加载数据
func_blocks_path = '/data/kun/jtrans/addressaware/func_blocks_addr.json'
vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt'

print(f"加载数据: {func_blocks_path}")
print(f"加载词表: {vocab_path}")
print()

with open(func_blocks_path, 'r') as f:
    func_blocks = json.load(f)

# 加载词表
vocab_stoi = {}
with open(vocab_path, 'r') as f:
    for idx, line in enumerate(f):
        token = line.strip()
        vocab_stoi[token] = idx

print(f"词表大小: {len(vocab_stoi)}")
print(f"特殊 tokens:")
print(f"  'address' ID: {vocab_stoi.get('address', 'NOT FOUND')}")
print(f"  'daddr' ID: {vocab_stoi.get('daddr', 'NOT FOUND')}")
print(f"  'var' ID: {vocab_stoi.get('var', 'NOT FOUND')}")
print()

# 模拟 finetune 的分词逻辑 (来自 data_json.py)
addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')

def parse_instruction(inst_text):
    """模拟 FunctionDataset_CL_AddressAware_JSON 的 _parse_instruction"""
    tokens = []
    positions = []
    var_offsets = []
    
    parts = inst_text.strip().split()
    
    for part in parts:
        # Check for opcode(address) pattern
        match = addr_pattern.match(part)
        if match:
            opcode = match.group(1)
            binary_pos = float(match.group(3))
            function_pos = float(match.group(4))
            bb_pos = float(match.group(5))
            
            tokens.append(opcode)
            positions.append((binary_pos, function_pos, bb_pos))
            var_offsets.append(-1)
            continue
        
        # Check for nested address() pattern
        nested_match = nested_addr_pattern.match(part)
        if nested_match:
            binary_pos = float(nested_match.group(2))
            function_pos = float(nested_match.group(3))
            bb_pos = float(nested_match.group(4))
            
            tokens.append('address')
            positions.append((binary_pos, function_pos, bb_pos))
            var_offsets.append(-1)
            continue
        
        # Check for daddr() pattern
        daddr_match = daddr_pattern.match(part)
        if daddr_match:
            binary_pos = float(daddr_match.group(2))
            function_pos = float(daddr_match.group(3))
            bb_pos = float(daddr_match.group(4))
            
            tokens.append('daddr')
            positions.append((binary_pos, function_pos, bb_pos))
            var_offsets.append(-1)
            continue
        
        # Check for var() pattern
        var_match = var_pattern.match(part)
        if var_match:
            hex_offset = var_match.group(1)
            offset_val = int(hex_offset, 16)
            if offset_val > 0x7FFFFFFFFFFFFFFF:
                offset_val = offset_val - 0x10000000000000000
            
            tokens.append('var')
            positions.append((-1.0, -1.0, -1.0))
            var_offsets.append(offset_val)
            continue
        
        # Regular token
        tokens.append(part)
        positions.append((-1.0, -1.0, -1.0))
        var_offsets.append(-1)
    
    return tokens, positions, var_offsets

# 检查前5个函数
print("="*80)
print("检查前5个函数的分词情况")
print("="*80)
print()

for func_id in range(min(5, len(func_blocks))):
    func_id_str = str(func_id)
    if func_id_str not in func_blocks:
        continue
    
    func = func_blocks[func_id_str]
    func_str = func.get('instructions', func.get('tokens', ''))
    
    print(f"{'='*80}")
    print(f"Function {func_id}: {func.get('function_name', 'unknown')}")
    print(f"Binary: {func.get('binary_name', 'unknown')}")
    print(f"Opt level: {func.get('optimization_level', 'unknown')}")
    print(f"{'='*80}")
    
    # 显示原始tokens (前100个字符)
    print(f"\n原始 tokens (前200字符):")
    print(f"  {func_str[:200]}")
    print()
    
    # 解析所有指令
    instructions = func_str.split('\t')
    
    all_tokens = []
    all_positions = []
    all_var_offsets = []
    
    address_count = 0
    daddr_count = 0
    var_count = 0
    
    for inst in instructions[:10]:  # 只显示前10条指令
        tokens, positions, var_offsets = parse_instruction(inst)
        all_tokens.extend(tokens)
        all_positions.extend(positions)
        all_var_offsets.extend(var_offsets)
        
        address_count += tokens.count('address')
        daddr_count += tokens.count('daddr')
        var_count += tokens.count('var')
    
    # 统计整个函数
    full_tokens = []
    full_address_count = 0
    full_daddr_count = 0
    full_var_count = 0
    
    for inst in instructions:
        tokens, _, _ = parse_instruction(inst)
        full_tokens.extend(tokens)
        full_address_count += tokens.count('address')
        full_daddr_count += tokens.count('daddr')
        full_var_count += tokens.count('var')
    
    print(f"统计 (全函数):")
    print(f"  指令数: {len(instructions)}")
    print(f"  Token总数: {len(full_tokens)}")
    print(f"  'address' 数量: {full_address_count}")
    print(f"  'daddr' 数量: {full_daddr_count}")
    print(f"  'var' 数量: {full_var_count}")
    print()
    
    print(f"前10条指令的解析结果:")
    print()
    
    for i, inst in enumerate(instructions[:10]):
        tokens, positions, var_offsets = parse_instruction(inst)
        
        print(f"  指令 {i+1}: {inst[:60]}{'...' if len(inst) > 60 else ''}")
        print(f"    解析tokens: {' '.join(tokens)}")
        
        # 高亮显示特殊tokens
        special_tokens = []
        for j, tok in enumerate(tokens):
            if tok in ['address', 'daddr', 'var']:
                pos = positions[j]
                if tok == 'var':
                    special_tokens.append(f"{tok}(offset={var_offsets[j]})")
                else:
                    special_tokens.append(f"{tok}(b={pos[0]:.2f},f={pos[1]:.2f},bb={pos[2]:.2f})")
        
        if special_tokens:
            print(f"    特殊tokens: {', '.join(special_tokens)}")
        print()
    
    # 转换为token IDs
    token_ids = []
    unknown_tokens = []
    for tok in all_tokens[:20]:  # 前20个tokens
        tok_id = vocab_stoi.get(tok, vocab_stoi.get('[UNK]', 1))
        token_ids.append(tok_id)
        if tok not in vocab_stoi:
            unknown_tokens.append(tok)
    
    print(f"Token ID 映射 (前20个):")
    for i in range(min(20, len(all_tokens))):
        tok = all_tokens[i]
        tok_id = token_ids[i]
        in_vocab = "✅" if tok in vocab_stoi else "❌"
        print(f"  [{i:2d}] '{tok:15s}' → ID {tok_id:5d} {in_vocab}")
    
    if unknown_tokens:
        print(f"\n⚠️  未知tokens (不在词表中): {set(unknown_tokens)}")
    
    print()

print("="*80)
print("总结")
print("="*80)
print()

print("检查要点:")
print("  1. ✅ 'address' tokens 是否正确提取")
print("  2. ✅ 'daddr' tokens 是否正确提取")
print("  3. ✅ 'var' tokens 是否正确提取")
print("  4. ✅ 位置信息是否正确解析")
print("  5. ✅ Token IDs 是否在词表中")
print()

print("如果发现问题:")
print("  - 'daddr' 数量为0: 检查原始数据是否包含 daddr(...) 格式")
print("  - Token IDs 不在词表: 需要更新词表或检查分词逻辑")
print("  - 位置信息异常: 检查正则表达式匹配")
print()
