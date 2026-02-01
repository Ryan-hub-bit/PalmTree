#!/usr/bin/env python3
"""
调试tokenization，对比evaluation和finetune的实际输出
"""
import torch
import json
from transformers import BertTokenizer
import sys

# 加载tokenizer
print("加载tokenizer...")
tokenizer = BertTokenizer.from_pretrained('/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')
print(f"Vocab size: {len(tokenizer)}")
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

# 加载evaluation数据的一个样本
print("\n加载evaluation数据...")
with open('/data/kun/jtrans/addressaware/eval/func_blocks_addr.json', 'r') as f:
    data = json.load(f)

# 取第一个函数
func_id = '0'
func_data = data[func_id]

print(f"\n函数ID: {func_id}")
print(f"Binary: {func_data['binary_name']}")
print(f"Function: {func_data['function_name']}")
print(f"Opt level: {func_data['optimization_level']}")
print(f"Num instructions: {func_data['num_instructions']}")

# 原始instructions
instructions_str = func_data['instructions']
print(f"\n原始instructions字符串长度: {len(instructions_str)}")
print(f"前200字符: {instructions_str[:200]}")

# Split by tab
instructions = instructions_str.split('\t')
print(f"\nSplit后instruction数量: {len(instructions)}")

# 检查空字符串
empty_count = sum(1 for inst in instructions if not inst.strip())
print(f"空字符串数量: {empty_count}")

# 显示每条指令
print("\n每条指令详情:")
for i, inst in enumerate(instructions):
    inst_stripped = inst.strip()
    if inst_stripped:
        print(f"  [{i}] ({len(inst_stripped)} chars): {inst_stripped[:80]}")
    else:
        print(f"  [{i}] (EMPTY)")

# 使用evaluation的_parse_instruction函数
import re

def _parse_instruction(inst_text):
    """Parse a single instruction (SAME as evaluation)."""
    tokens = []
    positions = []
    var_offsets = []
    
    parts = inst_text.split()
    for part in parts:
        # Handle opcode(...) pattern
        opcode_match = re.match(r'([a-z0-9_\.]+)\((0x[0-9a-f]+:[0-9\.]+:[0-9\.]+:[0-9\.]+)\)', part)
        if opcode_match:
            opcode = opcode_match.group(1)
            pos_str = opcode_match.group(2)
            tokens.append(opcode)
            
            pos_parts = pos_str.split(':')
            binary_pos = float(pos_parts[1]) if len(pos_parts) > 1 else -1.0
            function_pos = float(pos_parts[2]) if len(pos_parts) > 2 else -1.0
            bb_pos = float(pos_parts[3]) if len(pos_parts) > 3 else -1.0
            positions.append((binary_pos, function_pos, bb_pos))
            var_offsets.append(-1)
            continue
        
        # Handle address(...) pattern
        addr_match = re.match(r'(address|daddr)\((0x[0-9a-f]+:[0-9\.]+:[0-9\.]+:[0-9\.]+)\)', part)
        if addr_match:
            addr_type = addr_match.group(1)
            pos_str = addr_match.group(2)
            tokens.append(addr_type)
            
            pos_parts = pos_str.split(':')
            binary_pos = float(pos_parts[1]) if len(pos_parts) > 1 else -1.0
            function_pos = float(pos_parts[2]) if len(pos_parts) > 2 else -1.0
            bb_pos = float(pos_parts[3]) if len(pos_parts) > 3 else -1.0
            positions.append((binary_pos, function_pos, bb_pos))
            var_offsets.append(-1)
            continue
        
        # Handle var(...) pattern
        var_match = re.match(r'var\(([0-9]+)\)', part)
        if var_match:
            offset = int(var_match.group(1))
            tokens.append('var')
            positions.append((-1.0, -1.0, -1.0))
            var_offsets.append(offset)
            continue
        
        # Regular token (registers, immediates, etc.)
        tokens.append(part)
        positions.append((-1.0, -1.0, -1.0))
        var_offsets.append(-1)
    
    return tokens, positions, var_offsets

# Parse函数
print("\n\n" + "="*80)
print("使用evaluation的tokenization逻辑:")
print("="*80)

all_tokens = []
all_positions = []
all_var_offsets = []
all_segments = []

# Add <sos>
all_tokens.append('<sos>')
all_positions.append((-1.0, -1.0, -1.0))
all_var_offsets.append(-1)
all_segments.append(1)

for inst_idx, inst_text in enumerate(instructions):
    inst_text = inst_text.strip()
    if not inst_text:
        continue
    
    tokens, positions, var_offsets = _parse_instruction(inst_text)
    inst_segment = inst_idx + 1
    
    all_tokens.extend(tokens)
    all_positions.extend(positions)
    all_var_offsets.extend(var_offsets)
    all_segments.extend([inst_segment] * len(tokens))

# Add <eos>
last_segment = inst_idx + 1 if instructions else 1
all_tokens.append('<eos>')
all_positions.append((-1.0, -1.0, -1.0))
all_var_offsets.append(-1)
all_segments.append(last_segment)

print(f"\n总token数: {len(all_tokens)}")
print(f"前30个tokens: {all_tokens[:30]}")
print(f"对应的segments: {all_segments[:30]}")

# Convert to IDs
token_ids = []
unk_count = 0
for tok in all_tokens:
    token_id = tokenizer.convert_tokens_to_ids(tok)
    if token_id is None or token_id == tokenizer.unk_token_id:
        token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 1
        unk_count += 1
    token_ids.append(token_id)

print(f"\nUNK token数量: {unk_count}/{len(token_ids)} ({100*unk_count/len(token_ids):.1f}%)")
print(f"前30个token IDs: {token_ids[:30]}")

# 检查是否有异常的token ID
max_id = max(token_ids)
print(f"\n最大token ID: {max_id} (vocab size: {len(tokenizer)})")
if max_id >= len(tokenizer):
    print("⚠️  警告：存在超出vocabulary范围的token ID！")

# 对比与finetune数据
print("\n\n" + "="*80)
print("检查是否匹配finetune预期:")
print("="*80)

# 检查key tokens
expected_tokens = ['<sos>', '<eos>', 'lea', 'mov', 'cmp', 'jz', 'rax', 'rdi']
for tok in expected_tokens:
    tok_id = tokenizer.convert_tokens_to_ids(tok)
    in_data = tok in all_tokens
    print(f"  {tok:12s}: ID={tok_id:6d}  在数据中={in_data}")

print("\n结论:")
print("="*80)
if unk_count > len(token_ids) * 0.1:
    print(f"✗ UNK比例过高 ({100*unk_count/len(token_ids):.1f}%)，可能vocabulary不匹配")
else:
    print(f"✓ UNK比例正常 ({100*unk_count/len(token_ids):.1f}%)")

if max_id < len(tokenizer):
    print("✓ 所有token ID在vocabulary范围内")
else:
    print("✗ 存在超出vocabulary的token ID")
