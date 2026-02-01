#!/usr/bin/env python3
"""
测试Pretrain实际处理daddr的情况
"""
import sys
sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

from dataloader_addressaware import AddressAwareDataset
from vocab import WordVocab
import re

# 加载vocabulary
print("="*80)
print("加载Pretrain使用的Vocabulary")
print("="*80)

# 尝试多个可能的vocab路径
vocab_paths = [
    '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.pkl',
    '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt',
]

vocab = None
for vocab_path in vocab_paths:
    try:
        vocab = WordVocab.load_vocab(vocab_path)
        print(f"✓ 成功加载: {vocab_path}")
        break
    except:
        continue

if vocab is None:
    print("⚠️  无法加载vocab，使用简单测试")
    # 简单模拟
    vocab_stoi = {
        '<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4,
        'address': 100, 'lea': 200, 'rdi': 300, 'var': 400
    }
    print(f"使用模拟vocab")
else:
    vocab_stoi = vocab.stoi
    print(f"Vocab size: {len(vocab)}")
    
print(f"'address' token ID: {vocab_stoi.get('address', 'NOT FOUND')}")
print(f"'daddr' token ID: {vocab_stoi.get('daddr', 'NOT FOUND')}")

# 测试指令（包含daddr）
test_instruction = "lea(0xf170:0.56621473:0.00000000:0.00000000) rdi daddr(0x1a8c8:0.99618908:0.00000000:0.00000000)"

print("\n" + "="*80)
print("测试指令:")
print("="*80)
print(f"  {test_instruction}")

# 创建一个临时dataset实例来测试_parse_instruction
print("\n" + "="*80)
print("测试旧版本Pretrain的_parse_instruction (修复前)")
print("="*80)

# 模拟旧版本（没有daddr处理）
def old_parse_instruction(inst_text):
    """旧版本：不处理daddr"""
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    tokens = []
    
    # Match opcode
    match = addr_pattern.match(inst_text)
    if match:
        opcode = match.group(1)
        tokens.append(opcode)
        
        # Parse operands
        operands_text = inst_text[match.end():].strip()
        if operands_text:
            for operand in operands_text.split():
                nested_match = nested_addr_pattern.match(operand)
                var_match = var_pattern.match(operand)
                
                if nested_match:
                    tokens.append('address')
                elif var_match:
                    tokens.append('var')
                else:
                    # BUG: daddr(...) 会走到这里，被当成普通token
                    tokens.append(operand)
    
    return tokens

old_tokens = old_parse_instruction(test_instruction)
print(f"Parsed tokens: {old_tokens}")

# 转换为token IDs
old_token_ids = []
unk_count = 0
unk_index = vocab_stoi.get('<unk>', 1) if vocab else 1
for tok in old_tokens:
    tok_id = vocab_stoi.get(tok, unk_index)
    if tok_id == unk_index:
        unk_count += 1
        print(f"  ⚠️  '{tok}' → UNK (ID: {tok_id})")
    else:
        print(f"  ✓ '{tok}' → ID: {tok_id}")
    old_token_ids.append(tok_id)

print(f"\nToken IDs: {old_token_ids}")
print(f"UNK count: {unk_count}/{len(old_tokens)}")

# 测试新版本（修复后）
print("\n" + "="*80)
print("测试新版本Pretrain的_parse_instruction (修复后)")
print("="*80)

def new_parse_instruction(inst_text):
    """新版本：正确处理daddr"""
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    tokens = []
    
    # Match opcode
    match = addr_pattern.match(inst_text)
    if match:
        opcode = match.group(1)
        tokens.append(opcode)
        
        # Parse operands
        operands_text = inst_text[match.end():].strip()
        if operands_text:
            for operand in operands_text.split():
                nested_match = nested_addr_pattern.match(operand)
                daddr_match = daddr_pattern.match(operand)
                var_match = var_pattern.match(operand)
                
                if nested_match:
                    tokens.append('address')
                elif daddr_match:
                    # FIX: daddr也输出'address'
                    tokens.append('address')
                elif var_match:
                    tokens.append('var')
                else:
                    tokens.append(operand)
    
    return tokens

new_tokens = new_parse_instruction(test_instruction)
print(f"Parsed tokens: {new_tokens}")

# 转换为token IDs
new_token_ids = []
unk_count = 0
unk_index = vocab_stoi.get('<unk>', 1) if vocab else 1
for tok in new_tokens:
    tok_id = vocab_stoi.get(tok, unk_index)
    if tok_id == unk_index:
        unk_count += 1
        print(f"  ⚠️  '{tok}' → UNK (ID: {tok_id})")
    else:
        print(f"  ✓ '{tok}' → ID: {tok_id}")
    new_token_ids.append(tok_id)

print(f"\nToken IDs: {new_token_ids}")
print(f"UNK count: {unk_count}/{len(new_tokens)}")

# 对比
print("\n" + "="*80)
print("对比结果")
print("="*80)
print(f"旧版本tokens: {old_tokens}")
print(f"新版本tokens: {new_tokens}")
print()
if old_tokens != new_tokens:
    print("✗ 不一致！")
    print(f"  旧版本会把 'daddr(...)' 当成UNK")
    print(f"  新版本正确解析为 'address' token")
else:
    print("✓ 一致")

# 检查实际的dataloader代码
print("\n" + "="*80)
print("检查当前dataloader_addressaware.py的实际代码")
print("="*80)

with open('/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/dataloader_addressaware.py', 'r') as f:
    content = f.read()
    
    # 检查是否有daddr_match的使用
    if 'daddr_match' in content:
        print("✓ 代码中已经有 daddr_match 变量")
    else:
        print("✗ 代码中没有 daddr_match 变量")
    
    # 检查具体的处理逻辑
    if "elif daddr_match:" in content:
        print("✓ 代码中有 'elif daddr_match:' 分支")
    else:
        print("✗ 代码中没有 'elif daddr_match:' 分支")
    
    # 检查daddr_match后面是append什么
    if "elif daddr_match:" in content:
        start = content.find("elif daddr_match:")
        snippet = content[start:start+300]
        if "tokens.append('address')" in snippet:
            print("✓ daddr_match 分支输出 'address' token")
        elif "tokens.append('daddr')" in snippet:
            print("⚠️  daddr_match 分支输出 'daddr' token (需要改为'address')")
        else:
            print("? 无法确定daddr_match的输出")

print("\n" + "="*80)
print("结论")
print("="*80)
print("如果当前代码没有 'elif daddr_match:' 分支:")
print("  → Pretrain训练时 daddr(...) 被当成UNK")
print("  → 模型学到的是错误的表示")
print("  → 需要重新训练pretrain模型")
print()
print("修复后:")
print("  → daddr(...) 正确解析为 'address' token")
print("  → 和 address(...) 使用相同的token")
print("  → 位置embedding区分具体地址")
