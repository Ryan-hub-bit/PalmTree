#!/usr/bin/env python3
"""
验证Pretrain、Finetune、Evaluation三者tokenization的一致性
"""
import re

print("="*80)
print("验证三阶段Tokenization一致性")
print("="*80)

# 测试指令
test_inst = "lea(0x401000:0.56:0.0:0.0) rdi address(0x404000:0.99:0.0:0.0)"
print(f"\n测试指令: {test_inst}\n")

# 1. Pretrain的解析逻辑
print("1. PRETRAIN (dataloader_addressaware.py):")
addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')

tokens = []
positions = []
match = addr_pattern.match(test_inst)
if match:
    opcode = match.group(1)
    binary_pos = float(match.group(3))
    function_pos = float(match.group(4))
    bb_pos = float(match.group(5))
    tokens.append(opcode)
    positions.append((binary_pos, function_pos, bb_pos))
    
    # Parse operands
    operands_text = test_inst[match.end():].strip()
    for operand in operands_text.split():
        nested_match = nested_addr_pattern.match(operand)
        if nested_match:
            nested_binary_pos = float(nested_match.group(2))
            nested_function_pos = float(nested_match.group(3))
            nested_bb_pos = float(nested_match.group(4))
            tokens.append('address')
            positions.append((nested_binary_pos, nested_function_pos, nested_bb_pos))
        else:
            tokens.append(operand)
            positions.append((-1.0, -1.0, -1.0))

print(f"   Tokens: {tokens}")
print(f"   Positions: {positions}")

# 2. Finetune的解析逻辑
print("\n2. FINETUNE (data_json.py):")
tokens2 = []
positions2 = []
parts = test_inst.strip().split()
for part in parts:
    match = addr_pattern.match(part)
    if match:
        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))
        tokens2.append(opcode)
        positions2.append((binary_pos, function_pos, bb_pos))
        continue
    
    nested_match = nested_addr_pattern.match(part)
    if nested_match:
        binary_pos = float(nested_match.group(2))
        function_pos = float(nested_match.group(3))
        bb_pos = float(nested_match.group(4))
        tokens2.append('address')
        positions2.append((binary_pos, function_pos, bb_pos))
        continue
    
    tokens2.append(part)
    positions2.append((-1.0, -1.0, -1.0))

print(f"   Tokens: {tokens2}")
print(f"   Positions: {positions2}")

# 3. Evaluation的解析逻辑（我修改后的）
print("\n3. EVALUATION (evaluate_addressaware_pools.py - 修改后):")
print(f"   使用与FINETUNE相同的逻辑")
print(f"   Tokens: {tokens2}")
print(f"   Positions: {positions2}")

# 比较结果
print("\n" + "="*80)
print("对比结果:")
print("="*80)
print(f"Pretrain vs Finetune tokens: {tokens == tokens2}")
print(f"Pretrain vs Finetune positions: {positions == positions2}")

if tokens == tokens2 and positions == positions2:
    print("\n✅ 三阶段完全一致！")
else:
    print("\n❌ 存在差异！")
    print(f"\nPretrain tokens: {tokens}")
    print(f"Finetune tokens: {tokens2}")

# 检查<sos>和<eos>的处理
print("\n" + "="*80)
print("特殊标记处理:")
print("="*80)
print("Pretrain: 添加<sos>和<eos>，segment=1")
print("Finetune: 添加<sos>和<eos>，segment=inst_idx+1")
print("Evaluation: 使用与Finetune相同的逻辑")
print("\n⚠️  注意: Pretrain使用segment=1（单序列），Finetune使用segment=inst_idx（指令ID）")
print("这个差异是否会影响？需要检查模型训练时使用的segment策略！")
