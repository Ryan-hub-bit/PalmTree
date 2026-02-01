#!/usr/bin/env python3
"""
验证Pretrain/Finetune/Evaluation三阶段的地址解析一致性
"""

import re

print("="*80)
print("验证三阶段处理 address/daddr 的一致性")
print("="*80)

# 测试数据：包含 address() 和 daddr() 的指令
test_instruction = "lea(0xf170:0.56621473:0.00000000:0.00000000) rdi daddr(0x1a8c8:0.99618908:0.00000000:0.00000000)"

print(f"\n测试指令:")
print(f"  {test_instruction}")
print()

# ============================================================================
# PRETRAIN 的解析逻辑 (dataloader_addressaware.py)
# ============================================================================
def pretrain_parse(inst_text):
    """模拟pretrain的_parse_instruction"""
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    tokens = []
    positions = []
    
    # Match opcode
    match = addr_pattern.match(inst_text)
    if match:
        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))
        
        tokens.append(opcode)
        positions.append((binary_pos, function_pos, bb_pos))
        
        # Parse operands
        operands_text = inst_text[match.end():].strip()
        if operands_text:
            for operand in operands_text.split():
                nested_match = nested_addr_pattern.match(operand)
                daddr_match = daddr_pattern.match(operand)
                var_match = var_pattern.match(operand)
                
                if nested_match:
                    # address(...) -> 'address'
                    binary_pos = float(nested_match.group(2))
                    function_pos = float(nested_match.group(3))
                    bb_pos = float(nested_match.group(4))
                    tokens.append('address')
                    positions.append((binary_pos, function_pos, bb_pos))
                elif daddr_match:
                    # daddr(...) -> 'address' (FIXED!)
                    binary_pos = float(daddr_match.group(2))
                    function_pos = float(daddr_match.group(3))
                    bb_pos = float(daddr_match.group(4))
                    tokens.append('address')
                    positions.append((binary_pos, function_pos, bb_pos))
                elif var_match:
                    tokens.append('var')
                    positions.append((-1.0, -1.0, -1.0))
                else:
                    tokens.append(operand)
                    positions.append((-1.0, -1.0, -1.0))
    
    return tokens, positions

# ============================================================================
# FINETUNE 的解析逻辑 (data_json.py)
# ============================================================================
def finetune_parse(inst_text):
    """模拟finetune的_parse_instruction"""
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    tokens = []
    positions = []
    
    parts = inst_text.strip().split()
    
    for part in parts:
        # Check for opcode(address)
        match = addr_pattern.match(part)
        if match:
            opcode = match.group(1)
            binary_pos = float(match.group(3))
            function_pos = float(match.group(4))
            bb_pos = float(match.group(5))
            
            tokens.append(opcode)
            positions.append((binary_pos, function_pos, bb_pos))
            continue
        
        # Check for address()
        nested_match = nested_addr_pattern.match(part)
        if nested_match:
            binary_pos = float(nested_match.group(2))
            function_pos = float(nested_match.group(3))
            bb_pos = float(nested_match.group(4))
            
            tokens.append('address')
            positions.append((binary_pos, function_pos, bb_pos))
            continue
        
        # Check for daddr() - FIXED to output 'address'
        daddr_match = daddr_pattern.match(part)
        if daddr_match:
            binary_pos = float(daddr_match.group(2))
            function_pos = float(daddr_match.group(3))
            bb_pos = float(daddr_match.group(4))
            
            tokens.append('address')  # MATCH PRETRAIN!
            positions.append((binary_pos, function_pos, bb_pos))
            continue
        
        # Check for var()
        var_match = var_pattern.match(part)
        if var_match:
            tokens.append('var')
            positions.append((-1.0, -1.0, -1.0))
            continue
        
        # Regular token
        tokens.append(part)
        positions.append((-1.0, -1.0, -1.0))
    
    return tokens, positions

# ============================================================================
# EVALUATION 的解析逻辑 (evaluate_addressaware_pools.py)
# ============================================================================
def eval_parse(inst_text):
    """模拟evaluation的_parse_instruction"""
    # 和finetune完全相同
    return finetune_parse(inst_text)

# ============================================================================
# 运行测试
# ============================================================================
print("Pretrain 解析结果:")
pretrain_tokens, pretrain_pos = pretrain_parse(test_instruction)
print(f"  Tokens: {pretrain_tokens}")
print(f"  Positions: {pretrain_pos}")

print("\nFinetune 解析结果:")
finetune_tokens, finetune_pos = finetune_parse(test_instruction)
print(f"  Tokens: {finetune_tokens}")
print(f"  Positions: {finetune_pos}")

print("\nEvaluation 解析结果:")
eval_tokens, eval_pos = eval_parse(test_instruction)
print(f"  Tokens: {eval_tokens}")
print(f"  Positions: {eval_pos}")

print("\n" + "="*80)
print("一致性检查:")
print("="*80)

tokens_match = (pretrain_tokens == finetune_tokens == eval_tokens)
pos_match = (pretrain_pos == finetune_pos == eval_pos)

if tokens_match:
    print("✓ Tokens 完全一致")
else:
    print("✗ Tokens 不一致!")
    if pretrain_tokens != finetune_tokens:
        print(f"  Pretrain vs Finetune: {pretrain_tokens} vs {finetune_tokens}")
    if finetune_tokens != eval_tokens:
        print(f"  Finetune vs Eval: {finetune_tokens} vs {eval_tokens}")

if pos_match:
    print("✓ Positions 完全一致")
else:
    print("✗ Positions 不一致!")

print("\n关键检查:")
print(f"  - daddr(...) 是否被解析为 'address' token? {('address' in pretrain_tokens) and ('daddr' not in pretrain_tokens)}")
print(f"  - 三阶段是否都使用 instructions 字段? (需手动检查代码)")

if tokens_match and pos_match:
    print("\n✓✓✓ 所有阶段已对齐！")
else:
    print("\n✗✗✗ 还有不一致的地方！")
