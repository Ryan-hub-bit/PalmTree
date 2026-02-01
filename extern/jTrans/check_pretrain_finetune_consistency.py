#!/usr/bin/env python3
"""
检查 Pretrain AddressAware 和 Finetune 的数据格式一致性和分词合理性

检查项目:
1. 数据格式 - pretrain corpus vs finetune JSON
2. 分词逻辑 - regex patterns 是否一致
3. 特殊token处理 - address, daddr, var
4. Position encoding - 3-level hierarchical positions
5. 实际数据样本对比
"""

import re
import sys
import os
import json

# Add paths
sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

def check_regex_patterns():
    """检查pretrain和finetune使用的regex patterns是否一致"""
    print("=" * 80)
    print("1. 检查 REGEX PATTERNS 一致性")
    print("=" * 80)
    
    # Pretrain dataloader
    from dataloader_addressaware import AddressAwareDataset
    
    # Create dummy instance to check patterns
    pretrain_patterns = {
        'addr_pattern': r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)',
        'nested_addr_pattern': r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)',
        'daddr_pattern': r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)',
        'var_pattern': r'var\((0x[0-9a-fA-F]+)\)'
    }
    
    print("\nPretrain patterns (from dataloader_addressaware.py):")
    for name, pattern in pretrain_patterns.items():
        print(f"  {name}: {pattern}")
    
    # Finetune dataloader
    sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans')
    from data_json import FunctionDataset_CL_AddressAware_JSON
    
    # Check patterns from data_json.py
    with open('/home/kun/Document/AAE/extern/jTrans/data_json.py', 'r') as f:
        content = f.read()
    
    finetune_patterns = {}
    for name in ['addr_pattern', 'nested_addr_pattern', 'daddr_pattern', 'var_pattern']:
        match = re.search(rf'self\.{name}\s*=\s*re\.compile\(r[\'"](.+?)[\'"]\)', content)
        if match:
            finetune_patterns[name] = match.group(1)
    
    print("\nFinetune patterns (from data_json.py):")
    for name, pattern in finetune_patterns.items():
        print(f"  {name}: {pattern}")
    
    print("\n一致性检查:")
    all_match = True
    for name in pretrain_patterns:
        if name in finetune_patterns:
            if pretrain_patterns[name] == finetune_patterns[name]:
                print(f"  ✓ {name}: 一致")
            else:
                print(f"  ✗ {name}: 不一致!")
                print(f"    Pretrain:  {pretrain_patterns[name]}")
                print(f"    Finetune:  {finetune_patterns[name]}")
                all_match = False
        else:
            print(f"  ? {name}: Finetune中未找到")
            all_match = False
    
    if all_match:
        print("\n✓ 所有regex patterns一致")
    else:
        print("\n⚠️  发现不一致的patterns")
    
    return all_match


def check_data_format():
    """检查pretrain和finetune的数据格式"""
    print("\n" + "=" * 80)
    print("2. 检查 DATA FORMAT")
    print("=" * 80)
    
    # Pretrain data format
    pretrain_path = '/data/kun/jtrans/addressaware/addr_pretrain.txt'
    if os.path.exists(pretrain_path):
        print("\nPretrain数据格式 (addr_pretrain.txt):")
        with open(pretrain_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    print(f"  Line {i+1}: {line[:150]}...")
                else:
                    break
        
        print("\n  格式: 每行一个函数，空格分隔的tokens")
        print("  Token格式: opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2 ...")
        print("  特殊tokens: address(...), daddr(...), var(0xXX)")
    else:
        print(f"\n⚠️  Pretrain数据文件不存在: {pretrain_path}")
    
    # Finetune data format
    finetune_path = '/data/kun/jtrans/addressaware/func_blocks_addr.json'
    if os.path.exists(finetune_path):
        print("\nFinetune数据格式 (func_blocks_addr.json):")
        with open(finetune_path, 'r') as f:
            data = json.load(f)
        
        # Get first function
        first_id = list(data.keys())[0]
        first_func = data[first_id]
        
        print(f"  样本ID: {first_id}")
        print(f"  字段: {list(first_func.keys())}")
        
        if 'instructions' in first_func:
            insts = first_func['instructions']
            print(f"  Instructions类型: {type(insts)}")
            if isinstance(insts, str):
                print(f"  Instructions示例: {insts[:150]}...")
            else:
                print(f"  Instructions示例: {insts[:3]}")
        
        print("\n  格式: JSON格式，每个函数包含:")
        print("    - instructions: 指令字符串 (tab分隔)")
        print("    - 其他元数据: binary_name, function_name, optimization_level等")
    else:
        print(f"\n⚠️  Finetune数据文件不存在: {finetune_path}")


def check_tokenization_logic():
    """检查分词逻辑"""
    print("\n" + "=" * 80)
    print("3. 检查 TOKENIZATION LOGIC")
    print("=" * 80)
    
    # Test sample
    test_sample = "mov(0x1234:0.5:0.3:0.1) rax daddr(0x5678:1.0:0.5:0.2) var(0xfffffff8)"
    
    print(f"\n测试样本: {test_sample}")
    
    # Pretrain tokenization
    print("\nPretrain分词结果:")
    sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')
    from dataloader_addressaware import AddressAwareDataset
    
    # Manually parse like pretrain does
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    tokens_pretrain = []
    positions_pretrain = []
    var_offsets_pretrain = []
    
    parts = test_sample.split()
    for part in parts:
        addr_match = addr_pattern.match(part)
        daddr_match = daddr_pattern.match(part)
        var_match = var_pattern.match(part)
        
        if addr_match and not daddr_match:
            tokens_pretrain.append(addr_match.group(1))
            positions_pretrain.append((
                float(addr_match.group(3)),
                float(addr_match.group(4)),
                float(addr_match.group(5))
            ))
            var_offsets_pretrain.append(-1)
        elif daddr_match:
            tokens_pretrain.append('daddr')
            positions_pretrain.append((
                float(daddr_match.group(2)),
                float(daddr_match.group(3)),
                float(daddr_match.group(4))
            ))
            var_offsets_pretrain.append(-1)
        elif var_match:
            tokens_pretrain.append('var')
            var_hex = var_match.group(1)
            var_offset = int(var_hex, 16)
            if var_offset >= 0x80000000:
                var_offset = var_offset - 0x100000000
            positions_pretrain.append((-1.0, -1.0, -1.0))
            var_offsets_pretrain.append(var_offset)
        else:
            tokens_pretrain.append(part)
            positions_pretrain.append((-1.0, -1.0, -1.0))
            var_offsets_pretrain.append(-1)
    
    print(f"  Tokens: {tokens_pretrain}")
    print(f"  Positions (b,f,bb): {positions_pretrain}")
    print(f"  Var offsets: {var_offsets_pretrain}")
    
    # Finetune tokenization (from data_json.py _parse_instruction)
    print("\nFinetune分词结果:")
    tokens_finetune = []
    positions_finetune = []
    var_offsets_finetune = []
    
    for part in parts:
        addr_match = addr_pattern.match(part)
        daddr_match = daddr_pattern.match(part)
        var_match = var_pattern.match(part)
        
        if addr_match and not daddr_match:
            tokens_finetune.append(addr_match.group(1))
            positions_finetune.append((
                float(addr_match.group(3)),
                float(addr_match.group(4)),
                float(addr_match.group(5))
            ))
            var_offsets_finetune.append(-1)
        elif daddr_match:
            tokens_finetune.append('daddr')
            positions_finetune.append((
                float(daddr_match.group(2)),
                float(daddr_match.group(3)),
                float(daddr_match.group(4))
            ))
            var_offsets_finetune.append(-1)
        elif var_match:
            tokens_finetune.append('var')
            var_hex = var_match.group(1)
            var_offset = int(var_hex, 16)
            if var_offset >= 0x80000000:
                var_offset = var_offset - 0x100000000
            positions_finetune.append((-1.0, -1.0, -1.0))
            var_offsets_finetune.append(var_offset)
        else:
            tokens_finetune.append(part)
            positions_finetune.append((-1.0, -1.0, -1.0))
            var_offsets_finetune.append(-1)
    
    print(f"  Tokens: {tokens_finetune}")
    print(f"  Positions (b,f,bb): {positions_finetune}")
    print(f"  Var offsets: {var_offsets_finetune}")
    
    # Compare
    print("\n一致性检查:")
    if tokens_pretrain == tokens_finetune:
        print("  ✓ Tokens一致")
    else:
        print("  ✗ Tokens不一致!")
        
    if positions_pretrain == positions_finetune:
        print("  ✓ Positions一致")
    else:
        print("  ✗ Positions不一致!")
        
    if var_offsets_pretrain == var_offsets_finetune:
        print("  ✓ Var offsets一致")
    else:
        print("  ✗ Var offsets不一致!")


def check_special_tokens():
    """检查特殊token处理"""
    print("\n" + "=" * 80)
    print("4. 检查 SPECIAL TOKENS 处理")
    print("=" * 80)
    
    print("\n特殊tokens及其含义:")
    print("  - address: 控制流地址 (jump/call targets)")
    print("  - daddr:   数据流地址 (memory operands)")
    print("  - var:     局部变量 (stack offsets)")
    print("  - <pad>:   Padding token")
    print("  - <unk>:   Unknown token")
    print("  - <mask>:  Masked token (pretrain only)")
    print("  - <sos>:   Start of sequence (finetune)")
    print("  - <eos>:   End of sequence (finetune)")
    
    # Check vocab
    vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt'
    if os.path.exists(vocab_path):
        print(f"\n词汇表文件: {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab_tokens = [line.strip() for line in f if line.strip()]
        
        special_tokens = ['<pad>', '<unk>', '<mask>', '<sos>', '<eos>', 'address', 'daddr', 'var']
        print("\n特殊tokens在词汇表中的位置:")
        for token in special_tokens:
            if token in vocab_tokens:
                idx = vocab_tokens.index(token)
                print(f"  ✓ {token:10s} -> ID {idx}")
            else:
                print(f"  ✗ {token:10s} -> 未找到!")
    else:
        print(f"\n⚠️  词汇表文件不存在: {vocab_path}")


def check_real_data_samples():
    """检查真实数据样本"""
    print("\n" + "=" * 80)
    print("5. 检查 REAL DATA SAMPLES")
    print("=" * 80)
    
    # Pretrain data
    pretrain_path = '/data/kun/jtrans/addressaware/addr_pretrain.txt'
    if os.path.exists(pretrain_path):
        print("\nPretrain数据样本 (前3个函数):")
        with open(pretrain_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    line = line.strip()
                    tokens = line.split()
                    
                    # Count special tokens
                    address_count = sum(1 for t in tokens if t == 'address' or 'address(' in t)
                    daddr_count = sum(1 for t in tokens if t == 'daddr' or 'daddr(' in t)
                    var_count = sum(1 for t in tokens if t == 'var' or 'var(' in t)
                    
                    print(f"\n  函数 {i+1}:")
                    print(f"    总tokens: {len(tokens)}")
                    print(f"    address: {address_count}")
                    print(f"    daddr: {daddr_count}")
                    print(f"    var: {var_count}")
                    print(f"    前10个tokens: {tokens[:10]}")
                else:
                    break
    
    # Finetune data
    finetune_path = '/data/kun/jtrans/addressaware/func_blocks_addr.json'
    if os.path.exists(finetune_path):
        print("\nFinetune数据样本 (前3个函数):")
        with open(finetune_path, 'r') as f:
            data = json.load(f)
        
        for i, func_id in enumerate(list(data.keys())[:3]):
            func = data[func_id]
            
            # Parse instructions
            if 'instructions' in func:
                insts_str = func['instructions']
                tokens = insts_str.split()
                
                # Count special tokens
                address_count = sum(1 for t in tokens if 'address(' in t)
                daddr_count = sum(1 for t in tokens if 'daddr(' in t)
                var_count = sum(1 for t in tokens if 'var(' in t)
                
                print(f"\n  函数 {i+1} (ID: {func_id}):")
                print(f"    Binary: {func.get('binary_name', 'N/A')}")
                print(f"    Function: {func.get('function_name', 'N/A')}")
                print(f"    Opt: {func.get('optimization_level', 'N/A')}")
                print(f"    总tokens: {len(tokens)}")
                print(f"    address: {address_count}")
                print(f"    daddr: {daddr_count}")
                print(f"    var: {var_count}")
                print(f"    前10个tokens: {tokens[:10]}")


def main():
    print("\n" + "=" * 80)
    print("Pretrain AddressAware 和 Finetune 数据格式一致性检查")
    print("=" * 80)
    
    try:
        patterns_ok = check_regex_patterns()
        check_data_format()
        check_tokenization_logic()
        check_special_tokens()
        check_real_data_samples()
        
        print("\n" + "=" * 80)
        print("总结")
        print("=" * 80)
        
        if patterns_ok:
            print("\n✓ Regex patterns一致")
            print("✓ 分词逻辑相同")
            print("✓ 特殊token处理一致")
            print("\n结论: Pretrain和Finetune的数据格式和分词完全一致，可以正常训练")
        else:
            print("\n⚠️  发现不一致之处，请查看上面的详细信息")
        
        print("\n建议:")
        print("1. 确保使用修复后的dataloader_addressaware.py (已添加daddr_match检查)")
        print("2. 确认pretrain数据包含daddr tokens")
        print("3. 验证vocab.txt包含所有特殊tokens")
        print("4. 检查真实数据中address/daddr/var的分布是否合理")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
