#!/usr/bin/env python3
import json
import sys

print("="*80)
print("检查Evaluation数据实际格式")
print("="*80)

with open('/data/kun/jtrans/addressaware/eval/func_blocks_addr.json', 'r') as f:
    data = json.load(f)

print(f"总共加载了 {len(data)} 个函数\n")

for idx, (func_id, func_data) in enumerate(data.items()):
    if idx >= 2:
        break
        
    print(f"\n函数 {func_id}:")
    print(f"  字段: {list(func_data.keys())}")
    
    has_inst = 'instructions' in func_data
    has_tok = 'tokens' in func_data
    
    print(f"  Has instructions: {has_inst}")
    print(f"  Has tokens: {has_tok}")
    
    if has_inst:
        inst = func_data['instructions']
        print(f"  instructions 长度: {len(inst)}")
        print(f"  前200字符: {inst[:200]}")
        print(f"  包含tab: {chr(9) in inst}")
        if chr(9) in inst:
            parts = inst.split('\t')
            print(f"  指令数量: {len(parts)}")
            print(f"  第一条指令: {parts[0][:100] if len(parts[0]) > 100 else parts[0]}")
    
    if has_tok:
        tok = func_data['tokens']
        print(f"  tokens 长度: {len(tok)}")
        print(f"  前150字符: {tok[:150]}")
        print(f"  空格分隔的token数: {len(tok.split())}")

print("\n" + "="*80)
print("结论:")
print("="*80)

if has_inst and chr(9) in inst:
    print("✓ 数据包含instructions字段，且是tab分隔的格式")
elif has_inst and chr(9) not in inst:
    print("✗ 数据有instructions字段，但不是tab分隔！可能是空格分隔")
elif not has_inst:
    print("✗ 数据缺少instructions字段！")
