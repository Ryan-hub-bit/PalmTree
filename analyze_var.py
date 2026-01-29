#!/usr/bin/env python3
"""
分析 addr_pretrain.txt 文件中 var() 括号内数值的统计信息
"""

import re
from pathlib import Path

def analyze_var_values(file_path):
    """分析 var() 中的数值"""
    var_pattern = re.compile(r'var\((0x[0-9A-Fa-f]+)\)')
    
    values = []
    
    print(f"正在读取文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            matches = var_pattern.findall(line)
            for match in matches:
                # 将十六进制转换为有符号整数
                hex_val = int(match, 16)
                # 假设是64位有符号整数
                if hex_val > 0x7FFFFFFFFFFFFFFF:
                    signed_val = hex_val - 0x10000000000000000
                else:
                    signed_val = hex_val
                values.append(signed_val)
    
    if not values:
        print("未找到任何 var() 数值")
        return
    
    total_count = len(values)
    min_val = min(values)
    max_val = max(values)
    
    # 统计在 [-128, 128] 范围内的数量
    count_128 = sum(1 for v in values if -128 <= v <= 128)
    percent_128 = (count_128 / total_count) * 100
    
    # 统计在 [-256, 256] 范围内的数量
    count_256 = sum(1 for v in values if -256 <= v <= 256)
    percent_256 = (count_256 / total_count) * 100
    
    # 统计在 [-512, 512] 范围内的数量
    count_512 = sum(1 for v in values if -512 <= v <= 512)
    percent_512 = (count_512 / total_count) * 100
    
    # 打印统计结果
    print("\n" + "="*60)
    print("var() 数值统计分析")
    print("="*60)
    print(f"总数量: {total_count:,}")
    print(f"最小值: {min_val}")
    print(f"最大值: {max_val}")
    print()
    print(f"[-128, 128] 范围内:")
    print(f"  数量: {count_128:,}")
    print(f"  占比: {percent_128:.2f}%")
    print()
    print(f"[-256, 256] 范围内:")
    print(f"  数量: {count_256:,}")
    print(f"  占比: {percent_256:.2f}%")
    print()
    print(f"[-512, 512] 范围内:")
    print(f"  数量: {count_512:,}")
    print(f"  占比: {percent_512:.2f}%")
    print("="*60)
    
    # 额外统计信息
    print("\n额外统计信息:")
    print(f"负数数量: {sum(1 for v in values if v < 0):,} ({sum(1 for v in values if v < 0)/total_count*100:.2f}%)")
    print(f"零值数量: {sum(1 for v in values if v == 0):,} ({sum(1 for v in values if v == 0)/total_count*100:.2f}%)")
    print(f"正数数量: {sum(1 for v in values if v > 0):,} ({sum(1 for v in values if v > 0)/total_count*100:.2f}%)")

if __name__ == "__main__":
    file_path = "/data/kun/jtrans/addressaware/addr_pretrain.txt"
    
    if not Path(file_path).exists():
        print(f"错误: 文件不存在 - {file_path}")
        exit(1)
    
    analyze_var_values(file_path)
