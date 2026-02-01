#!/usr/bin/env python3
"""
总结：Pretrain/Finetune/Evaluation 三阶段对齐情况

数据格式:
- address(0xADDR:norm1:norm2:norm3) → code address (jump targets)
- daddr(0xADDR:norm1:norm2:norm3) → data address (.data, .rodata, .bss)

问题:
旧版本pretrain没有处理daddr()，导致整个字符串变成UNK token
"""

print("="*80)
print("三阶段Address解析策略对比")
print("="*80)

print("\n1. 数据生成阶段 (function_export_ida.py)")
print("-" * 80)
print("  - Code address (在text section):      address(0xADDR:norm1:norm2:norm3)")
print("  - Data address (在.data/.rodata/.bss): daddr(0xADDR:norm1:norm2:norm3)")
print("  ✓ 设计合理：区分code和data引用")

print("\n2. Pretrain阶段 (dataloader_addressaware.py)")
print("-" * 80)
print("  旧版本问题:")
print("    - 定义了 daddr_pattern 但从未使用")
print("    - daddr(...) 被当成普通token → UNK")
print("    - address(...) 正确解析为 'address' token")
print()
print("  修复后:")
print("    - 添加 daddr_pattern.match() 检查")
print("    - daddr(...) → 'address' token (统一处理)")
print("    - address(...) → 'address' token")
print("  ✓ 已修复")

print("\n3. Finetune阶段 (data_json.py)")
print("-" * 80)
print("  修复前:")
print("    - address(...) → 'address' token")
print("    - daddr(...) → 'daddr' token (不一致！)")
print()
print("  修复后:")
print("    - address(...) → 'address' token")
print("    - daddr(...) → 'address' token (对齐pretrain)")
print("  ✓ 已修复")

print("\n4. Evaluation阶段 (evaluate_addressaware_pools.py)")
print("-" * 80)
print("  修复前:")
print("    - 使用 'tokens' 字段（预处理过的，不含daddr）")
print("    - 或者 daddr(...) → 'daddr' token (不一致)")
print()
print("  修复后:")
print("    - 只使用 'instructions' 字段（原始格式）")
print("    - address(...) → 'address' token")
print("    - daddr(...) → 'address' token (对齐pretrain)")
print("  ✓ 已修复")

print("\n" + "="*80)
print("结论")
print("="*80)
print("✓ 三阶段现在完全一致：")
print("  - 都从 'instructions' 字段解析原始格式")
print("  - address(...) 和 daddr(...) 都输出 'address' token")
print("  - 位置信息(norm1:norm2:norm3)保持不变，用于embedding")
print()
print("✓ 设计合理性：")
print("  - 数据生成时区分 address/daddr 提供语义信息")
print("  - Token化时统一为 'address' 避免词表碎片化")
print("  - 位置embedding捕获实际地址差异")
print()
print("⚠ 重要：需要重新训练所有模型，因为修复了tokenization！")
print("  - Pretrain: 之前daddr是UNK，现在是'address'")
print("  - Finetune: 之前用错误的字段或'daddr' token")
print("  - Evaluation: 现在才和训练一致")
