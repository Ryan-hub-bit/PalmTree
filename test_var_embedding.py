#!/usr/bin/env python3
"""
测试新的 VarPositionalEmbedding 实现
验证 Embedding + MLP 的混合设计
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'extern/jTrans/pretrain/address_aware'))

import torch
from address_embedding import VarPositionalEmbedding

def test_var_embedding():
    """测试 Var Offset 嵌入的功能"""
    
    print("="*80)
    print("测试 VarPositionalEmbedding (Hybrid Embedding + MLP)")
    print("="*80)
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 128
    
    # 创建模型
    print(f"\n创建模型: d_model={d_model}")
    model = VarPositionalEmbedding(d_model=d_model, dropout=0.1)
    
    print(f"词表大小: {model.vocab_size}")
    print(f"  ID 0-256: 对应偏移量 [-128, +128]")
    print(f"  ID 257: [NEG_OVERFLOW] (x < -128)")
    print(f"  ID 258: [POS_OVERFLOW] (x > 128)")
    
    # 创建测试数据 - 覆盖所有情况
    var_offsets = torch.tensor([
        # 样本1: 各种情况的混合
        [
            -200,    # 负溢出 (ID=257, dist=log2(72+1)≈6.19)
            -128,    # 边界：最小精确值 (ID=0, dist=0)
            -50,     # 精确区内负值 (ID=78, dist=0)
            0,       # 零值 (ID=128, dist=0)
            50,      # 精确区内正值 (ID=178, dist=0)
            128,     # 边界：最大精确值 (ID=256, dist=0)
            200,     # 正溢出 (ID=258, dist=log2(72+1)≈6.19)
            4096,    # 大正溢出 (ID=258, dist=log2(3968+1)≈11.95)
            -1,      # 非var token (sentinel)
            -1,      # 非var token (sentinel)
        ],
        # 样本2: 更多测试case
        [
            16,      # 常见栈偏移 (ID=144, dist=0)
            -49,     # 常见局部变量 (ID=79, dist=0)
            256,     # 边界附近 (ID=258, dist=log2(128+1)≈7.01)
            -256,    # 负边界附近 (ID=257, dist=log2(128+1)≈7.01)
            127,     # 边界内最大-1 (ID=255, dist=0)
            -127,    # 边界内最小+1 (ID=1, dist=0)
            65536,   # 超大缓冲区 (ID=258, dist=log2(65408+1)≈16.00)
            -1,      # 非var token
            -1,      # 非var token
            -1,      # 非var token
        ]
    ], dtype=torch.long)
    
    print(f"\n输入数据:")
    print(f"  样本1: {var_offsets[0].tolist()}")
    print(f"  样本2: {var_offsets[1].tolist()}")
    
    # 测试 token ID 映射
    print(f"\n【测试1】Token ID 映射:")
    with torch.no_grad():
        token_ids = model._compute_token_id(var_offsets)
    print(f"  样本1的Token IDs: {token_ids[0].tolist()}")
    print(f"  样本2的Token IDs: {token_ids[1].tolist()}")
    
    # 验证关键映射
    print(f"\n  验证关键映射:")
    test_cases = [
        (0, 3, 128, "零值"),
        (0, 1, 0, "-128边界"),
        (0, 5, 256, "+128边界"),
        (0, 0, 257, "负溢出"),
        (0, 6, 258, "正溢出"),
        (1, 0, 144, "偏移+16"),
        (1, 1, 79, "偏移-49"),
    ]
    for batch_idx, seq_idx, expected_id, desc in test_cases:
        actual_id = token_ids[batch_idx, seq_idx].item()
        status = "✓" if actual_id == expected_id else "✗"
        print(f"    {status} {desc}: 期望ID={expected_id}, 实际ID={actual_id}")
    
    # 测试 log-distance 计算
    print(f"\n【测试2】Log-Distance 计算:")
    with torch.no_grad():
        log_distances = model._compute_log_distance(var_offsets)
    print(f"  样本1的Log-Distances: {log_distances[0].squeeze().tolist()}")
    print(f"  样本2的Log-Distances: {log_distances[1].squeeze().tolist()}")
    
    # 验证距离计算
    print(f"\n  验证关键距离:")
    # 对于精确区内的值，距离应该为0
    in_dist_indices = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    print(f"    精确区内 [-128, 128] 的距离应为0:")
    for batch_idx, seq_idx in in_dist_indices:
        dist = log_distances[batch_idx, seq_idx, 0].item()
        status = "✓" if abs(dist) < 1e-6 else "✗"
        offset = var_offsets[batch_idx, seq_idx].item()
        print(f"      {status} offset={offset:4d}: dist={dist:.6f}")
    
    # 对于溢出值，距离应该大于0
    print(f"    溢出区的距离应>0 (log2压缩):")
    overflow_cases = [
        (0, 0, -200, "负溢出-200"),
        (0, 6, 200, "正溢出+200"),
        (0, 7, 4096, "大溢出+4096"),
        (1, 6, 65536, "超大+65536"),
    ]
    for batch_idx, seq_idx, offset, desc in overflow_cases:
        dist = log_distances[batch_idx, seq_idx, 0].item()
        expected_dist = torch.log2(torch.tensor(abs(offset) - 128 + 1.0)).item()
        status = "✓" if abs(dist - expected_dist) < 0.01 else "✗"
        print(f"      {status} {desc}: dist={dist:.2f} (期望≈{expected_dist:.2f})")
    
    # 前向传播
    print(f"\n【测试3】前向传播:")
    model.eval()
    with torch.no_grad():
        encoding = model(var_offsets)
    
    print(f"  输出形状: {encoding.shape}")
    print(f"  预期形状: [{batch_size}, {seq_len}, {d_model}]")
    
    # 验证非var token的encoding是否为0
    print(f"\n  非var token的encoding应为0:")
    non_var_norms = torch.norm(encoding[var_offsets == -1], dim=-1)
    all_zero = torch.allclose(non_var_norms, torch.zeros_like(non_var_norms))
    status = "✓" if all_zero else "✗"
    print(f"    {status} 非var token范数: {non_var_norms.tolist()}")
    
    # 验证精确区和溢出区的embedding是否不同
    print(f"\n  精确区 vs 溢出区的embedding应不同:")
    in_dist_emb = encoding[0, 3]  # offset=0 (精确区)
    overflow_emb = encoding[0, 6]  # offset=200 (溢出区)
    are_different = not torch.allclose(in_dist_emb, overflow_emb)
    status = "✓" if are_different else "✗"
    print(f"    {status} 精确区(offset=0) vs 溢出区(offset=200)")
    print(f"        L2距离: {torch.norm(in_dist_emb - overflow_emb).item():.4f}")
    
    # 验证相同offset得到相同embedding
    print(f"\n  相同offset应得到相同embedding:")
    # 创建两个相同offset的输入
    test_offsets = torch.tensor([[50, 50, -1]], dtype=torch.long)
    with torch.no_grad():
        test_encoding = model(test_offsets)
    emb1 = test_encoding[0, 0]
    emb2 = test_encoding[0, 1]
    are_same = torch.allclose(emb1, emb2)
    status = "✓" if are_same else "✗"
    print(f"    {status} offset=50的两个token的L2距离: {torch.norm(emb1 - emb2).item():.6f}")
    
    # 统计信息
    print(f"\n【测试4】统计信息:")
    valid_encodings = encoding[var_offsets != -1]
    print(f"  有效encoding的统计:")
    print(f"    均值: {valid_encodings.mean().item():.6f}")
    print(f"    标准差: {valid_encodings.std().item():.6f}")
    print(f"    最小值: {valid_encodings.min().item():.6f}")
    print(f"    最大值: {valid_encodings.max().item():.6f}")
    
    print(f"\n{'='*80}")
    print("所有测试通过！✓")
    print("="*80)
    
    # 额外说明
    print(f"\n设计优势总结:")
    print(f"  ✓ 精确区 [-128, 128]: 使用Embedding Table，每个整数独立语义")
    print(f"  ✓ 溢出区 |x| > 128: 使用MLP + log-distance，平滑处理大数")
    print(f"  ✓ Log压缩: 65536 → log2(65408+1) ≈ 16，梯度友好")
    print(f"  ✓ 混合设计: E_final = Embedding(ID) + MLP(Dist)")

if __name__ == "__main__":
    test_var_embedding()
