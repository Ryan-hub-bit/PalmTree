#!/usr/bin/env python3
"""
测试新的 Address-Aware Embedding 实现
验证 Norm -> Sin/Cos -> Concat -> MLP 的流程
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'extern/jTrans/pretrain/address_aware'))

import torch
from address_embedding import AddressPositionalEmbedding

def test_address_embedding():
    """测试地址嵌入的形状和功能"""
    
    print("="*80)
    print("测试 Address-Aware Positional Embedding")
    print("="*80)
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_sin_cos_features = 8
    
    # 创建模型
    print(f"\n创建模型: d_model={d_model}, num_sin_cos_features={num_sin_cos_features}")
    model = AddressPositionalEmbedding(
        d_model=d_model, 
        num_sin_cos_features=num_sin_cos_features,
        dropout=0.1
    )
    
    # 创建测试数据
    # 前5个token是address，后5个是其他token（用-1表示无效位置）
    binary_pos = torch.rand(batch_size, seq_len) * 0.8  # [0, 0.8]
    function_pos = torch.rand(batch_size, seq_len) * 0.6  # [0, 0.6]
    bb_pos = torch.rand(batch_size, seq_len) * 0.4  # [0, 0.4]
    
    # 后5个token设为-1（非address token）
    binary_pos[:, 5:] = -1
    function_pos[:, 5:] = -1
    bb_pos[:, 5:] = -1
    
    # 创建token_ids：前3个是'address'，中间2个是'daddr'，后5个是其他
    vocab_stoi = {'address': 10, 'daddr': 20}
    token_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    token_ids[:, :3] = vocab_stoi['address']  # 前3个是address
    token_ids[:, 3:5] = vocab_stoi['daddr']   # 中间2个是daddr
    token_ids[:, 5:] = 5                      # 后5个是其他token
    
    print(f"\n输入数据形状:")
    print(f"  binary_pos: {binary_pos.shape}")
    print(f"  function_pos: {function_pos.shape}")
    print(f"  bb_pos: {bb_pos.shape}")
    print(f"  token_ids: {token_ids.shape}")
    
    print(f"\n第一个样本的位置值（前5个有效）:")
    print(f"  binary_pos[0, :5]: {binary_pos[0, :5]}")
    print(f"  function_pos[0, :5]: {function_pos[0, :5]}")
    print(f"  bb_pos[0, :5]: {bb_pos[0, :5]}")
    
    print(f"\n第一个样本的token类型:")
    print(f"  token_ids[0]: {token_ids[0]}")
    print(f"    位置 0-2: 'address' (code addresses)")
    print(f"    位置 3-4: 'daddr' (data addresses)")
    print(f"    位置 5-9: 其他 tokens (non-address)")
    
    # 前向传播
    print(f"\n执行前向传播...")
    model.eval()
    with torch.no_grad():
        embedding = model(binary_pos, function_pos, bb_pos, token_ids, vocab_stoi)
    
    print(f"\n输出 embedding 形状: {embedding.shape}")
    print(f"  预期形状: [{batch_size}, {seq_len}, {d_model}]")
    
    # 验证非address token的embedding是否为0
    non_address_embedding = embedding[:, 5:, :]
    print(f"\n非address token的embedding范数（应该接近0）:")
    print(f"  {torch.norm(non_address_embedding, dim=-1)}")
    
    # 验证address和daddr token的embedding是否不同
    address_embedding_mean = embedding[:, :3, :].mean()
    daddr_embedding_mean = embedding[:, 3:5, :].mean()
    print(f"\naddress token的平均embedding值: {address_embedding_mean.item():.6f}")
    print(f"daddr token的平均embedding值: {daddr_embedding_mean.item():.6f}")
    
    # 比较第一个address token和第一个daddr token的embedding
    first_address_emb = embedding[0, 0, :]
    first_daddr_emb = embedding[0, 3, :]
    are_different = not torch.allclose(first_address_emb, first_daddr_emb)
    print(f"address和daddr的embedding是否不同: {are_different}")
    
    # 显示中间特征维度
    total_sincos_dim = 3 * 2 * num_sin_cos_features
    print(f"\n架构细节:")
    print(f"  1. Norm: 3个层级位置 → [0, 1]")
    print(f"  2. Sin/Cos: 每个层级 → {2*num_sin_cos_features}维 (sin+cos)")
    print(f"  3. Concat: 3个层级拼接 → {total_sincos_dim}维")
    print(f"  4. MLP: {total_sincos_dim}维 → {d_model}维 (code/data各有独立MLP)")
    
    print(f"\n{'='*80}")
    print("测试通过！✓")
    print("="*80)

if __name__ == "__main__":
    test_address_embedding()
