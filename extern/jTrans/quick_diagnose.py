#!/usr/bin/env python3
"""
快速诊断 Recall@1 低的原因
"""

import torch
import numpy as np
import json
import sys
import os

def check_evaluation_code():
    """检查评估代码的相似度计算"""
    print("=" * 80)
    print("1. 检查评估代码")
    print("=" * 80)
    
    eval_file = '/home/kun/Document/AAE/extern/jTrans/evaluate_addressaware_pools.py'
    
    with open(eval_file, 'r') as f:
        content = f.read()
    
    # Check similarity computation
    if 'np.linalg.norm' in content and 'pool_norm =' in content:
        print("✓ 评估使用了L2 normalization (cosine similarity)")
    else:
        print("⚠️  评估可能没有使用正确的cosine similarity!")
    
    # Check for common bugs
    if 'np.dot' in content or 'np.matmul' in content:
        print("✓ 使用dot product计算相似度")
    else:
        print("⚠️  未找到dot product计算")
    
    # Check ground truth usage
    if 'ground_truth[i]' in content or 'gt_idx' in content:
        print("✓ 正确使用ground truth索引")
    else:
        print("⚠️  Ground truth使用可能有问题")
    
    print("\n查看compute_similarity函数:")
    lines = content.split('\n')
    in_func = False
    func_lines = []
    for line in lines:
        if 'def compute_similarity' in line:
            in_func = True
        if in_func:
            func_lines.append(line)
            if len(func_lines) > 15:
                break
    
    for line in func_lines:
        print(f"  {line}")


def check_finetune_training():
    """检查finetune训练日志"""
    print("\n" + "=" * 80)
    print("2. 检查Finetune训练状态")
    print("=" * 80)
    
    checkpoint_dir = '/home/kun/Document/AAE/output/jtrans/addressaware_finetune_sincos'
    
    if not os.path.exists(checkpoint_dir):
        print(f"⚠️  Checkpoint目录不存在: {checkpoint_dir}")
        return
    
    # List available epochs
    epochs = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith('finetune_epoch_'):
            epoch_num = int(item.split('_')[-1])
            epochs.append(epoch_num)
    
    if epochs:
        epochs.sort()
        print(f"✓ 找到{len(epochs)}个epoch checkpoints: {epochs}")
        print(f"  最新epoch: {max(epochs)}")
    else:
        print("⚠️  未找到任何epoch checkpoints!")
    
    # Check for log files
    log_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.log')]
    if log_files:
        print(f"\n✓ 找到训练日志: {log_files}")
        
        # Read last few lines
        with open(os.path.join(checkpoint_dir, log_files[0]), 'r') as f:
            lines = f.readlines()
        
        print("\n最后10行日志:")
        for line in lines[-10:]:
            print(f"  {line.strip()}")
    else:
        print("\n⚠️  未找到训练日志文件")


def check_pool_evaluation_setup():
    """检查pool evaluation的设置"""
    print("\n" + "=" * 80)
    print("3. 检查Pool Evaluation设置")
    print("=" * 80)
    
    pool_dir = '/data/kun/jtrans/addressaware/eval/pools'
    
    if not os.path.exists(pool_dir):
        print(f"⚠️  Pool目录不存在: {pool_dir}")
        return
    
    # List pools
    pools = [f for f in os.listdir(pool_dir) if f.startswith('pool_') and f.endswith('.json')]
    queries = [f for f in os.listdir(pool_dir) if f.startswith('query_') and f.endswith('.json')]
    
    print(f"✓ 找到{len(pools)}个pool files")
    print(f"✓ 找到{len(queries)}个query files")
    
    if pools:
        # Check one pool file
        sample_pool = os.path.join(pool_dir, pools[0])
        with open(sample_pool, 'r') as f:
            pool_data = json.load(f)
        
        print(f"\n示例pool: {pools[0]}")
        print(f"  Pool大小: {len(pool_data.get('pool', []))}")
        print(f"  Keys: {list(pool_data.keys())}")
    
    if queries:
        # Check one query file
        sample_query = os.path.join(pool_dir, queries[0])
        with open(sample_query, 'r') as f:
            query_data = json.load(f)
        
        print(f"\n示例query: {queries[0]}")
        print(f"  Query大小: {len(query_data.get('queries', []))}")
        print(f"  Ground truth大小: {len(query_data.get('ground_truth', []))}")
        print(f"  Keys: {list(query_data.keys())}")
        
        # Check if ground truth indices are valid
        gt = query_data.get('ground_truth', [])
        if gt:
            pool_size = len(pool_data.get('pool', []))
            print(f"\n  Ground truth检查:")
            print(f"    最小索引: {min(gt)}")
            print(f"    最大索引: {max(gt)}")
            print(f"    Pool大小: {pool_size}")
            
            if max(gt) >= pool_size:
                print(f"    ⚠️  Ground truth索引超出pool范围!")
            else:
                print(f"    ✓ Ground truth索引在有效范围内")


def check_model_output_distribution():
    """检查模型输出的简单分布"""
    print("\n" + "=" * 80)
    print("4. 模型Embedding分布简单检查")
    print("=" * 80)
    
    # Check if we can load a checkpoint
    checkpoint_path = '/home/kun/Document/AAE/output/jtrans/addressaware_finetune_sincos/finetune_epoch_15/pytorch_model.bin'
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint不存在: {checkpoint_path}")
        return
    
    print(f"✓ 找到checkpoint: {checkpoint_path}")
    
    # Load state dict
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\n模型参数统计:")
        print(f"  总参数数: {len(state_dict)}")
        
        # Check some key layers
        key_layers = [
            'embeddings.token.weight',
            'embeddings.code_address_projection.0.weight',
            'embeddings.data_address_projection.0.weight',
            'encoder.layer.0.attention.self.query.weight',
            'encoder.layer.11.output.dense.weight'
        ]
        
        print(f"\n关键层参数统计:")
        for key in key_layers:
            if key in state_dict:
                weight = state_dict[key]
                print(f"  {key}:")
                print(f"    形状: {weight.shape}")
                print(f"    均值: {weight.mean().item():.6f}")
                print(f"    标准差: {weight.std().item():.6f}")
                print(f"    范数: {weight.norm().item():.4f}")
                
                if weight.norm().item() < 0.1:
                    print(f"    ⚠️  参数接近0，可能未训练!")
        
        # Check if data_address_projection is trained
        if 'embeddings.data_address_projection.0.weight' in state_dict:
            data_weight = state_dict['embeddings.data_address_projection.0.weight']
            code_weight = state_dict.get('embeddings.code_address_projection.0.weight')
            
            if code_weight is not None:
                data_norm = data_weight.norm().item()
                code_norm = code_weight.norm().item()
                
                print(f"\n  Dual MLP检查:")
                print(f"    Code address MLP norm: {code_norm:.4f}")
                print(f"    Data address MLP norm: {data_norm:.4f}")
                print(f"    比值 (data/code): {data_norm/code_norm:.4f}")
                
                if data_norm / code_norm < 0.1:
                    print(f"    ⚠️  Data address MLP可能未被训练!")
                elif abs(data_norm - code_norm) / code_norm < 0.1:
                    print(f"    ⚠️  两个MLP参数太相似，可能共享权重或都未训练")
                else:
                    print(f"    ✓ Dual MLP都已训练")
    
    except Exception as e:
        print(f"⚠️  加载checkpoint出错: {e}")


def check_data_statistics():
    """检查数据统计"""
    print("\n" + "=" * 80)
    print("5. 数据统计检查")
    print("=" * 80)
    
    func_blocks_path = '/data/kun/jtrans/addressaware/func_blocks_addr.json'
    gt_path = '/data/kun/jtrans/addressaware/ground_truth_addr.json'
    
    if os.path.exists(func_blocks_path):
        print(f"✓ Function blocks: {func_blocks_path}")
        with open(func_blocks_path, 'r') as f:
            func_blocks = json.load(f)
        print(f"  总函数数: {len(func_blocks)}")
        
        # Sample a function
        sample_id = list(func_blocks.keys())[0]
        sample = func_blocks[sample_id]
        print(f"\n  示例函数 (ID={sample_id}):")
        print(f"    Binary: {sample.get('binary_name', 'N/A')}")
        print(f"    Function: {sample.get('function_name', 'N/A')}")
        print(f"    Opt: {sample.get('optimization_level', 'N/A')}")
        
        if 'instructions' in sample:
            insts = sample['instructions']
            tokens = insts.split()
            print(f"    Tokens: {len(tokens)}")
            
            # Count special tokens
            address_count = sum(1 for t in tokens if 'address(' in t)
            daddr_count = sum(1 for t in tokens if 'daddr(' in t)
            var_count = sum(1 for t in tokens if 'var(' in t)
            
            print(f"    address: {address_count}")
            print(f"    daddr: {daddr_count}")
            print(f"    var: {var_count}")
    
    if os.path.exists(gt_path):
        print(f"\n✓ Ground truth: {gt_path}")
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        
        pairs = gt_data.get('pairs', [])
        print(f"  总pairs数: {len(pairs)}")
        
        if pairs:
            # Count opt combinations
            opt_combos = {}
            for pair in pairs[:1000]:  # Sample first 1000
                opt1 = pair.get('opt1', 'N/A')
                opt2 = pair.get('opt2', 'N/A')
                key = f"{opt1}-{opt2}"
                opt_combos[key] = opt_combos.get(key, 0) + 1
            
            print(f"\n  Opt组合 (前1000 pairs):")
            for combo, count in sorted(opt_combos.items(), key=lambda x: -x[1]):
                print(f"    {combo}: {count}")


def main():
    print("\n" + "=" * 80)
    print("AddressAware Recall@1 低原因快速诊断")
    print("=" * 80)
    
    try:
        check_evaluation_code()
        check_finetune_training()
        check_pool_evaluation_setup()
        check_model_output_distribution()
        check_data_statistics()
        
        print("\n" + "=" * 80)
        print("诊断总结")
        print("=" * 80)
        print("""
常见导致Recall@1低的原因:

1. ⚠️  Data address MLP未训练
   - Pretrain时daddr被忽略 → data_address_projection未学习
   - 解决: 使用修复后的dataloader重新pretrain

2. ⚠️  相似度计算错误
   - 未使用L2 normalization
   - 解决: 确保使用cosine similarity

3. ⚠️  Ground truth索引错误
   - Pool和query不匹配
   - 解决: 重新生成evaluation pools

4. ⚠️  模型未收敛
   - 训练epochs不足
   - Learning rate过高/过低
   - 解决: 增加训练epochs或调整超参数

5. ⚠️  数据问题
   - Finetune数据质量差
   - Train/eval数据分布不匹配
   - 解决: 检查数据生成pipeline

运行完整诊断:
python diagnose_low_recall.py --checkpoint <path> --data <path>
        """)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
