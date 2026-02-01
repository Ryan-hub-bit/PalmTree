#!/usr/bin/env python3
"""
检查baseline模型是否使用了不规范的方法导致效果好

重点检查：
1. Negative sampling策略 - 是否使用了hard negative mining
2. 数据泄漏 - evaluation时是否使用了训练集中的函数
3. Position embedding技巧 - position_embeddings = word_embeddings
4. 评估时的相似度计算 - 是否正确使用cosine similarity
"""

import sys
import os
import json
import re

def check_negative_sampling():
    """检查negative sampling策略"""
    print("=" * 80)
    print("1. 检查 NEGATIVE SAMPLING 策略")
    print("=" * 80)
    
    with open('data_json.py', 'r') as f:
        content = f.read()
    
    # 查找__getitem__方法中的negative sampling部分
    lines = content.split('\n')
    in_getitem = False
    negative_lines = []
    
    for i, line in enumerate(lines):
        if 'def __getitem__' in line and 'FunctionDataset_CL_Load_JSON' in '\n'.join(lines[max(0, i-20):i]):
            in_getitem = True
        
        if in_getitem:
            negative_lines.append(f"Line {i+1}: {line}")
            if 'return' in line and '(' in line:
                break
    
    print("\nBaseline的Negative Sampling代码:")
    print("-" * 80)
    for line in negative_lines[-30:]:
        if 'neg' in line.lower() or 'anchor' in line.lower():
            print(line)
    
    # 分析策略
    content_lower = content.lower()
    if 'same opt as anchor' in content_lower or 'same.*opt.*anchor' in content_lower:
        print("\n⚠️  发现: Baseline使用HARD NEGATIVE MINING!")
        print("   策略: negative从不同函数但SAME optimization level采样")
        print("   影响: 这是合理的策略，增加训练难度，但也可能导致模型过度关注optimization-specific特征")
    else:
        print("\n✓  Baseline使用随机negative sampling (没有hard negative mining)")
    

def check_data_leakage():
    """检查评估数据是否与训练数据重叠"""
    print("\n" + "=" * 80)
    print("2. 检查 DATA LEAKAGE")
    print("=" * 80)
    
    # 检查训练和评估使用的数据文件
    print("\n训练数据路径 (baseline):")
    with open('finetune.py', 'r') as f:
        finetune_content = f.read()
    
    # 查找baseline相关的数据路径
    train_paths = re.findall(r'func_blocks.*baseline.*\.json', finetune_content, re.IGNORECASE)
    print(f"  {train_paths if train_paths else '未在代码中硬编码，由命令行参数指定'}")
    
    print("\n评估数据路径 (baseline):")
    with open('evaluate_baseline_pools.py', 'r') as f:
        eval_content = f.read()
    
    eval_paths = re.findall(r'default=.*func_blocks.*baseline.*\.json', eval_content, re.IGNORECASE)
    if eval_paths:
        print(f"  {eval_paths}")
    else:
        eval_paths = re.findall(r'/data/kun/jtrans/baseline/.*\.json', eval_content)
        print(f"  {eval_paths if eval_paths else '未找到默认路径'}")
    
    print("\n需要检查:")
    print("  - 训练集: 应该使用 func_blocks_baseline.json + ground_truth_baseline.json")
    print("  - 评估集: 应该使用独立的 eval/func_blocks_baseline.json 和 pools/")
    print("  - 如果训练和评估使用同一个文件 → 数据泄漏!")


def check_position_embedding_trick():
    """检查position embedding技巧"""
    print("\n" + "=" * 80)
    print("3. 检查 POSITION EMBEDDING 技巧")
    print("=" * 80)
    
    # 检查finetune.py
    with open('finetune.py', 'r') as f:
        finetune_lines = f.readlines()
    
    print("\nBaseline模型定义 (finetune.py):")
    print("-" * 80)
    for i, line in enumerate(finetune_lines):
        if 'class BinBertModel' in line:
            # 打印class定义及其后30行
            for j in range(i, min(i+35, len(finetune_lines))):
                print(f"Line {j+1}: {finetune_lines[j]}", end='')
            break
    
    # 检查evaluate_baseline_pools.py
    with open('evaluate_baseline_pools.py', 'r') as f:
        eval_lines = f.readlines()
    
    print("\n\nEvaluation模型定义 (evaluate_baseline_pools.py):")
    print("-" * 80)
    for i, line in enumerate(eval_lines):
        if 'class BinBertModel' in line:
            # 打印class定义及其后30行
            for j in range(i, min(i+35, len(eval_lines))):
                print(f"Line {j+1}: {eval_lines[j]}", end='')
            break
    
    print("\n\n分析:")
    print("✓  position_embeddings = word_embeddings 是jTrans的核心技巧")
    print("   这使得模型可以用token ID作为position ID")
    print("   这是合理的设计，不是作弊")


def check_similarity_computation():
    """检查相似度计算"""
    print("\n" + "=" * 80)
    print("4. 检查 SIMILARITY COMPUTATION")
    print("=" * 80)
    
    with open('evaluate_baseline_pools.py', 'r') as f:
        eval_lines = f.readlines()
    
    print("\nBaseline评估的相似度计算:")
    print("-" * 80)
    for i, line in enumerate(eval_lines):
        if 'def compute_similarity' in line:
            # 打印函数定义及其后20行
            for j in range(i, min(i+20, len(eval_lines))):
                print(f"Line {j+1}: {eval_lines[j]}", end='')
            break
    
    # 检查是否有L2 normalization
    with open('evaluate_baseline_pools.py', 'r') as f:
        content = f.read()
    
    if 'np.linalg.norm' in content and 'pool_norm =' in content:
        print("\n✓  使用了L2 normalization (正确的cosine similarity)")
    else:
        print("\n⚠️  可能没有使用L2 normalization!")
    
    # 对比addressaware
    print("\n\nAddressAware评估的相似度计算:")
    print("-" * 80)
    with open('evaluate_addressaware_pools.py', 'r') as f:
        addr_lines = f.readlines()
    
    for i, line in enumerate(addr_lines):
        if 'def compute_similarity' in line:
            for j in range(i, min(i+20, len(addr_lines))):
                print(f"Line {j+1}: {addr_lines[j]}", end='')
            break


def check_training_hyperparameters():
    """检查训练超参数"""
    print("\n" + "=" * 80)
    print("5. 检查 TRAINING HYPERPARAMETERS")
    print("=" * 80)
    
    # 查找baseline和addressaware的训练脚本
    baseline_scripts = []
    addr_scripts = []
    
    for root, dirs, files in os.walk('.'):
        for f in files:
            if 'baseline' in f.lower() and ('finetune' in f or 'train' in f) and f.endswith('.sh'):
                baseline_scripts.append(os.path.join(root, f))
            elif 'addressaware' in f.lower() and 'finetune' in f and f.endswith('.sh'):
                addr_scripts.append(os.path.join(root, f))
    
    print("\nBaseline训练脚本:")
    for script in baseline_scripts[:3]:
        print(f"  {script}")
        with open(script, 'r') as f:
            content = f.read()
            # 提取关键参数
            params = ['batch_size', 'learning_rate', 'lr', 'epoch', 'triplet_margin', 'data_ratio']
            for param in params:
                matches = re.findall(rf'--{param}\s+(\S+)', content)
                if matches:
                    print(f"    --{param}: {matches[0]}")
    
    print("\nAddressAware训练脚本:")
    for script in addr_scripts[:3]:
        print(f"  {script}")
        with open(script, 'r') as f:
            content = f.read()
            params = ['batch_size', 'learning_rate', 'lr', 'epoch', 'triplet_margin', 'data_ratio']
            for param in params:
                matches = re.findall(rf'--{param}\s+(\S+)', content)
                if matches:
                    print(f"    --{param}: {matches[0]}")


def main():
    print("\n" + "=" * 80)
    print("检查 BASELINE 模型是否使用了不规范的方法")
    print("=" * 80)
    
    os.chdir('/home/kun/Document/AAE/extern/jTrans')
    
    check_negative_sampling()
    check_data_leakage()
    check_position_embedding_trick()
    check_similarity_computation()
    check_training_hyperparameters()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
需要重点关注的问题:
1. ⚠️  Hard Negative Mining - 如果baseline使用但addressaware没用，会导致不公平对比
2. ⚠️  Data Leakage - 检查训练和评估数据是否独立
3. ⚠️  训练时长 - baseline是否训练了更多epochs
4. ⚠️  数据规模 - baseline和addressaware使用的训练数据量是否相同

建议的公平对比方案:
- 使用相同的negative sampling策略
- 使用相同的训练数据量 (data_ratio)
- 使用相同的训练epochs和batch size
- 使用相同的相似度计算方法 (都用cosine similarity with L2 norm)
    """)


if __name__ == '__main__':
    main()
