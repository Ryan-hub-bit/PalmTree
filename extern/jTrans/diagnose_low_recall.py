#!/usr/bin/env python3
"""
深度诊断 AddressAware 模型 Recall@1 低的问题

检查项目:
1. 模型embedding是否正常工作
2. Position encoding是否有效
3. 同一函数不同opt的embedding相似度
4. 不同函数的embedding相似度
5. Embedding的分布和范数
6. 评估代码的正确性
"""

import torch
import numpy as np
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans')
sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

def load_model_and_data(checkpoint_path, data_path):
    """加载模型和数据"""
    print("=" * 80)
    print("1. 加载模型和数据")
    print("=" * 80)
    
    # Import without transformers (will use config from checkpoint)
    import importlib.util
    spec = importlib.util.spec_from_file_location("bert_model", "/home/kun/anaconda3/envs/jtrans/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py")
    if spec is None:
        # Fallback: try to import normally
        try:
            from transformers import BertConfig, BertModel
        except:
            print("⚠️  无法导入transformers，使用简化版本")
            return None, None, None, None
    else:
        bert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bert_module)
        BertModel = bert_module.BertModel
        BertConfig = bert_module.BertConfig
    
    from address_embedding import AddressAwareBERTEmbedding
    
    # Load vocab
    vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt'
    print(f"\n加载词汇表: {vocab_path}")
    
    vocab_dict = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4}
    with open(vocab_path, 'r') as f:
        for line in f:
            token = line.strip()
            if token and token not in vocab_dict:
                vocab_dict[token] = len(vocab_dict)
    
    print(f"词汇表大小: {len(vocab_dict)}")
    print(f"特殊tokens: address={vocab_dict.get('address')}, daddr={vocab_dict.get('daddr')}, var={vocab_dict.get('var')}")
    
    # Load config
    config_path = os.path.join(checkpoint_path, 'config.json')
    print(f"\n加载配置: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = BertConfig(**config_dict)
    print(f"模型配置:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    
    # Create model
    print(f"\n创建模型...")
    bert_model = BertModel(config, add_pooling_layer=False)
    
    # Replace embeddings with address-aware version
    bert_model.embeddings = AddressAwareBERTEmbedding(
        vocab_size=config.vocab_size,
        embed_size=config.hidden_size,
        dropout=0.1,
        max_len=config.max_position_embeddings,
        use_address_embedding=True,
        use_var_embedding=True,
        segment_types=256,
        vocab_stoi=vocab_dict
    )
    
    # Load weights
    weights_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
    print(f"\n加载权重: {weights_path}")
    
    state_dict = torch.load(weights_path, map_location='cpu')
    missing_keys, unexpected_keys = bert_model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  缺失的keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"⚠️  意外的keys: {unexpected_keys[:5]}...")
    
    bert_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)
    print(f"模型已加载到: {device}")
    
    # Load data
    print(f"\n加载数据: {data_path}")
    with open(data_path, 'r') as f:
        func_blocks = json.load(f)
    
    print(f"数据大小: {len(func_blocks)} functions")
    
    return bert_model, vocab_dict, func_blocks, device


def test_embedding_sanity(model, vocab_dict, device):
    """测试embedding的基本健全性"""
    print("\n" + "=" * 80)
    print("2. Embedding 健全性检查")
    print("=" * 80)
    
    # Test 1: 检查embedding是否都是0
    print("\n测试1: 检查embedding layers是否已初始化")
    
    token_emb_norm = model.embeddings.token.weight.norm().item()
    print(f"  Token embedding norm: {token_emb_norm:.4f}")
    
    if hasattr(model.embeddings, 'code_address_projection'):
        code_addr_norm = model.embeddings.code_address_projection[0].weight.norm().item()
        print(f"  Code address MLP norm: {code_addr_norm:.4f}")
    
    if hasattr(model.embeddings, 'data_address_projection'):
        data_addr_norm = model.embeddings.data_address_projection[0].weight.norm().item()
        print(f"  Data address MLP norm: {data_addr_norm:.4f}")
    
    if token_emb_norm < 0.1:
        print("  ⚠️  Token embedding接近0，可能未正确初始化!")
    else:
        print("  ✓ Embedding已初始化")
    
    # Test 2: 相同输入应该产生相同输出
    print("\n测试2: 确定性检查")
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    
    input_ids = torch.randint(0, min(100, len(vocab_dict)), (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    token_type_ids = torch.zeros(batch_size, seq_len).long().to(device)
    binary_pos = torch.rand(batch_size, seq_len).to(device)
    function_pos = torch.rand(batch_size, seq_len).to(device)
    bb_pos = torch.rand(batch_size, seq_len).to(device)
    var_offsets = torch.full((batch_size, seq_len), -1).long().to(device)
    
    with torch.no_grad():
        # Get embeddings
        embeddings1 = model.embeddings(
            input_ids, token_type_ids,
            binary_pos, function_pos, bb_pos, var_offsets
        )
        
        embeddings2 = model.embeddings(
            input_ids, token_type_ids,
            binary_pos, function_pos, bb_pos, var_offsets
        )
        
        diff = (embeddings1 - embeddings2).abs().max().item()
        print(f"  两次forward的最大差异: {diff:.10f}")
        
        if diff > 1e-6:
            print("  ⚠️  模型不是确定性的!")
        else:
            print("  ✓ 模型是确定性的")
    
    # Test 3: Position encoding是否有效
    print("\n测试3: Position encoding效果")
    
    with torch.no_grad():
        # Same tokens, different positions
        same_tokens = torch.full((2, seq_len), vocab_dict.get('mov', 50)).to(device)
        
        # Position 1: all zeros
        pos_zero = torch.zeros(2, seq_len).to(device)
        emb_zero = model.embeddings(
            same_tokens, token_type_ids,
            pos_zero, pos_zero, pos_zero, var_offsets
        )
        
        # Position 2: different values
        pos_diff = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(2, 1).to(device)
        emb_diff = model.embeddings(
            same_tokens, token_type_ids,
            pos_diff, pos_diff, pos_diff, var_offsets
        )
        
        # Check difference
        pos_effect = (emb_zero - emb_diff).abs().mean().item()
        print(f"  Position变化导致的embedding差异: {pos_effect:.6f}")
        
        if pos_effect < 0.01:
            print("  ⚠️  Position encoding几乎没有效果!")
        else:
            print("  ✓ Position encoding有明显效果")


def test_same_function_similarity(model, vocab_dict, func_blocks, device):
    """测试同一函数不同优化级别的相似度"""
    print("\n" + "=" * 80)
    print("3. 同一函数不同优化级别的相似度")
    print("=" * 80)
    
    from data_json import FunctionDataset_CL_AddressAware_JSON
    
    # Load ground truth to find function pairs
    gt_path = '/data/kun/jtrans/addressaware/ground_truth_addr.json'
    
    if not os.path.exists(gt_path):
        print(f"⚠️  Ground truth文件不存在: {gt_path}")
        return
    
    print(f"\n加载ground truth: {gt_path}")
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    pairs = gt_data.get('pairs', [])
    print(f"总共 {len(pairs)} 个function pairs")
    
    # Sample a few pairs
    sample_size = min(20, len(pairs))
    sampled_pairs = pairs[:sample_size]
    
    print(f"\n测试前{sample_size}个pairs:")
    
    from evaluate_addressaware_pools import parse_address_aware_function, tokenize_function
    
    similarities = []
    
    for i, pair in enumerate(sampled_pairs):
        func_id1 = str(pair['func_id1'])
        func_id2 = str(pair['func_id2'])
        opt1 = pair['opt1']
        opt2 = pair['opt2']
        
        if func_id1 not in func_blocks or func_id2 not in func_blocks:
            continue
        
        # Get embeddings for both
        try:
            # Parse function 1
            func1 = func_blocks[func_id1]
            func_str1 = func1.get('tokens', func1.get('instructions', ''))
            
            tokens1, binary_pos1, function_pos1, bb_pos1, var_offsets1 = parse_address_aware_function(func_str1)
            
            # Add CLS and SEP
            tokens1 = ['<sos>'] + tokens1 + ['<eos>']
            binary_pos1 = [-1.0] + binary_pos1 + [-1.0]
            function_pos1 = [-1.0] + function_pos1 + [-1.0]
            bb_pos1 = [-1.0] + bb_pos1 + [-1.0]
            var_offsets1 = [-1] + var_offsets1 + [-1]
            
            # Convert to IDs
            max_len = 512
            token_ids1 = [vocab_dict.get(t, vocab_dict['<unk>']) for t in tokens1]
            
            # Truncate/pad
            if len(token_ids1) > max_len:
                token_ids1 = token_ids1[:max_len]
                binary_pos1 = binary_pos1[:max_len]
                function_pos1 = function_pos1[:max_len]
                bb_pos1 = bb_pos1[:max_len]
                var_offsets1 = var_offsets1[:max_len]
            else:
                padding = max_len - len(token_ids1)
                token_ids1 += [vocab_dict['<pad>']] * padding
                binary_pos1 += [-1.0] * padding
                function_pos1 += [-1.0] * padding
                bb_pos1 += [-1.0] * padding
                var_offsets1 += [-1] * padding
            
            # Same for function 2
            func2 = func_blocks[func_id2]
            func_str2 = func2.get('tokens', func2.get('instructions', ''))
            
            tokens2, binary_pos2, function_pos2, bb_pos2, var_offsets2 = parse_address_aware_function(func_str2)
            
            tokens2 = ['<sos>'] + tokens2 + ['<eos>']
            binary_pos2 = [-1.0] + binary_pos2 + [-1.0]
            function_pos2 = [-1.0] + function_pos2 + [-1.0]
            bb_pos2 = [-1.0] + bb_pos2 + [-1.0]
            var_offsets2 = [-1] + var_offsets2 + [-1]
            
            token_ids2 = [vocab_dict.get(t, vocab_dict['<unk>']) for t in tokens2]
            
            if len(token_ids2) > max_len:
                token_ids2 = token_ids2[:max_len]
                binary_pos2 = binary_pos2[:max_len]
                function_pos2 = function_pos2[:max_len]
                bb_pos2 = bb_pos2[:max_len]
                var_offsets2 = var_offsets2[:max_len]
            else:
                padding = max_len - len(token_ids2)
                token_ids2 += [vocab_dict['<pad>']] * padding
                binary_pos2 += [-1.0] * padding
                function_pos2 += [-1.0] * padding
                bb_pos2 += [-1.0] * padding
                var_offsets2 += [-1] * padding
            
            # Create tensors
            input_ids = torch.tensor([token_ids1, token_ids2]).to(device)
            attention_mask = torch.tensor([
                [1 if tid != vocab_dict['<pad>'] else 0 for tid in token_ids1],
                [1 if tid != vocab_dict['<pad>'] else 0 for tid in token_ids2]
            ]).to(device)
            token_type_ids = torch.zeros_like(input_ids).to(device)
            
            binary_pos_t = torch.tensor([binary_pos1, binary_pos2], dtype=torch.float32).to(device)
            function_pos_t = torch.tensor([function_pos1, function_pos2], dtype=torch.float32).to(device)
            bb_pos_t = torch.tensor([bb_pos1, bb_pos2], dtype=torch.float32).to(device)
            var_offsets_t = torch.tensor([var_offsets1, var_offsets2], dtype=torch.long).to(device)
            
            # Get embeddings
            with torch.no_grad():
                embeddings = model.embeddings(
                    input_ids, token_type_ids,
                    binary_pos_t, function_pos_t, bb_pos_t, var_offsets_t
                )
                
                # Pass through encoder
                outputs = model.encoder(
                    embeddings,
                    attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                )
                
                sequence_output = outputs[0]
                # Get CLS token
                cls_emb = sequence_output[:, 0, :]
                
                # Compute similarity
                emb1 = cls_emb[0]
                emb2 = cls_emb[1]
                
                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                
                similarities.append(sim)
                
                if i < 5:  # Print first 5
                    print(f"  Pair {i+1}: {opt1} vs {opt2}, similarity={sim:.4f}")
        
        except Exception as e:
            print(f"  Error processing pair {i+1}: {e}")
            continue
    
    if similarities:
        print(f"\n统计 (同一函数不同opt):")
        print(f"  平均相似度: {np.mean(similarities):.4f}")
        print(f"  中位相似度: {np.median(similarities):.4f}")
        print(f"  最小相似度: {np.min(similarities):.4f}")
        print(f"  最大相似度: {np.max(similarities):.4f}")
        
        if np.mean(similarities) < 0.5:
            print("\n  ⚠️  同一函数的相似度太低! 模型没有学到函数语义!")
        else:
            print("\n  ✓ 相似度合理")
    else:
        print("\n⚠️  无法计算相似度")


def test_embedding_distribution(model, vocab_dict, func_blocks, device):
    """测试embedding的分布"""
    print("\n" + "=" * 80)
    print("4. Embedding 分布检查")
    print("=" * 80)
    
    from evaluate_addressaware_pools import parse_address_aware_function
    
    # Sample some functions
    sample_ids = list(func_blocks.keys())[:50]
    
    print(f"\n生成{len(sample_ids)}个函数的embeddings...")
    
    embeddings_list = []
    
    for func_id in tqdm(sample_ids):
        func = func_blocks[func_id]
        func_str = func.get('tokens', func.get('instructions', ''))
        
        try:
            tokens, binary_pos, function_pos, bb_pos, var_offsets = parse_address_aware_function(func_str)
            
            tokens = ['<sos>'] + tokens + ['<eos>']
            binary_pos = [-1.0] + binary_pos + [-1.0]
            function_pos = [-1.0] + function_pos + [-1.0]
            bb_pos = [-1.0] + bb_pos + [-1.0]
            var_offsets = [-1] + var_offsets + [-1]
            
            max_len = 512
            token_ids = [vocab_dict.get(t, vocab_dict['<unk>']) for t in tokens]
            
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
                binary_pos = binary_pos[:max_len]
                function_pos = function_pos[:max_len]
                bb_pos = bb_pos[:max_len]
                var_offsets = var_offsets[:max_len]
            else:
                padding = max_len - len(token_ids)
                token_ids += [vocab_dict['<pad>']] * padding
                binary_pos += [-1.0] * padding
                function_pos += [-1.0] * padding
                bb_pos += [-1.0] * padding
                var_offsets += [-1] * padding
            
            input_ids = torch.tensor([token_ids]).to(device)
            attention_mask = torch.tensor([[1 if tid != vocab_dict['<pad>'] else 0 for tid in token_ids]]).to(device)
            token_type_ids = torch.zeros_like(input_ids).to(device)
            
            binary_pos_t = torch.tensor([binary_pos], dtype=torch.float32).to(device)
            function_pos_t = torch.tensor([function_pos], dtype=torch.float32).to(device)
            bb_pos_t = torch.tensor([bb_pos], dtype=torch.float32).to(device)
            var_offsets_t = torch.tensor([var_offsets], dtype=torch.long).to(device)
            
            with torch.no_grad():
                embeddings = model.embeddings(
                    input_ids, token_type_ids,
                    binary_pos_t, function_pos_t, bb_pos_t, var_offsets_t
                )
                
                outputs = model.encoder(
                    embeddings,
                    attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                )
                
                cls_emb = outputs[0][:, 0, :].cpu().numpy()
                embeddings_list.append(cls_emb[0])
        
        except Exception as e:
            continue
    
    if embeddings_list:
        embeddings_array = np.array(embeddings_list)
        
        print(f"\nEmbedding统计:")
        print(f"  形状: {embeddings_array.shape}")
        print(f"  均值: {embeddings_array.mean():.6f}")
        print(f"  标准差: {embeddings_array.std():.6f}")
        print(f"  最小值: {embeddings_array.min():.6f}")
        print(f"  最大值: {embeddings_array.max():.6f}")
        
        # Check if all embeddings are similar
        norms = np.linalg.norm(embeddings_array, axis=1)
        print(f"\n  L2范数统计:")
        print(f"    均值: {norms.mean():.4f}")
        print(f"    标准差: {norms.std():.4f}")
        print(f"    最小值: {norms.min():.4f}")
        print(f"    最大值: {norms.max():.4f}")
        
        # Pairwise similarities
        normalized = embeddings_array / (np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8)
        sim_matrix = np.dot(normalized, normalized.T)
        
        # Get off-diagonal similarities (different functions)
        mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
        off_diag_sims = sim_matrix[mask]
        
        print(f"\n  不同函数间的相似度:")
        print(f"    均值: {off_diag_sims.mean():.4f}")
        print(f"    标准差: {off_diag_sims.std():.4f}")
        print(f"    最小值: {off_diag_sims.min():.4f}")
        print(f"    最大值: {off_diag_sims.max():.4f}")
        
        if off_diag_sims.mean() > 0.9:
            print("\n  ⚠️  所有embedding都太相似! 模型可能退化了!")
        elif off_diag_sims.mean() < 0.3:
            print("\n  ✓ Embedding分布合理")
        else:
            print("\n  ⚠️  Embedding相似度偏高")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to finetuned model')
    parser.add_argument('--data', type=str, 
                       default='/data/kun/jtrans/addressaware/func_blocks_addr.json',
                       help='Path to function blocks')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("AddressAware 模型深度诊断")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    
    try:
        # Load model and data
        model, vocab_dict, func_blocks, device = load_model_and_data(args.checkpoint, args.data)
        
        # Run tests
        test_embedding_sanity(model, vocab_dict, device)
        test_same_function_similarity(model, vocab_dict, func_blocks, device)
        test_embedding_distribution(model, vocab_dict, func_blocks, device)
        
        print("\n" + "=" * 80)
        print("诊断完成")
        print("=" * 80)
        print("\n如果发现问题:")
        print("1. Embedding未初始化 → 检查checkpoint是否正确加载")
        print("2. Position encoding无效 → 检查AddressAwareBERTEmbedding")
        print("3. 同一函数相似度低 → 模型训练不足或数据有问题")
        print("4. 所有embedding相似 → 模型退化，需要重新训练")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
