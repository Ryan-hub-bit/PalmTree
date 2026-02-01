#!/usr/bin/env python3
"""
验证修复后的 dataloader_addressaware.py 是否正确处理 daddr tokens

测试要点:
1. daddr_match 是否被正确检查
2. daddr token 是否被正确提取
3. Position信息是否正确提取
4. 与真实pretrain数据对比
"""

import sys
import os

sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

def test_dataloader_with_daddr():
    """测试dataloader处理daddr的能力"""
    print("=" * 80)
    print("测试修复后的 dataloader_addressaware.py")
    print("=" * 80)
    
    # Create test corpus with daddr
    test_corpus = """push(0x1000:0.5:0.0:0.0) rbp mov(0x1001:0.5:0.0:0.1) rbp rsp
lea(0x2000:0.6:0.0:0.0) rax daddr(0x3000:0.7:0.0:0.0) mov(0x2005:0.6:0.0:0.2) rbx rax
call(0x3000:0.8:0.0:0.0) address(0x4000:0.9:0.0:0.0) ret(0x3005:0.8:0.0:0.5)
mov(0x4000:1.0:0.0:0.0) rax daddr(0x5000:1.1:0.0:0.0) add(0x4007:1.0:0.0:0.1) rax var(0xfffffff8)
"""
    
    # Write to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_corpus)
        temp_path = f.name
    
    try:
        # Load vocab
        import pickle
        vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt'
        
        # Build simple vocab from txt
        vocab_dict = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4}
        with open(vocab_path, 'r') as f:
            for line in f:
                token = line.strip()
                if token and token not in vocab_dict:
                    vocab_dict[token] = len(vocab_dict)
        
        # Create mock WordVocab object
        class MockVocab:
            def __init__(self, stoi):
                self.stoi = stoi
                self.itos = {v: k for k, v in stoi.items()}
                self.pad_index = stoi.get('<pad>', 0)
                self.unk_index = stoi.get('<unk>', 1)
                self.eos_index = stoi.get('<eos>', 2)
                self.sos_index = stoi.get('<sos>', 3)
                self.mask_index = stoi.get('<mask>', 4)
            
            def __len__(self):
                return len(self.stoi)
        
        vocab = MockVocab(vocab_dict)
        
        # Import and test dataloader
        from dataloader_addressaware import AddressAwareDataset
        
        print("\n创建 AddressAwareDataset...")
        dataset = AddressAwareDataset(
            corpus_path=temp_path,
            vocab=vocab,
            seq_len=100,
            token_mask_prob=0.0,  # No masking for testing
            on_memory=True,
            data_percentage=1.0
        )
        
        print(f"加载了 {len(dataset)} 个函数")
        
        # Test each function
        print("\n" + "=" * 80)
        print("解析结果:")
        print("=" * 80)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Get tokens (convert IDs back to tokens)
            token_ids = sample['bert_input'].tolist()
            tokens = [vocab.itos.get(tid, '<unk>') for tid in token_ids]
            
            # Get positions
            binary_pos = sample['binary_pos'].tolist()
            function_pos = sample['function_pos'].tolist()
            bb_pos = sample['bb_pos'].tolist()
            var_offsets = sample['var_offsets'].tolist()
            
            # Find actual tokens (non-padding)
            actual_len = 0
            for tid in token_ids:
                if tid == vocab.pad_index:
                    break
                actual_len += 1
            
            print(f"\n函数 {i+1}:")
            print(f"  原始: {test_corpus.strip().split(chr(10))[i]}")
            print(f"  Tokens: {tokens[:actual_len]}")
            
            # Check for daddr
            daddr_indices = [j for j, t in enumerate(tokens[:actual_len]) if t == 'daddr']
            if daddr_indices:
                print(f"  ✓ 找到 {len(daddr_indices)} 个 'daddr' tokens 在位置: {daddr_indices}")
                for idx in daddr_indices:
                    print(f"    daddr at {idx}: pos=({binary_pos[idx]:.2f}, {function_pos[idx]:.2f}, {bb_pos[idx]:.2f}), var={var_offsets[idx]}")
            else:
                print(f"  ✗ 未找到 'daddr' token!")
            
            # Check for address
            address_indices = [j for j, t in enumerate(tokens[:actual_len]) if t == 'address']
            if address_indices:
                print(f"  ✓ 找到 {len(address_indices)} 个 'address' tokens 在位置: {address_indices}")
            
            # Check for var
            var_indices = [j for j, t in enumerate(tokens[:actual_len]) if t == 'var']
            if var_indices:
                print(f"  ✓ 找到 {len(var_indices)} 个 'var' tokens 在位置: {var_indices}")
                for idx in var_indices:
                    print(f"    var at {idx}: offset={var_offsets[idx]}")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print("如果上面显示找到了 'daddr' tokens 并且position信息正确，")
    print("说明修复后的dataloader工作正常。")


def test_with_real_data():
    """使用真实pretrain数据测试"""
    print("\n" + "=" * 80)
    print("使用真实 Pretrain 数据测试")
    print("=" * 80)
    
    pretrain_path = '/data/kun/jtrans/addressaware/addr_pretrain.txt'
    
    if not os.path.exists(pretrain_path):
        print(f"⚠️  数据文件不存在: {pretrain_path}")
        return
    
    # Load vocab
    vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt'
    vocab_dict = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4}
    with open(vocab_path, 'r') as f:
        for line in f:
            token = line.strip()
            if token and token not in vocab_dict:
                vocab_dict[token] = len(vocab_dict)
    
    class MockVocab:
        def __init__(self, stoi):
            self.stoi = stoi
            self.itos = {v: k for k, v in stoi.items()}
            self.pad_index = stoi.get('<pad>', 0)
            self.unk_index = stoi.get('<unk>', 1)
            self.eos_index = stoi.get('<eos>', 2)
            self.sos_index = stoi.get('<sos>', 3)
            self.mask_index = stoi.get('<mask>', 4)
        
        def __len__(self):
            return len(self.stoi)
    
    vocab = MockVocab(vocab_dict)
    
    # Import dataloader
    from dataloader_addressaware import AddressAwareDataset
    
    print(f"\n加载真实数据 (前100行): {pretrain_path}")
    
    # Create a temp file with first 100 lines
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        with open(pretrain_path, 'r') as src:
            for i, line in enumerate(src):
                if i < 100:
                    f.write(line)
                else:
                    break
        temp_path = f.name
    
    try:
        dataset = AddressAwareDataset(
            corpus_path=temp_path,
            vocab=vocab,
            seq_len=512,
            token_mask_prob=0.0,
            on_memory=True,
            data_percentage=1.0
        )
        
        print(f"加载了 {len(dataset)} 个函数")
        
        # Statistics
        total_daddr = 0
        total_address = 0
        total_var = 0
        functions_with_daddr = 0
        
        print("\n统计前20个函数:")
        for i in range(min(20, len(dataset))):
            sample = dataset[i]
            token_ids = sample['bert_input'].tolist()
            tokens = [vocab.itos.get(tid, '<unk>') for tid in token_ids]
            
            # Count
            daddr_count = sum(1 for t in tokens if t == 'daddr')
            address_count = sum(1 for t in tokens if t == 'address')
            var_count = sum(1 for t in tokens if t == 'var')
            
            total_daddr += daddr_count
            total_address += address_count
            total_var += var_count
            
            if daddr_count > 0:
                functions_with_daddr += 1
            
            if daddr_count > 0:
                print(f"  函数 {i+1}: address={address_count}, daddr={daddr_count}, var={var_count}")
        
        print(f"\n总计 (前20个函数):")
        print(f"  address tokens: {total_address}")
        print(f"  daddr tokens: {total_daddr}")
        print(f"  var tokens: {total_var}")
        print(f"  包含daddr的函数: {functions_with_daddr}/20")
        
        if total_daddr > 0:
            print("\n✓ 成功从真实数据中提取了 daddr tokens!")
            print("  修复后的dataloader工作正常")
        else:
            print("\n⚠️  未找到 daddr tokens")
            print("  可能原因:")
            print("  1. 真实数据中确实没有daddr (需要检查数据生成)")
            print("  2. Dataloader仍有bug (需要再次检查代码)")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    print("\n" + "=" * 80)
    print("验证修复后的 dataloader_addressaware.py")
    print("=" * 80)
    
    try:
        test_dataloader_with_daddr()
        test_with_real_data()
        
        print("\n" + "=" * 80)
        print("验证完成")
        print("=" * 80)
        print("\n如果上面的测试都显示找到了daddr tokens，说明:")
        print("✓ dataloader_addressaware.py 的bug已经修复")
        print("✓ daddr_match 检查已经正确添加")
        print("✓ 可以用于重新生成pretrain数据或直接训练")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
