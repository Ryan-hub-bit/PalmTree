# Baseline效果好的原因分析

## 检查结果总结

### 1. ✅ Negative Sampling策略 - **公平**
- **Baseline**: 使用Hard Negative Mining (从同opt level的不同函数采样)
- **AddressAware**: 也使用Hard Negative Mining (完全相同策略)
- **结论**: 两边策略相同，公平

### 2. ✅ 相似度计算 - **公平**
- **Baseline**: 使用L2 normalization后计算cosine similarity
- **AddressAware**: 使用L2 normalization后计算cosine similarity  
- **结论**: 两边计算方法相同，公平

### 3. ✅ Position Embedding技巧 - **公平**
- **Baseline**: `position_embeddings = word_embeddings` (jTrans核心技巧)
- **AddressAware**: 使用独立的hierarchical position encoding (3-level sin/cos)
- **结论**: 各有不同的设计，都是合理架构

### 4. ⚠️ 训练超参数 - **可能不公平**
```bash
# Baseline
--batch_size: 16
--lr: 2e-5
--epoch: 2          # ← 只训练2个epoch
--triplet_margin: 0.5
--data_ratio: 1.0

# AddressAware  
--batch_size: 16
--lr: 2e-5
--epoch: 15         # ← 训练15个epoch
--triplet_margin: 0.2
--data_ratio: 1.0
```

**关键发现**:
- Baseline只训练**2 epochs**
- AddressAware训练**15 epochs**
- **但baseline可能已经过拟合或收敛**

### 5. ❓ 数据泄漏 - **需要进一步检查**
- 训练集: `func_blocks_baseline.json` + `ground_truth_baseline.json`
- 评估集: `eval/func_blocks_baseline.json` + `eval/pools/`
- **需要确认**: 这两个数据集是否完全独立（没有重叠的函数）

---

## 可能导致Baseline效果好的真实原因

### 理论1: **模型容量问题**
- Baseline使用简单的token embedding (所有token共享embedding空间)
- AddressAware使用复杂的dual MLP + hierarchical encoding
- **但**: AddressAware的pretrain dataloader有bug，`data_address_projection`从未被训练
- **结果**: AddressAware实际只用了50%的模型容量，而且架构更复杂导致优化困难

### 理论2: **数据效率**
- Baseline的`position_embeddings = word_embeddings`技巧使得模型可以直接从pretrain学到有用的representation
- AddressAware需要额外学习position encoding和address/daddr的区别，但pretrain数据没有daddr → 冷启动问题

### 理论3: **优化难度**
- Baseline是标准BERT架构，优化算法成熟
- AddressAware有多个position embedding分支，梯度可能不平衡
- Triplet margin: Baseline=0.5 (更宽松), AddressAware=0.2 (更严格) → 可能导致AddressAware收敛慢

### 理论4: **Evaluation bias**
- 如果evaluation pool中的函数在训练时见过 → data leakage
- 或者evaluation task本身更适合简单的token-level representation

---

## 建议的调查步骤

### 第一优先级: 修复AddressAware的根本问题
```bash
# 1. 重新生成pretrain数据（使用修复后的dataloader）
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
./run_addressaware_pretrain.sh

# 2. 从头训练pretrain模型（确保dual MLP都被训练）
# 这样data_address_projection才能学到有用的representation

# 3. 使用新的pretrain checkpoint重新finetune
cd /home/kun/Document/AAE/extern/jTrans
./run_finetune_addressaware.sh

# 4. 重新评估
python evaluate_addressaware_pools.py --model <new_checkpoint> ...
```

### 第二优先级: 验证数据独立性
```bash
# 检查训练集和评估集是否有重叠
python3 << 'EOF'
import json

# 加载训练集
with open('/data/kun/jtrans/baseline/func_blocks_baseline.json') as f:
    train_blocks = json.load(f)
train_func_ids = set(train_blocks.keys())

# 加载评估集  
with open('/data/kun/jtrans/baseline/eval/func_blocks_baseline.json') as f:
    eval_blocks = json.load(f)
eval_func_ids = set(eval_blocks.keys())

# 检查重叠
overlap = train_func_ids & eval_func_ids
print(f"训练集大小: {len(train_func_ids)}")
print(f"评估集大小: {len(eval_func_ids)}")
print(f"重叠数量: {len(overlap)}")
print(f"重叠比例: {len(overlap)/len(eval_func_ids)*100:.2f}%")

if len(overlap) > 0:
    print("\n⚠️  发现数据泄漏!")
else:
    print("\n✓  数据集独立")
EOF
```

### 第三优先级: 公平对比实验
```bash
# 使用相同的训练epochs重新训练baseline
# 选项1: Baseline训练15 epochs (匹配AddressAware)
./run_finetune_baseline.sh  # 修改epoch=15

# 选项2: AddressAware训练2 epochs (匹配Baseline)  
./run_finetune_addressaware.sh  # 修改epoch=2

# 然后对比结果
```

---

## 最可能的结论

**Baseline效果好的主要原因不是"不规范手段"，而是:**

1. **AddressAware的pretrain有重大bug** (daddr never used)
   - 导致50%模型容量未被训练
   - Finetune时需要从零学习daddr branch → 收敛慢/效果差

2. **架构复杂度 vs 数据质量**
   - AddressAware架构更复杂，需要更多高质量的pretrain数据
   - 但pretrain数据质量不足（缺少daddr） → 复杂架构反而成为负担

3. **简单就是美**
   - Baseline的`position_embeddings = word_embeddings`虽然简单，但非常有效
   - 对于binary code这种sequential data，可能不需要太复杂的position encoding

**建议**: 先修复pretrain bug并重新训练，再做公平对比。如果修复后AddressAware还是不如Baseline，那可能说明hierarchical address encoding对于function similarity这个任务来说过于复杂了。
