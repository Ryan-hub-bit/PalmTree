# 🎯 Var Offset Embedding 重构完成

## ✅ 完成的工作

### 1. 核心设计实现

按照 **Embedding + MLP 混合架构** 完全重构了 `VarPositionalEmbedding` 类：

```
精确区 [-128, 128]     溢出区 |x| > 128
        ↓                      ↓
  Embedding(259)          Log-Distance
        ↓                      ↓
      E_emb                  MLP
        ↓                      ↓
        └──────────┬───────────┘
                   ↓
            E_final = E_emb + E_mlp
```

### 2. 关键公式

**Token ID 映射**:
- `x < -128`: ID = 257 ([NEG_OVERFLOW])
- `-128 ≤ x ≤ 128`: ID = x + 128 (精确值)
- `x > 128`: ID = 258 ([POS_OVERFLOW])

**Log-Distance 计算**:
- 精确区: `Dist = 0`
- 溢出区: `Dist = log₂(|x| - 128 + 1)`

### 3. 测试验证 ✓

所有测试通过 (`test_var_embedding.py`):
- ✓ Token ID 映射正确
- ✓ Log-Distance 计算准确 (65536 → 16.00)
- ✓ 精确区距离为 0，溢出区 > 0
- ✓ 相同 offset 产生相同 embedding
- ✓ 精确区和溢出区的 embedding 可区分

### 4. 数据驱动设计

基于 `addr_pretrain.txt` 的实际分析:
- **72.93%** 的 var offset 在 [-128, 128] 内
- **89.68%** 在 [-512, 512] 内
- → 精确区覆盖了绝大部分常见情况

## 🎨 设计亮点

### 优势 1: 精确 + 平滑兼顾
- **精确区**: 每个整数独立语义（如栈帧偏移 -8, -16, -24）
- **溢出区**: 连续平滑表示（如大buffer 4096, 65536）

### 优势 2: 梯度友好
```
原始值      Log压缩后
200      →  6.19
4096     →  11.95
65536    →  16.00
```
→ 即使 offset 增长 300 倍，log-distance 只增长 ~2.6 倍

### 优势 3: 参数高效
- 精确区: 257 个 embedding (常见值)
- 溢出区: 共享 MLP 权重 (罕见值)
- 总词表: 仅 259 tokens

### 优势 4: 可解释性强
每一步都有清晰的物理意义：
1. **Norm**: 划分精确区和溢出区
2. **ID Mapping**: 离散化常见值
3. **Log-Distance**: 压缩大数
4. **Fusion**: 混合离散和连续表示

## 📊 实际效果示例

### 示例 1: 常见栈变量
```python
offset = -49  # [rbp-0x31]
→ Token ID = 79 (精确区)
→ Log-Dist = 0
→ 纯靠 Embedding 表示
```

### 示例 2: 大缓冲区
```python
offset = 65536  # 大buffer
→ Token ID = 258 ([POS_OVERFLOW])
→ Log-Dist = 16.00
→ Embedding(258) + MLP(16.00)
```

## 🔧 技术实现

### 新增方法
1. `_compute_token_id()`: 将 offset 映射到 [0, 258]
2. `_compute_log_distance()`: 计算 log₂ 压缩距离
3. `forward()`: 混合 Embedding + MLP

### 词表结构
```
ID    范围           说明
0     offset=-128    边界最小值
1     offset=-127    
...   ...           精确区
128   offset=0       零点
...   ...           
256   offset=+128    边界最大值
257   x < -128       负溢出标记
258   x > 128        正溢出标记
```

## 🧪 测试覆盖

测试用例涵盖:
- ✓ 精确区边界: -128, 0, +128
- ✓ 精确区内部: -49, 16, 50, 127
- ✓ 负溢出: -200, -256
- ✓ 正溢出: 200, 256, 4096, 65536
- ✓ 非 var token: -1 (sentinel)
- ✓ 相同 offset 的一致性

## 📈 对比旧版

| 方面 | 旧版 (Raw MLP) | 新版 (Hybrid) |
|------|---------------|---------------|
| 精确区 | 线性投影 | Embedding Table |
| 溢出区 | Clamp + 线性 | Log + MLP |
| 参数量 | MLP权重 | Embedding(259) + MLP |
| 梯度稳定性 | 大数不稳定 | Log压缩稳定 |
| 语义表达 | 模糊 | 精确 + 平滑 |

## 🚀 下一步

1. **实验验证**:
   - 预训练任务上的 loss 对比
   - 下游任务 (函数相似度) 的效果

2. **消融实验**:
   - 测试不同精确区范围: [-64,64], [-128,128], [-256,256]
   - 测试不同 MLP 深度: 1层 vs 2层

3. **可视化分析**:
   - t-SNE 降维可视化 var embeddings
   - 观察精确区和溢出区的分布

## 📝 代码位置

- **主文件**: `extern/jTrans/pretrain/address_aware/address_embedding.py`
- **测试**: `test_var_embedding.py`
- **文档**: `ADDRESS_EMBEDDING_UPGRADE.md`

## 💡 设计灵感来源

这个设计融合了多个经典思想:
1. **离散化 + 连续化**: 类似 Word2Vec (常见词查表，罕见词共享)
2. **Log压缩**: 类似 Positional Encoding 的频率调制
3. **混合表示**: 类似 Hybrid Attention (local + global)

---

**总结**: 通过精确区和溢出区的差异化处理，实现了 **精确性** 和 **泛化性** 的最佳平衡！✨
