# Address-Aware Embedding 架构升级

## 修改概述

本次升级重构了两个核心组件：
1. **AddressPositionalEmbedding** (addr/daddr): 按照 Norm → Sin/Cos → Concat → MLP 流程
2. **VarPositionalEmbedding** (var offsets): 采用 Embedding + MLP 混合设计

---

## 第一部分: Address & DAddr 重构

### 核心设计理念

### 1. **Norm（标准化）**
- **目的**: 把原材料（地址数据）标准化，切成一样的大小
- **实现**: 将三个层级的位置归一化到 [0, 1] 范围
  - `binary_pos`: 在整个二进制文件中的位置（全局上下文）
  - `function_pos`: 在函数内的位置（中等上下文）
  - `bb_pos`: 在基本块内的位置（局部上下文）

### 2. **Sin/Cos（多尺度编码）**
- **目的**: 用显微镜把数据展开，同时看到轮廓和纹理
- **实现**: 
  - 使用多个频率的 sin/cos 函数编码每个位置
  - 低频 → 捕捉全局结构（轮廓）
  - 高频 → 捕捉局部细节（纹理）
  - 每个层级独立编码，保留层级特有的频率特征
- **输出**: 每个层级 → 16维 (8个sin + 8个cos)

### 3. **Concat（拼接）**
- **目的**: 把不同层级的零件组装在一起，而不是熔化
- **实现**: 
  - 将三个层级的编码直接拼接（不是求和）
  - 保留每个层级的独特信息
  - 就像组装零件：能看清每个部分的贡献
- **输出**: 48维 (3个层级 × 16维)

### 4. **MLP（控制器）**
- **目的**: 决定怎么读取仪表盘数据，翻译成最终指令
- **实现**:
  - **双MLP架构**:
    - `code_address_projection`: 处理 `address` token（代码地址/控制流）
    - `data_address_projection`: 处理 `daddr` token（数据地址/数据流）
  - 48维 → 256维 → d_model维
  - 自动学习如何解读多层级的位置信息

## 架构对比

### 修改前
```
3个归一化位置 → 直接MLP → d_model维
```
- 问题: 直接在原始归一化值上学习，缺乏多尺度感知

### 修改后
```
3个归一化位置 
  ↓ (各自独立)
Sin/Cos编码 (每个16维)
  ↓
拼接 (48维)
  ↓
MLP (双路：code/data)
  ↓
d_model维
```
- 优势: 
  1. 多尺度感知能力（低频+高频）
  2. 保留层级独立性（拼接而非相加）
  3. 代码/数据地址分离处理

## 技术细节

### Sin/Cos频率设计
```python
frequencies = 2^0, 2^1, 2^2, ..., 2^7
# 结果: [1, 2, 4, 8, 16, 32, 64, 128]
```
- 频率指数增长，覆盖从粗到细的多个尺度
- 对于位置 p ∈ [0, 1]:
  - sin(p × 1 × π) : 最粗的尺度（全局）
  - sin(p × 128 × π) : 最细的尺度（局部）

### 参数量变化
- **修改前**: MLP输入维度 = 3
- **修改后**: MLP输入维度 = 48 (3层 × 8对sin/cos)
- **影响**: 轻微增加参数量，但大幅提升表达能力

## 测试验证

运行 `test_address_embedding.py` 验证:
- ✓ 输出形状正确: [batch, seq_len, d_model]
- ✓ 非address token的embedding为0
- ✓ address和daddr使用不同的MLP，产生不同的embedding
- ✓ 多层级信息正确拼接和投影

## 使用方法

模型会自动使用新的架构，接口保持兼容：

```python
from address_embedding import AddressAwareBERTEmbedding

# 创建模型
model = AddressAwareBERTEmbedding(
    vocab_size=vocab_size,
    embed_size=768,
    num_sin_cos_features=8,  # Address编码的频率数量 (可选，默认8)
    use_address_embedding=True,
    use_var_embedding=True,
    vocab_stoi=vocab_stoi  # 需要提供词表映射
)

# 前向传播（接口不变）
embedding = model(
    token_ids,        # [batch, seq_len]
    segment_labels,   # [batch, seq_len]
    binary_pos,       # [batch, seq_len] in [0, 1]
    function_pos,     # [batch, seq_len] in [0, 1]
    bb_pos,           # [batch, seq_len] in [0, 1]
    var_offsets       # [batch, seq_len] integer offsets
)
```

### Var Offset 的输入格式

```python
# var_offsets 示例
var_offsets = torch.tensor([
    -200,   # 负溢出 → ID 257, dist=log2(73)≈6.19
    -128,   # 边界 → ID 0, dist=0
    -49,    # 精确区 → ID 79, dist=0
    0,      # 零 → ID 128, dist=0
    16,     # 精确区 → ID 144, dist=0
    128,    # 边界 → ID 256, dist=0
    4096,   # 正溢出 → ID 258, dist=log2(3969)≈11.95
    -1,     # 非var token (sentinel)
])
```

## 文件修改清单

### Address & DAddr
- `extern/jTrans/pretrain/address_aware/address_embedding.py`
  - 重构 `AddressPositionalEmbedding` 类
  - 添加 `_apply_sincos_encoding()` 方法
  - 更新文档注释

### Var Offset
- `extern/jTrans/pretrain/address_aware/address_embedding.py`
  - 完全重写 `VarPositionalEmbedding` 类
  - 添加 `_compute_token_id()` 方法
  - 添加 `_compute_log_distance()` 方法
  - 实现混合 Embedding + MLP 架构

### 测试脚本
- `test_address_embedding.py` - 测试 Address/DAddr 编码
- `test_var_embedding.py` - 测试 Var Offset 编码
- `analyze_var.py` - 统计 var offset 的分布

## 理论优势总结

### Address & DAddr
1. **多尺度感知**: Sin/Cos 编码同时捕捉全局和局部
2. **层级独立性**: Concat 保留每个层级的独特信息
3. **语义分离**: 代码地址和数据地址使用不同 MLP
4. **可解释性**: 流程清晰 (Norm → Sin/Cos → Concat → MLP)

### Var Offset
1. **精确表示**: 常见偏移 (72.93%) 使用 Embedding，独立语义
2. **平滑过渡**: Log-Distance 让大数压缩平滑，梯度稳定
3. **参数效率**: 溢出区共享 MLP，不需要为每个大数分配参数
4. **数据驱动**: 设计基于实际数据分布分析

## 下一步建议

1. **验证效果**: 在预训练和下游任务上对比新旧架构
2. **消融实验**: 
   - Address: 测试不同的 `num_sin_cos_features` (4, 8, 16)
   - Var: 测试不同的精确区范围 ([-64,64], [-128,128], [-256,256])
3. **可视化分析**:
   - 不同频率的 Sin/Cos 特征可视化
   - Var embedding 的 t-SNE 降维可视化
4. **性能对比**: 测量训练速度和内存使用

## 参考资料

- 数据分析: `analyze_var.py` 的统计结果
- 测试验证: `test_address_embedding.py`, `test_var_embedding.py`
- 原始论文设计: Transformer 的 Positional Encoding

---

## 第二部分: Var Offset 重构

### 核心设计逻辑

将输入空间划分为两个区域，采用混合策略：

1. **精确区 (In-Distribution)**: `[-128, 128]`
   - 每个整数都有独立语义
   - 使用 **Embedding Table** (259个token)
   
2. **溢出区 (Out-of-Distribution)**: `|x| > 128`
   - 只关心"大概多大"
   - 使用 **MLP + Log-Distance** 平滑处理

### 词表 (Vocabulary) 结构

共 **259 个 Token**:
- **ID 0-256**: 对应偏移量 `[-128, +128]` (257个精确值)
  - `ID = offset + 128`
  - 例: `-128 → ID 0`, `0 → ID 128`, `+128 → ID 256`
- **ID 257**: `[NEG_OVERFLOW]` 负溢出标记 (`x < -128`)
- **ID 258**: `[POS_OVERFLOW]` 正溢出标记 (`x > 128`)

### 数学公式

#### Step 1: Token ID 映射

```
ID = {
    257           if x < -128     (负溢出)
    x + 128       if -128 ≤ x ≤ 128  (精确值)
    258           if x > 128      (正溢出)
}
```

#### Step 2: Log-Distance 计算

用于 MLP 输入，平滑压缩大数：

```
Dist = {
    0                    if -128 ≤ x ≤ 128  (精确区，距离为0)
    log₂(|x| - 128 + 1)  if |x| > 128      (溢出区，log压缩)
}
```

**设计要点**:
- **为什么减去 128?** 让溢出部分从0开始，平滑过渡
- **为什么用 Log?** 栈偏移可能达到几万(Buffer)
  - 线性距离会导致梯度爆炸
  - Log 把 65536 压缩到 ~16，对神经网络友好
  - 例: `log₂(65408+1) ≈ 16.00`

#### Step 3: 最终融合

```
E_final = Embedding(token_id) + MLP(log_distance)
```

- **精确区**: `Embedding(ID) + MLP(0)` → 主要靠 Embedding
- **溢出区**: `Embedding(257/258) + MLP(log_dist)` → 主要靠 MLP

### 架构对比

#### 修改前
```
Raw Offset → Clamp to [-8192, 8192] → MLP → d_model
```
- 问题: 大数压缩不够平滑，梯度不稳定

#### 修改后
```
Raw Offset
  ↓
Token ID Mapping (259 tokens)
  ↓
Embedding Table Lookup → E_emb
  
Raw Offset
  ↓
Log-Distance Computation → Dist
  ↓
MLP Projection → E_mlp

Final: E_final = E_emb + E_mlp
```

### 优势总结

| 方面 | 精确区 [-128, 128] | 溢出区 \|x\| > 128 |
|------|-------------------|-------------------|
| **策略** | Embedding Table | MLP + Log-Distance |
| **表示能力** | 每个整数独立语义 | 连续平滑表示 |
| **参数效率** | 257个固定embedding | 共享MLP权重 |
| **梯度稳定性** | 离散优化 | Log压缩后稳定 |
| **适用场景** | 常见栈偏移 | 大buffer/heap偏移 |

### 实际数据分布

根据 `addr_pretrain.txt` 的统计分析：
- **[-128, 128] 范围**: 72.93%
- **[-256, 256] 范围**: 83.49%
- **[-512, 512] 范围**: 89.68%

→ **绝大部分 var offset 都在精确区内**，使用 Embedding Table 正好！

### 测试验证

运行 `test_var_embedding.py` 验证:
- ✓ Token ID 映射正确 (精确区/溢出区)
- ✓ Log-Distance 计算准确
- ✓ 精确区内距离为 0
- ✓ 溢出区距离 > 0 且经过 log 压缩
- ✓ 非var token 的 encoding 为 0
- ✓ 相同 offset 产生相同 embedding
- ✓ 精确区和溢出区的 embedding 不同

### 压缩效果示例

| 原始Offset | Log-Distance | 说明 |
|-----------|--------------|------|
| 0 | 0.00 | 精确区 |
| 128 | 0.00 | 边界 |
| 200 | 6.19 | 小溢出 |
| 4096 | 11.95 | 大溢出 |
| 65536 | 16.00 | 超大buffer |

→ 即使 offset 从 200 暴涨到 65536，log-distance 只从 6.19 增长到 16.00

## 使用方法
