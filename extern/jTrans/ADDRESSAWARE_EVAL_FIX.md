# AddressAware Evaluation问题诊断与修复

## 问题描述
Evaluation结果非常差 (MRR: 0.0069, Recall@1: 0.10%)，远低于预期。

## 根本原因

### 核心问题：三个阶段的tokenization方式不一致

**Pretrain (dataloader_addressaware.py):**
- 从原始`instructions`字段解析address-aware格式
- 使用自定义`_parse_instruction()`解析每条指令
- 动态生成tokens和position arrays，天然对齐

**Finetune (data_json.py):**
- 从原始`instructions`字段解析address-aware格式
- 使用自定义`_parse_address_aware_function()`方法
- 动态生成tokens和position arrays，天然对齐

**Evaluation (原来的错误方式):**
- ❌ 使用预tokenized的`tokens`字段 + `encode_plus()`
- ❌ `encode_plus()`会用WordPiece重新切分
- ❌ 导致tokens与position arrays不对齐
- ❌ 或者直接使用预tokenized的`tokens`但没有重新解析位置信息

### 数据格式
```python
func_block = {
    'instructions': 'lea(0x401000:0.56:0.0:0.0) rdi address(0x404000:0.99:0.0:0.0)\tmov(...)',  # 原始格式
    'tokens': 'lea rdi address mov rax var',  # 预tokenized（不应该用于evaluation！）
    'binary_pos': [0.56, -1.0, 0.99, ...],    # 与预tokenized tokens对齐，但evaluation应重新生成
    ...
}
```

## 修复方案：保持三阶段一致

### 正确的方式
Pretrain、Finetune、Evaluation都必须从`instructions`字段解析，使用相同的解析逻辑：

```python
# ✓ 统一的解析方法（三个阶段都用这个）
func_str = func_block['instructions']  # 使用原始instructions
result = _parse_address_aware_function(func_str, tokenizer, max_length)
# 这样生成的tokens和positions完全一致
```

### 核心解析函数（三阶段共享）
1. `_parse_instruction()` - 解析单条指令，提取opcode、address、var等
2. `_parse_address_aware_function()` - 解析整个函数（tab分隔的指令序列）
3. 添加`<sos>`和`<eos>`标记
4. 生成segment labels（instruction IDs）
5. 转换tokens为IDs，生成position arrays

## 已修复的内容

### 1. evaluate_addressaware_pools.py
**完全重写tokenize_function()和添加解析函数:**
```python
# 新增函数（复制自finetune）
def _parse_instruction(inst_text):
    """解析单条address-aware指令"""
    # 处理opcode(0xADDR:bnorm:fnorm:bbnorm)
    # 处理address(), daddr(), var()
    # 返回tokens, positions, var_offsets

def _parse_address_aware_function(func_str, tokenizer, max_length):
    """解析整个函数（SAME as finetune）"""
    # Split by '\t' to get instructions
    # Parse each instruction
    # Add <sos> and <eos>
    # Generate segment labels
    # Return torch tensors

def tokenize_function(func_block, tokenizer, max_len=512):
    """使用instructions字段，调用_parse_address_aware_function"""
    func_str = func_block['instructions']  # NOT 'tokens'!
    result = _parse_address_aware_function(func_str, tokenizer, max_len)
    return result
```

### 2. evaluate_baseline_pools.py
- ✅ 添加`weights_only=False`参数消除torch.load警告

### 3. Tensor类型
- ✅ binary_pos/function_pos/bb_pos使用`torch.FloatTensor`（归一化值）
- ✅ token_ids/attention_mask/token_type_ids/var_offsets使用`torch.LongTensor`

## 三阶段对比

| 阶段 | 输入数据 | 解析方法 | 一致性 |
|------|---------|---------|--------|
| Pretrain | instructions (txt文件) | dataloader_addressaware.py | ✓ |
| Finetune | instructions (json字段) | data_json.py | ✓ |
| Evaluation (修复后) | instructions (json字段) | evaluate_addressaware_pools.py | ✓ |

**关键点**：三个阶段都从原始`instructions`解析，不使用预tokenized的`tokens`字段。

## 为什么不能用预tokenized的'tokens'字段？

1. **不一致性**：预tokenized的`tokens`是数据生成时parse的结果，可能与运行时的tokenizer行为不完全一致
2. **缺少特殊标记**：预tokenized的`tokens`没有`<sos>`和`<eos>`
3. **segment labels**：预tokenized没有生成segment labels（instruction IDs）
4. **动态对齐**：从instructions解析可以保证tokens和positions在同一个函数调用中生成，天然对齐

## 运行测试
```bash
cd /home/kun/Document/AAE/extern/jTrans
./run_addressaware_pool_evaluation.sh \
    /home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_5 \
    /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
```

现在三个阶段使用完全相同的tokenization方式，evaluation结果应该合理！
