# Address-Aware Model Evaluation Guide

## 文件说明

已为 Address-Aware 模型创建了完整的 evaluation pipeline:

### 生成的文件

1. **create_filtered_addressaware_pools.py** - 创建评估 pool 数据
2. **evaluate_addressaware_pools.py** - 在 pool 上评估模型
3. **run_create_addressaware_pools.sh** - 创建 pool 的脚本
4. **run_addressaware_pool_evaluation.sh** - 运行评估的脚本
5. **test_addressaware_evaluation.sh** - 快速测试脚本

---

## 使用流程

### 步骤 1: 创建 Evaluation Pools

生成评估用的 pool 数据（只需运行一次）：

```bash
bash run_create_addressaware_pools.sh
```

这会从 `func_blocks_addr.json` 和 `ground_truth_addr.json` 创建 pool 文件：
- Pool sizes: 100, 1000, 10000
- Opt pairs: O0_vs_O3, O1_vs_O3, O2_vs_O3
- 输出位置: `/data/kun/jtrans/addressaware/eval/pools_filtered/`

**过滤策略:**
- 排除 instruction count <= 10 的函数
- 排除 query 和 GT 完全相同的 instruction pairs
- Pool 去重: 确保每个 instruction sequence 只出现一次

---

### 步骤 2: 运行 Evaluation

在 finetuned 模型上评估：

```bash
# 评估所有 pools
bash run_addressaware_pool_evaluation.sh \
  /home/kun/Document/AAE/output/jtrans/addressaware_finetune_sincos/finetune_epoch_2 \
  /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware

# 只评估特定 pool size (例如 1000)
bash run_addressaware_pool_evaluation.sh \
  /home/kun/Document/AAE/output/jtrans/addressaware_finetune_sincos/finetune_epoch_2 \
  /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
  1000

# 只评估特定 opt pair (例如 O0_vs_O3)
bash run_addressaware_pool_evaluation.sh \
  /home/kun/Document/AAE/output/jtrans/addressaware_finetune_sincos/finetune_epoch_2 \
  /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
  10000 \
  O0_vs_O3
```

**评估指标:**
- MRR (Mean Reciprocal Rank)
- Recall@1
- Recall@5
- Recall@10

---

### 步骤 3: 快速测试 (可选)

在运行完整评估前，可以先用小数据测试：

```bash
bash test_addressaware_evaluation.sh
```

这会：
1. 创建 size=100 的测试 pool
2. 在 O0_vs_O3 pool 上运行评估
3. 验证整个 pipeline 是否正常工作

---

## 数据要求

### 输入数据

1. **func_blocks_addr.json** - Address-aware 函数数据
   - 位置: `/data/kun/jtrans/addressaware/func_blocks_addr.json`
   - 格式: 
     ```json
     {
       "func_id": {
         "tokens": "mov(0xADDR:0.1:0.2:0.3) rax address(0xADDR2:0.4:0.5:0.6) ...",
         "num_instructions": 25,
         "binary": "binary_name",
         "function": "func_name"
       }
     }
     ```

2. **ground_truth_addr.json** - Ground truth pairs
   - 位置: `/data/kun/jtrans/addressaware/ground_truth_addr.json`
   - 格式:
     ```json
     {
       "pairs": [
         {
           "O0": "func_id_1",
           "O1": "func_id_2",
           "O2": "func_id_3",
           "O3": "func_id_4",
           "binary_name": "...",
           "function_name": "..."
         }
       ]
     }
     ```

### 输出数据

1. **Pool 文件** - `/data/kun/jtrans/addressaware/eval/pools_filtered/`
   - 格式: `pool_O0_vs_O3_1000.json`
   - 包含: pool pairs + queries

2. **评估结果** - `/home/kun/Document/AAE/output/jtrans/`
   - 格式: `addressaware_eval_TIMESTAMP.json`
   - 包含: 所有 metrics

---

## 与 Baseline 的差异

| 特性 | Baseline | Address-Aware |
|------|----------|---------------|
| Tokenization | BertTokenizer | Address-aware parsing |
| Position Encoding | position=word trick | Hierarchical (binary/function/bb) |
| Address Tokens | - | address, daddr 区分 |
| Var Tokens | - | var(0xXX) with offset |
| Model Loading | BinBertModel | AddressAwareBertWrapper |

---

## 故障排查

### 问题 1: "vocab.txt not found"

确保 tokenizer 目录包含 vocab.txt:
```bash
ls /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt
```

### 问题 2: "Invalid token IDs"

检查 vocab_size 是否匹配:
```bash
wc -l /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab.txt
# 应该与 model config.json 中的 vocab_size 一致
```

### 问题 3: "CUDA out of memory"

降低 batch_size:
```bash
# 在 run_addressaware_pool_evaluation.sh 中修改
BATCH_SIZE=16  # 从 32 改为 16
```

---

## 完整 Workflow

```bash
# 1. Pretrain (如果还没有 pretrain 模型)
cd pretrain/address_aware
bash run_addressaware_pretrain.sh

# 2. Finetune
cd ../..
bash run_finetune_addressaware.sh

# 3. Create evaluation pools (一次性)
bash run_create_addressaware_pools.sh

# 4. Evaluate
bash run_addressaware_pool_evaluation.sh \
  output/jtrans/addressaware_finetune_sincos/finetune_epoch_2 \
  pretrain/address_aware

# 5. 查看结果
cat output/jtrans/addressaware_eval_*.json
```

---

## 预期结果

良好的 address-aware 模型应该有:
- **Recall@1**: > 0.60 (60%+)
- **Recall@10**: > 0.85 (85%+)
- **MRR**: > 0.70

如果结果显著低于这些值，检查:
1. Pretrain 是否充分 (epoch >= 10)
2. Finetune 配置 (learning rate, margin, etc.)
3. 数据质量 (address normalization 是否正确)
4. vocab_stoi 是否正确传递

---

## 参考

- Baseline evaluation: `run_baseline_pool_evaluation.sh`
- Finetune script: `run_finetune_addressaware.sh`
- Model wrapper: `finetune.py` (AddressAwareBertWrapper)
- Pretrain: `pretrain/address_aware/train_addressaware.py`
