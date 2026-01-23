# jTrans_instr Complete Pipeline

完整的数据生成、公平池生成、finetune和evaluation流程。

**重要**: 所有阶段(pretrain, finetune, evaluation)使用相同的数据格式和分词策略，确保完全一致。详见 [CONSISTENCY_CHECK.md](CONSISTENCY_CHECK.md)

---

## 流程概览

```
1. 生成数据 → 2. 生成公平池 → 3. Finetune → 4. Evaluation
```

---

## 数据格式标准 (所有阶段统一)

**instruction分隔符**: 使用 `\t` (tab) 分隔不同instructions，空格分隔tokens

```
push rbp\tmov rsp , rbp\tsub rsp , 0x10\tcall instr_addr_5
```

**一致性保证**:
- ✅ Pretrain数据生成: 使用`\t`分隔
- ✅ Pretrain数据加载: 按`\t`解析instruction边界
- ✅ Finetune数据生成: 使用`\t`分隔
- ✅ Finetune数据加载: 按`\t`解析instruction边界
- ✅ Evaluation: 按`\t`解析instruction边界

**分词策略**:
- Pretrain: 训练tokenizer，生成`input_ids`和`instruction_ids`
- Finetune: **复用pretrain tokenizer**，只生成`instruction_ids`
- Evaluation: **复用pretrain tokenizer**，只生成`instruction_ids`

---

## Step 1: 生成数据

### 生成instruction-level数据集

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
./generate_instr.sh
```

### 交互式选项

脚本会自动检测是否有已存在的pickle文件：

- **如果检测到pickle文件**：
  - 选择 `y` → 复用现有pickle，跳过IDA提取（**推荐，节省时间**）
  - 选择 `n` → 删除旧文件，重新运行IDA提取（慢，需要数小时）

- **如果没有pickle文件**：
  - 自动运行完整pipeline（IDA Pro提取 + 数据生成）

### 输出文件

- `/data/kun/jtrans_instr/extract/` - Pickle文件（可复用）
- `/data/kun/jtrans_instr/func_blocks_instr.json` - 函数数据（使用`\t`分隔instructions）
- `/data/kun/jtrans_instr/ground_truth_instr.json` - 相似度标签

### 数据格式验证

检查数据格式是否正确：

```bash
# 查看一个函数的instructions
python3 -c "
import json
with open('/data/kun/jtrans_instr/func_blocks_instr.json') as f:
    data = json.load(f)
    func_id = list(data.keys())[0]
    # 替换\t为换行符，每行一条instruction
    print(data[func_id]['instructions'].replace('\t', '\n'))
"
```

每行应该是一条完整instruction（包含operator和operands）。
- ✅ 不依赖硬编码的指令列表
- ✅ 通用于任何架构

---

## Step 2: 生成公平池

### 为什么需要公平池？

为了**公平对比**三个模型（baseline、addressaware、jTrans_instr），需要确保：
- 使用相同的函数集合
- 相同的pool/query划分
- 相同的优化级别对

### 生成公平池文件

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr
./run_generate_pools.sh
```

### 输出文件

保存在 `/data/kun/jtransdata/fair_pools/`：

**Pool文件**（用于检索）：
```
pool_100_O0_vs_O3.json
pool_1000_O0_vs_O3.json
pool_10000_O0_vs_O3.json
pool_100_O1_vs_O3.json
pool_1000_O1_vs_O3.json
...
```

**Query文件**（查询函数）：
```
query_pool_100_O0_vs_O3.json
query_pool_1000_O0_vs_O3.json
...
```

**Metadata文件**（包含三个模型的ID映射）：
```
pool_100_O0_vs_O3_meta.json
pool_1000_O0_vs_O3_meta.json
...
```

### Pool文件格式

每个pool entry包含：
```json
{
  "binary": "coreutils",
  "function_name": "main",
  "opt": "O3",
  "baseline_func_id": "1234",
  "addressaware_func_id": "5678",
  "instr_func_id": "9012"
}
```

### Pool配置

- **Pool大小**：100, 1000, 10000
- **优化级别对**：O0_vs_O3, O1_vs_O3, O2_vs_O3
- **总计**：3 × 3 = 9 个pool配置

---

## Step 3: Finetune

### 运行Finetune训练

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr
./run_finetune_instr.sh
```

### 配置参数

编辑 `run_finetune_instr.sh` 修改训练参数：

```bash
# GPU设置
export CUDA_VISIBLE_DEVICES=1

# 数据路径
FUNC_BLOCKS="/data/kun/jtrans_instr/func_blocks_instr.json"
GROUND_TRUTH="/data/kun/jtrans_instr/ground_truth_instr.json"

# 模型路径
TOKENIZER="/home/kun/Document/AAE/extern/jTrans_instr/pretrain"
MODEL_PATH="/home/kun/Document/AAE/output/jtrans_instr/pretrain_run1/checkpoint_epoch_10"
OUTPUT_PATH="/home/kun/Document/AAE/output/jtrans_instr/finetune"

# 训练超参数
BATCH_SIZE=32          # 批量大小（GPU内存不够可改为16）
LR=1e-5               # 学习率
EPOCHS=5              # 训练轮数
FREEZE_CNT=10         # 冻结前N层
DATA_RATIO=1.0        # 使用全部数据（0.001用于快速测试）
```

### 快速测试模式

```bash
# 编辑 run_finetune_instr.sh，修改：
DATA_RATIO=0.001  # 仅使用500个样本
EPOCHS=1          # 只训练1轮
```

### 输出

模型保存在：
```
/home/kun/Document/AAE/output/jtrans_instr/finetune/
├── finetune_epoch_1/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── vocab.txt
├── finetune_epoch_2/
├── ...
└── finetune_epoch_5/
```

训练日志：
```
/home/kun/Document/AAE/output/jtrans_instr/finetune/finetune.log
```

---

## Step 4: Evaluation

### 运行评估

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr
./run_evaluate_instr.sh
```

### 配置参数

编辑 `run_evaluate_instr.sh`：

```bash
# 模型路径（使用finetune后的模型）
MODEL_PATH="/home/kun/Document/AAE/output/jtrans_instr/finetune/finetune_epoch_5"

# 数据路径
TOKENIZER="/home/kun/Document/AAE/extern/jTrans_instr/pretrain"
FUNC_BLOCKS="/data/kun/jtrans_instr/func_blocks_instr.json"

# 公平池路径
POOL_DIR="/data/kun/jtransdata/fair_pools"

# 结果输出路径
OUTPUT_DIR="/home/kun/Document/AAE/extern/jTrans_instr/eval_results"
```

### 评估配置

脚本会自动评估所有配置：

**Pool大小**：
- 100
- 1000  
- 10000

**优化级别对**：
- O0_vs_O3
- O1_vs_O3
- O2_vs_O3

**总计**：3 × 3 = 9 个评估

### 输出结果

结果保存在 `/home/kun/Document/AAE/extern/jTrans_instr/eval_results/`：

```
results_100_O0_vs_O3.txt
results_1000_O0_vs_O3.txt
results_10000_O0_vs_O3.txt
results_100_O1_vs_O3.txt
...
```

### 结果格式

每个文件包含：
```
Pool file: pool_1000_O0_vs_O3.json
Number of queries: 200
MRR:       0.8234
Recall@1:  0.7400
Recall@5:  0.9200
Recall@10: 0.9600
```

**指标说明**：
- **MRR** (Mean Reciprocal Rank): 平均倒数排名
- **Recall@K**: 前K个结果中包含正确答案的比例

---

## 公平对比三个模型

### 确保公平性

三个模型使用：
- ✅ 相同的函数集合
- ✅ 相同的pool/query划分
- ✅ 相同的优化级别对
- ✅ 相同的评估指标

### 对比结果

```bash
# Baseline结果
cat /path/to/baseline/eval_results/results_1000_O0_vs_O3.txt

# AddressAware结果
cat /path/to/addressaware/eval_results/results_1000_O0_vs_O3.txt

# jTrans_instr结果
cat /home/kun/Document/AAE/extern/jTrans_instr/eval_results/results_1000_O0_vs_O3.txt
```

---

## 关键技术细节

### Instruction边界检测

**方法**：使用 `\t` 分隔符

**生成时**（create_instr_dataset.py）：
```python
instruction_tokens = []
for instruction in all_instructions:
    tokens = tokenize_instruction(instruction)
    instruction_tokens.append(' '.join(tokens))

return '\t'.join(instruction_tokens)
```

**加载时**（data_json_instr.py, evaluate_instr_with_pools.py）：
```python
instructions = func_str.split('\t')
for instr in instructions:
    # Process each instruction
    ...
```

**优点**：
- ✅ 准确 - 边界明确定义
- ✅ 简单 - 直接split即可
- ✅ 通用 - 适用任何架构

### 与Baseline/AddressAware的区别

| 模型 | Position表示 | 特殊Token |
|------|-------------|----------|
| **Baseline** | `position_embeddings = word_embeddings` (共享) | `JUMP_ADDR_X` (token位置) |
| **AddressAware** | 层次化（binary + function + BB + var） | `JUMP_ADDR_X` |
| **jTrans_instr** | 标准BERT + instruction_embeddings | `instr_addr_{i}` (instruction索引) |

---

## 故障排除

### 问题1：找不到数据文件

**错误**：`FileNotFoundError: func_blocks_instr.json`

**解决**：
```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
./generate_instr.sh
```

### 问题2：找不到公平池文件

**错误**：`FileNotFoundError: pool_100_O0_vs_O3.json`

**解决**：
```bash
cd /home/kun/Document/AAE/extern/jTrans_instr
./run_generate_pools.sh
```

### 问题3：CUDA out of memory

**解决**：减小batch size
```bash
# 在 run_finetune_instr.sh 中
BATCH_SIZE=16  # 从32改为16
```

### 问题4：找不到预训练模型

**错误**：`FileNotFoundError: checkpoint_epoch_10`

**解决**：修改 `run_finetune_instr.sh` 中的 `MODEL_PATH`

---

## 完整示例

### 快速测试（小数据集）

```bash
# 1. 生成数据（复用pickle）
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
./generate_instr.sh
# 选择 y 复用pickle

# 2. 生成公平池
cd /home/kun/Document/AAE/extern/jTrans_instr
./run_generate_pools.sh

# 3. 快速finetune测试
# 编辑 run_finetune_instr.sh: DATA_RATIO=0.001, EPOCHS=1
./run_finetune_instr.sh

# 4. 评估
./run_evaluate_instr.sh

# 5. 查看结果
cat eval_results/results_1000_O0_vs_O3.txt
```

### 完整训练

```bash
# 1. 生成数据
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
./generate_instr.sh

# 2. 生成公平池
cd /home/kun/Document/AAE/extern/jTrans_instr
./run_generate_pools.sh

# 3. 完整finetune
# 编辑 run_finetune_instr.sh: DATA_RATIO=1.0, EPOCHS=5
./run_finetune_instr.sh

# 4. 完整评估
./run_evaluate_instr.sh

# 5. 批量查看结果
for f in eval_results/results_*.txt; do
    echo "=== $f ==="
    cat "$f"
    echo ""
done
```

---

## 文件清单

### 数据生成
- `datautils/generate_instr.sh` - 数据生成pipeline
- `datautils/run_instr.py` - IDA批量处理
- `datautils/process_instr.py` - IDA Pro脚本
- `datautils/create_instr_dataset.py` - 生成JSON数据

### 训练和评估
- `data_json_instr.py` - 数据加载器
- `finetune_instr.py` - Finetune脚本
- `evaluate_instr_with_pools.py` - 评估脚本
- `run_finetune_instr.sh` - Finetune启动脚本
- `run_evaluate_instr.sh` - 评估启动脚本

### 公平池生成
- `generate_fair_pools_all.py` - 公平池生成脚本
- `run_generate_pools.sh` - 公平池生成启动脚本

### 模型
- `pretrain/model_instr.py` - Instruction-level BERT模型
- `pretrain/config.json` - 模型配置
- `pretrain/vocab.txt` - 词表

---

**最后更新**: 2026-01-23
