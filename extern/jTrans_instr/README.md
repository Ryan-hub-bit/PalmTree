# jTrans_instr: Instruction-Level Jump Addressing

This variant of jTrans uses instruction-level addressing for jump targets instead of token-level addressing.

## Key Differences from Baseline jTrans

### 1. Jump Address Representation
- **Baseline jTrans**: Uses `JUMP_ADDR_X` where X is the absolute token position
- **jTrans_instr**: Uses `instr_addr_{i}` where i is the instruction index

Example:
```
# Baseline jTrans:
jz JUMP_ADDR_45

# jTrans_instr:
jz instr_addr_12
```

### 2. Position Embeddings
- **Baseline jTrans**: `position_embeddings = word_embeddings` (jTrans trick)
- **jTrans_instr**: Uses standard BERT absolute position embeddings

### 3. JTP (Jump-Target Prediction)
- **Baseline jTrans**: Predicts absolute token position (0-511)
- **jTrans_instr**: Predicts instruction index that the jump targets

## Directory Structure

```
jTrans_instr/
├── README.md                    # This file
├── pretrain/                    # Pretraining code
│   ├── model_instr.py          # Model with standard BERT embeddings
│   ├── dataloader_instr.py     # Data loader with instruction-level addressing
│   ├── train_instr.py          # Training script
│   ├── vocab_instr.txt         # Vocabulary with instr_addr_{i} tokens
│   └── run_instr_pretrain.sh   # Launch script
├── finetune.py                  # Finetuning script
├── data_json_instr.py           # Data loader for finetuning
└── evaluate_instr.py            # Evaluation script
```

## Usage

### Pretraining
```bash
cd pretrain
./run_instr_pretrain.sh
```

### Finetuning
```bash
python finetune.py \
    --model_type instr \
    --data_type json \
    --func_blocks /path/to/func_blocks_instr.json \
    --ground_truth /path/to/ground_truth.json \
    --tokenizer ./pretrain \
    --model_path ./pretrain/output/checkpoint_epoch_10 \
    --output_path ./output/finetune
```

### Evaluation
```bash
python evaluate_instr.py \
    --model_path ./output/finetune/finetune_epoch_5 \
    --func_blocks /path/to/func_blocks_instr.json \
    --pool_file /path/to/pool.json \
    --query_file /path/to/query.json
```

## Data Format

Input data should have jump addresses converted to instruction-level format:

```
# Original assembly:
mov rax, rbx
test rax, rax
jz loc_140001234
add rax, 1

# Converted to jTrans_instr format (assuming jz targets instruction 15):
mov rax rbx
test rax rax
jz instr_addr_15
add rax CONST
```

## Model Architecture

- **Base**: BERT-base (12 layers, 768 hidden, 12 heads)
- **Position Embeddings**: Standard BERT absolute (0-511)
- **Type Embeddings**: Standard (vocab_size = 2)
- **Tasks**: MLM + JTP (instruction-level)

## Performance Expectations

Expected to perform similarly to baseline jTrans, with potential advantages:
- More interpretable jump targets (instruction-level vs token-level)
- Potentially better generalization across different tokenization schemes
- Standard BERT embeddings may transfer better to other tasks

## Notes

- Vocabulary size includes `instr_addr_0` through `instr_addr_511` tokens
- Maximum 512 instructions per function supported
- Compatible with standard BERT pretrained models for transfer learning
