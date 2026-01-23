# jTrans_instr Consistency Verification

This document verifies the consistency of instruction boundary detection across all phases (pretrain, finetune, evaluation).

## Data Format Standard

**Format**: Instructions separated by `\t` (tab), tokens within each instruction separated by space

```
Example:
mov rdi rsp\tpush rbp\txor eax eax\tjmp instr_addr_5
```

## Phase-by-Phase Verification

### 1. Pretrain Data Generation

**File**: `/extern/jTrans_instr/datautils/convert_pkl_to_text.py`

**Method**: `convert_function_to_text_with_jumps()`

**Implementation**:
```python
instruction_strings = []
for i, asm_str in enumerate(asm_list):
    tokens = [operator, op1, op2, ...]
    instruction_strings.append(' '.join(tokens))  # Space within instruction
return '\t'.join(instruction_strings)  # Tab between instructions
```

**Output**: `/data/kun/jtrans_instr/instr_pretrain.txt`

**Status**: ✅ Uses `\t` separator

---

### 2. Pretrain Data Loading

**File**: `/extern/jTrans_instr/pretrain/dataloader_instr.py`

**Method**: `_parse_instruction_boundaries()`

**Implementation**:
```python
def _parse_instruction_boundaries(self, func_str):
    instructions = func_str.split('\t')  # Split by tab
    
    for instr in instructions:
        instr_tokens = instr.split()  # Split by space
        # Process tokens...
```

**Input**: Lines from `instr_pretrain.txt`

**Status**: ✅ Uses `\t` separator

---

### 3. Finetune Data Generation

**File**: `/extern/jTrans_instr/datautils/create_instr_dataset.py`

**Method**: `tokenize_function_instr()`

**Implementation**:
```python
def tokenize_function_instr(func_data):
    for asm_str in asm_list:
        # Process instruction...
        instruction_tokens.append(' '.join(tokens))  # Space within instruction
    
    return '\t'.join(instruction_tokens)  # Tab between instructions
```

**Output**: `/data/kun/jtrans_instr/func_blocks_instr.json`

**Status**: ✅ Uses `\t` separator

---

### 4. Finetune Data Loading

**File**: `/extern/jTrans_instr/data_json_instr.py`

**Method**: `_tokenize_with_instruction_ids()`

**Implementation**:
```python
def _tokenize_with_instruction_ids(self, func_str):
    # Use pretrain tokenizer
    encoding = self.tokenizer(func_str, ...)
    
    # Generate instruction_ids by counting \t
    instructions = func_str.split('\t')  # Split by tab
    
    for instr in instructions:
        instr_enc = self.tokenizer(instr, add_special_tokens=False)
        instruction_token_counts.append(len(instr_enc['input_ids']))
```

**Input**: JSON from `func_blocks_instr.json`

**Status**: ✅ Uses `\t` separator

---

### 5. Evaluation Data Loading

**File**: `/extern/jTrans_instr/evaluate_instr_with_pools.py`

**Method**: `generate_instruction_ids()`

**Implementation**:
```python
def generate_instruction_ids(tokenizer, func_str, maxlen=512):
    # Tokenize using pretrain tokenizer
    encoding = tokenizer(func_str, ...)
    
    # Count tokens per instruction
    instructions = func_str.split('\t')  # Split by tab
    
    for instr in instructions:
        instr_enc = tokenizer(instr, add_special_tokens=False)
        instruction_token_counts.append(len(instr_enc['input_ids']))
```

**Input**: JSON from `func_blocks_instr.json` or pool files

**Status**: ✅ Uses `\t` separator

---

## Tokenization Strategy

All phases use the **same tokenizer** trained during pretrain:

1. **Pretrain**: Trains tokenizer on `instr_pretrain.txt`
2. **Finetune**: Reuses pretrain tokenizer + generates `instruction_ids`
3. **Evaluation**: Reuses pretrain tokenizer + generates `instruction_ids`

**Key Point**: Finetune and evaluation do NOT re-tokenize. They:
- Use pretrain's tokenizer for `input_ids`, `attention_mask`
- Only generate `instruction_ids` tensor by counting tokens per instruction

## Consistency Summary

| Phase | File | Method | Uses `\t`? | Tokenizer |
|-------|------|--------|-----------|-----------|
| Pretrain Data Gen | convert_pkl_to_text.py | convert_function_to_text_with_jumps | ✅ | N/A |
| Pretrain Loading | dataloader_instr.py | _parse_instruction_boundaries | ✅ | Pretrain tokenizer |
| Finetune Data Gen | create_instr_dataset.py | tokenize_function_instr | ✅ | N/A |
| Finetune Loading | data_json_instr.py | _tokenize_with_instruction_ids | ✅ | **Pretrain tokenizer** |
| Evaluation | evaluate_instr_with_pools.py | generate_instruction_ids | ✅ | **Pretrain tokenizer** |

## Verification Tests

To verify consistency, run these tests:

### Test 1: Check Data Format
```bash
# Check pretrain data
head -1 /data/kun/jtrans_instr/instr_pretrain.txt | tr '\t' '\n'

# Check finetune data
python3 -c "
import json
with open('/data/kun/jtrans_instr/func_blocks_instr.json') as f:
    data = json.load(f)
    func_id = list(data.keys())[0]
    print(data[func_id]['instructions'].replace('\t', '\n'))
"
```

Both should show one instruction per line.

### Test 2: Verify instruction_ids Generation
```bash
cd /home/kun/Document/AAE/extern/jTrans_instr

python3 -c "
from transformers import BertTokenizer
from data_json_instr import FunctionDataset_CL_Load_JSON_Instr

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('./jtrans_tokenizer')

# Test function string
func_str = 'push rbp\tmov rdi rsp\txor eax eax\tjmp instr_addr_5'

# Create dataset (minimal)
dataset = FunctionDataset_CL_Load_JSON_Instr([[func_str]], tokenizer)

# Get instruction_ids
input_ids, _, _, instruction_ids = dataset._tokenize_with_instruction_ids(func_str)

print('Function:', func_str)
print('Tokens:', tokenizer.convert_ids_to_tokens(input_ids.tolist()))
print('Instruction IDs:', instruction_ids.tolist()[:20])
"
```

Should show instruction indices increasing at instruction boundaries.

### Test 3: Compare Pretrain vs Finetune
```python
# pretrain/dataloader_instr.py
from pretrain.dataloader_instr import InstrPretrainingDataset

dataset = InstrPretrainingDataset(
    data_path='/data/kun/jtrans_instr/instr_pretrain.txt',
    tokenizer=tokenizer,
    on_memory=False
)

# Get one sample
sample = dataset[0]
print('Pretrain instruction_ids:', sample['instruction_ids'][:20])
```

Compare with finetune's instruction_ids - should follow same pattern.

## Conclusion

✅ **All phases are now consistent**:
- Use `\t` to separate instructions
- Use space to separate tokens within instructions  
- Reuse pretrain tokenizer in finetune/evaluation
- Generate instruction_ids by counting tokens per instruction (split by `\t`)

**No discrepancies found.**
