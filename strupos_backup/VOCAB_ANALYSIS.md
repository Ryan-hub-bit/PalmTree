# Vocabulary Analysis Summary

## Your Vocabulary Statistics

**Actual Vocabulary Size: 1,214 tokens**

### Special Tokens (5 total)
- `<pad>` → ID: 0 (padding)
- `<unk>` → ID: 1 (unknown tokens)
- `<eos>` → ID: 2 (end of sequence)
- `<sos>` → ID: 3 (start of sequence)  
- `<mask>` → ID: 4 (for MLM masking)

### Top 20 Most Frequent Tokens

| Rank | Token | ID | Frequency |
|------|-------|-----|-----------|
| 1 | mov | 5 | 12,891,775 |
| 2 | + | 6 | 12,301,692 |
| 3 | [ | 7 | 12,212,040 |
| 4 | ] | 8 | 12,212,040 |
| 5 | rbp | 9 | 11,002,426 |
| 6 | var | 10 | 9,457,774 |
| 7 | address | 11 | 8,117,925 |
| 8 | rax | 12 | 6,529,581 |
| 9 | imm | 13 | 5,740,326 |
| 10 | rdi | 14 | 2,950,208 |
| 11 | eax | 15 | 2,836,213 |
| 12 | rsp | 16 | 2,670,870 |
| 13 | call | 17 | 2,209,537 |
| 14 | rcx | 18 | 1,894,715 |
| 15 | lea | 19 | 1,802,733 |
| 16 | cmp | 20 | 1,479,517 |
| 17 | jmp | 21 | 1,368,550 |
| 18 | rsi | 22 | 1,232,522 |
| 19 | add | 23 | 1,165,150 |
| 20 | al | 24 | 984,324 |

### Token Categories

Based on the inspection, your vocabulary contains:

- **Opcodes**: mov, call, jmp, lea, cmp, add, sub, xor, test, push, pop, etc.
- **Registers**: rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, eax, ebx, etc.
- **Operands**: +, -, *, [, ], ptr, byte, dword, qword
- **Special symbols**: address, var, imm, disp, arg, dest, s, s1
- **XMM registers**: xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6
- **Branch instructions**: jz, jnz, jge, jle, jb, jbe, ja, jnb, etc.

## The "126 tokens" Bug (FIXED)

### Problem:
The run script was using `wc -l` to count lines in a **binary pickle file**, which doesn't work:
```bash
VOCAB_SIZE=$(wc -l <"$VOCAB_FILE")  # Wrong! Pickle is binary
```

### Solution:
Now properly loads the pickle file and counts tokens:
```bash
VOCAB_SIZE=$(python3 -c "import pickle; vocab = pickle.load(open('$VOCAB_FILE', 'rb')); print(len(vocab))")
```

### Fixed Files:
- ✅ `run_consecutive.sh`
- ✅ `run_multi_to_one.sh`

## How to Inspect Your Vocabulary

### Quick Check:
```bash
python3 inspect_vocab.py --vocab ./vocab.pkl
```

### Show All Tokens:
```bash
python3 inspect_vocab.py --vocab ./vocab.pkl --all
```

### Save to File:
```bash
python3 inspect_vocab.py --vocab ./vocab.pkl --all --output vocab_full
# Creates: vocab_full.txt
```

## Vocabulary Quality

Your vocabulary looks **good** for binary code analysis:

✅ **Comprehensive opcode coverage**: All major x86-64 instructions  
✅ **Register coverage**: All general-purpose and XMM registers  
✅ **Address handling**: 'address', 'var', 'imm', 'disp' tokens  
✅ **Reasonable size**: 1,214 tokens (not too small, not too large)  
✅ **Frequency-based**: Most common tokens (mov, registers) have low IDs

## Next Steps

Now that the vocabulary is properly counted, your training script will correctly show:
```
Vocabulary size: 1,214 tokens
```

Instead of the incorrect:
```
Vocabulary size: 126 tokens
```
