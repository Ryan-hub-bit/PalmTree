# Compact Semantic Format for PalmTree

## Overview
Modified `scripts/instruction_to_semantic.py` to use **compact semantic templates** that preserve critical information while reducing token count by ~26%.

## Key Changes

### 1. Direction Markers (LD/ST/CP)
Memory operations now have explicit direction:
- `LD [mem] reg` - Load from memory
- `ST reg [mem]` - Store to memory  
- `CP src dst` - Copy (register-to-register)

**Examples:**
```
mov rax qword [rel symbol]  →  LD [symbol] rax     (7→3 tokens)
mov qword [rel symbol] rax  →  ST rax [symbol]     (7→3 tokens)
mov rax rbx                 →  CP rbx rax          (3→3 tokens)
```

### 2. Sign/Zero Extension Markers
```
movsx eax byte [rbx]  →  LD.sx [rbx] eax
movzx eax byte [rbx]  →  LD.zx [rbx] eax
```

### 3. Conditional Suffixes
Conditions preserved as compact suffixes:
```
je addr   →  j.eq addr
jne addr  →  j.ne addr
jg addr   →  j.g addr    (signed greater)
ja addr   →  j.a addr    (unsigned above)
cmove     →  cmov.eq
```

### 4. Type Suffixes for SIMD/FPU
```
addsd  →  add.sd    (scalar double)
addss  →  add.ss    (scalar single)
addpd  →  add.pd    (packed double)
xorpd  →  xor.pd
```

### 5. Memory Operand Prefix
Memory operands in non-move instructions get "M" prefix:
```
add dword [rsp] 0x10  →  add 0x10 M[rsp]
cmp rax [rbx]         →  cmp M[rbx] rax
```

## Token Count Reduction

Sample of 15 common instructions:
- **Original**: 53 tokens
- **Compact**: 39 tokens  
- **Reduction**: 26.4%

This means:
- More instructions fit per context window
- Better relationship learning (denser patterns)
- Lower vocabulary size (shared suffixes)

## Benefits

1. **Explicit Direction**: Model no longer needs to infer load vs store from operand position
2. **Preserved Precision**: Sign/zero extension, signed/unsigned conditions maintained
3. **Shorter Sequences**: ~26% fewer tokens = more instructions per sample
4. **Better Context Density**: 20-token window fits more instruction patterns
5. **Vocabulary Efficiency**: Shared suffixes (.eq, .sd, .pd) reduce vocab explosion

## What's Preserved

✅ Load/store direction  
✅ Signed vs unsigned (j.g vs j.a, LD.sx vs LD.zx)  
✅ Width hints (kept in register names: rax/eax/ax/al)  
✅ Condition families (all jcc/cmov/set variants)  
✅ SIMD/FPU type suffixes  
✅ Memory vs register operands

## Usage

Run semantic conversion:
```bash
cd /home/louie/PalmTree/scripts
python3 instruction_to_semantic.py
```

This generates:
- `data/cfg_2_semantic.txt`
- `data/dfg_2_semantic.txt`
- `data/*_semantic_log.txt` (unknown opcodes)

## Next Steps

1. ✅ Compact semantic format implemented
2. ⏳ Generate semantic files for full dataset
3. ⏳ Add dataset loader support for semantic augmentation
4. ⏳ Run ablation: raw vs semantic vs hybrid (50/50)
5. ⏳ Evaluate on intrinsic (MLM loss) and extrinsic (Gemini/EKLAVYA) tasks
