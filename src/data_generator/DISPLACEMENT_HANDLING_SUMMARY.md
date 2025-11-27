# Displacement Handling Applied to Both CFG and DFG

## Summary

✅ **Successfully applied displacement handling to both CFG and DFG generation scripts**

## Files Updated

### 1. CFG Generation (IDA version)
**File**: `/src/data_generator/cfg_hierarchical_icfg_ida.py`

**Changes**:
- `clean_ida_disasm()` - Detects `o_displ` operands, marks as `disp_0xXX`
- `build_chunk_inline()` - Strips `disp_` prefix, outputs plain hex (no `address()` wrapper)

### 2. DFG Generation (IDA version)  
**File**: `/src/data_generator/dfg_hierarchical_idfg_ida.py`

**Changes**:
- `clean_ida_disasm()` - Detects `o_displ` operands, marks as `disp_0xXX`
- `build_chunk_inline()` - Strips `disp_` prefix, outputs plain hex (no `address()` wrapper)

## Logic Flow (Identical for Both CFG and DFG)

```
┌─────────────────────────────────────────────────────────────────┐
│ IDA Pro Instruction Analysis                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ clean_ida_disasm(ea)                                            │
│                                                                 │
│ for each operand:                                               │
│   op_type = idc.get_operand_type(ea, i)                        │
│   op_value = idc.get_operand_value(ea, i)                      │
│                                                                 │
│   if op_type == idc.o_imm:                                     │
│       → "imm"                                                  │
│                                                                 │
│   elif op_type == idc.o_displ:  ← DISPLACEMENT HANDLING        │
│       if not already formatted:                                 │
│           → f"disp_{hex(op_value)}"  ← SPECIAL MARKER          │
│       else:                                                     │
│           → keep as-is (e.g., "[rbp+0x20]")                    │
│                                                                 │
│   elif op_type in [idc.o_near, idc.o_mem, idc.o_far]:        │
│       if not already formatted:                                 │
│           → hex(op_value)                                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ build_chunk_inline(seq, start_idx, k, ctx)                     │
│                                                                 │
│ for each token in operands:                                     │
│                                                                 │
│   if tok.startswith('disp_0x'):  ← CHECK FOR DISPLACEMENT      │
│       hex_part = tok[5:]  # Remove "disp_" prefix              │
│       formatted_ops.append(hex_part)                           │
│       continue  ← SKIP address() wrapping                      │
│                                                                 │
│   elif tok.startswith('0x'):                                   │
│       tgt = int(tok, 16)                                       │
│       if tgt >= min_addr and not is_immediate:                 │
│           if tgt in addr_positions:                            │
│               → f"address({tok}:{func:bb:inst})"               │
│           else:                                                 │
│               → f"address({tok}:{sec:off:0.0})"                │
│       else:                                                     │
│           → "imm"                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Operand Type Handling

| Operand Type | IDA Type Constant | Handling | Output |
|--------------|-------------------|----------|--------|
| Immediate | `idc.o_imm` | Mark as "imm" | `imm` |
| **Displacement** | **`idc.o_displ`** | **Mark as "disp_0xXX"** | **Plain hex (e.g., `0x20`)** |
| Near call/jump | `idc.o_near` | Convert to "0xADDR" | `address(0xADDR:...)` |
| Far call/jump | `idc.o_far` | Convert to "0xADDR" | `address(0xADDR:...)` |
| Memory reference | `idc.o_mem` | Convert to "0xADDR" | `address(0xADDR:...)` |

## Examples

### CFG Example
```assembly
# Input instruction
mov rax, [rax + field_offset]

# IDA analysis: o_displ, value=0x20

# After clean_ida_disasm()
mov, rax, disp_0x20

# After build_chunk_inline()
mov(0xa000:0.25:0.18:0.13) rax [ rax + 0x20 ]
                                          ^^^^
                                          Plain hex, no address()!
```

### DFG Example (Data Flow)
```assembly
# Input instruction
lea rdi, [rbx + 0x10]

# IDA analysis: o_displ, value=0x10

# After clean_ida_disasm()
lea, rdi, disp_0x10

# After build_chunk_inline()
lea(0xb000:0.30:0.20:0.15) rdi [ rbx + 0x10 ]
                                          ^^^^
                                          Plain hex, no address()!
```

## Why This Matters

### For CFG (Control Flow Graph):
- Instructions reference struct fields via displacements
- Example: `mov rax, [rdi + 0x20]` accessing a struct member
- Should NOT wrap `0x20` as an address (it's just an offset)

### For DFG (Data Flow Graph):
- Data dependencies also use displacement addressing
- Example: `lea rax, [rbp + 0x10]` calculating an address
- Should NOT wrap `0x10` as an address (it's just an offset)

## Consistency Across CFG and DFG

Both scripts now handle operands identically:
- ✅ Displacements (`o_displ`) stay as plain hex
- ✅ Code addresses (`o_near`, `o_far`) get `address()` wrapper
- ✅ Data addresses (`o_mem`) get `address()` wrapper
- ✅ Immediates (`o_imm`) marked as `imm`

This ensures the model sees consistent tokenization in both:
- **CFG data**: Control flow sequences
- **DFG data**: Data dependency sequences

## Testing

Run the existing test scripts to verify:
```bash
cd /home/kun/Document/PalmTree/src/data_generator
python test_displacement.py
python test_pipeline.py
```

Both tests validate the logic works correctly for all operand types.

## Impact

When you regenerate your training data:
1. Small offsets (struct fields, stack variables) will be plain hex
2. Real addresses will have hierarchical position encoding
3. Model can learn the semantic difference between offsets and addresses
4. Both CFG and DFG will have consistent representation

✅ **Ready for data regeneration!**
