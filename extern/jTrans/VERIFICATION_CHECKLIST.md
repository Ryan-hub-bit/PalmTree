# Address-Aware Model Alignment Verification

## ✅ PRETRAIN vs FINETUNE vs EVALUATION - All Aligned

### Data Format
- [x] Tab-separated instructions (`\t`)
- [x] `<sos>` token at beginning with segment 1
- [x] NO `<eos>` after each instruction
- [x] ONE `<eos>` token at the end with segment 1
- [x] All instruction tokens in segment 1
- [x] Padding tokens in segment 0

### Var Offset Handling
- [x] Two's complement: `if offset_val > 0x7FFFFFFFFFFFFFFF:`
- [x] Conversion: `offset_val - 0x10000000000000000`
- [x] Non-var tokens: `-1`
- [x] Var tokens without positions: `(-1.0, -1.0, -1.0)`

### Token Parsing
- [x] `opcode(0xADDR:bnorm:fnorm:bbnorm)` → opcode with positions
- [x] `address(0xADDR:bnorm:fnorm:bbnorm)` → 'address' with positions
- [x] `daddr(0xADDR:bnorm:fnorm:bbnorm)` → 'daddr' with positions
- [x] `var(0xOFFSET)` → 'var' with offset value, no positions
- [x] Regular tokens → token with (-1, -1, -1) positions

### Files Modified
1. **data_json.py** (finetune): Fixed to match pretrain
2. **evaluate_addressaware_with_pools.py** (evaluation): Fixed to match pretrain

### Action Required
✅ **Rerun finetune** - Previous model was trained with wrong format
✅ **Then run evaluation** - With new correctly-finetuned model

---

Generated: 2026-01-15
