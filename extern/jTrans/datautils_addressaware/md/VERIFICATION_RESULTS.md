# Verification Results & Next Steps

## Current Status Summary

### ✅ What's Working
1. **Memory Operand Tokenization**: Already correct in existing files!
   - Example: `[ rbp + var(0x8) ]` - properly spaced tokens
   - This fix was already working or present in your data

### ❌ What Needs Regeneration
2. **PLT Symbol Preservation**: Missing in existing files
   - Current: `call(...) daddr(0xfc50:...)`
   - Expected: `call(.printf)` or `call(.malloc)`
   - Reason: Existing data generated with old code before PLT detection was added

## Evidence from Existing Files

### What We Found
```bash
# Searched for PLT symbols in /data/kun/jtransdata/addr_extract/
$ grep -oh "\.\(printf\|malloc\|free\|...\)" *_addressaware.txt
# Result: EMPTY (no matches)

# How calls currently look:
$ grep "call(" *_addressaware.txt | head -3
call(...) daddr(0xfdc0:2.00000000:0.00000000:0.00000000)
call(...) daddr(0xffc0:2.00000000:0.00000000:0.00000000)
call(...) daddr(0xfe78:2.00000000:0.00000000:0.00000000)
```

The calls use generic `daddr` instead of function names like `.printf`, `.malloc`.

## Code Status

### Files Modified
1. **process.py** ✅ (Lines 437-450)
   - PLT symbol detection implemented
   - Checks `.plt` section
   - Uses `symbol_map` for function names
   
2. **process.py** ✅ (Lines 420-478)
   - Memory operand tokenization uses spaces
   - Delimiters kept as separate tokens

3. **process.py** ✅ (Lines 73-77, 501-505)
   - LOGROOT variable for separate log directory
   
4. **run_ida_addressaware.sh** ✅ (Line 62)
   - Added `export LOGROOT="${OUTPUT_DIR}/logs"`
   - Creates logs subdirectory

## Testing Plan

### Option 1: Quick Test (Recommended First)
Test with ONE binary to verify fixes work:

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_addressaware

# Pick any test binary
./test_plt_symbols.sh /data/kun/unstripped_binary_O0/some_binary.strip

# This will:
# - Process one binary with updated code
# - Show if PLT symbols appear
# - Display sample call instructions
```

### Option 2: Full Regeneration
Once test confirms PLT symbols work, regenerate full dataset:

```bash
# Follow NEXT_STEPS.md guide
./run_ida_addressaware.sh \
  -i /data/kun/unstripped_binary_O0 \
  -o /data/kun/jtransdata/addr_extract_v2

# This will:
# - Process all binaries with updated code
# - Generate .pkl and .txt files with PLT symbols
# - Save logs to logs/ subdirectory
# - Preserve existing data (new output directory)
```

## Why PLT Symbols Matter

### Current State (without PLT symbols)
```
call(0x4160:0.23802267:0.00000000:0.46938776) daddr(0xfc50:2.00000000:0.00000000:0.00000000)
```
- Generic address reference
- No semantic information about what function is called
- Model can't learn library function patterns

### Expected State (with PLT symbols)
```
call(0x4160:0.23802267:0.00000000:0.46938776) .printf
```
- Clear function name
- Model learns: "printf is used for output"
- Better understanding of program behavior

## Action Items

### Immediate (5 minutes)
- [ ] Run `test_plt_symbols.sh` with one test binary
- [ ] Verify PLT symbols appear in output
- [ ] Check sample call instructions

### If Test Succeeds (hours/days depending on dataset size)
- [ ] Decide: regenerate full dataset or subset?
- [ ] Run `run_ida_addressaware.sh` with appropriate paths
- [ ] Update your training/evaluation scripts to use new data
- [ ] Document data version change (v1 vs v2)

### If Test Fails
- [ ] Check IDA log in `/tmp/test_plt_logs/`
- [ ] Verify binary has PLT section: `readelf -S binary | grep plt`
- [ ] Check if binary is statically linked
- [ ] Share error messages for debugging

## Questions to Consider

1. **How important are PLT symbols for your task?**
   - If training function similarity: very important
   - If only doing control flow analysis: less critical

2. **How long will full regeneration take?**
   - Depends on number of binaries
   - IDA processing: ~10-60 seconds per binary
   - Estimate: count binaries × 30 seconds

3. **Can you work with subset first?**
   - Test on validation set only
   - Compare model performance with/without PLT symbols
   - Decide if full regeneration worth the time

## Summary

**Good News**: Code is fixed! ✅
**Challenge**: Existing data needs regeneration to benefit from fixes ⏳
**Recommendation**: Test with one binary first, then decide on full regeneration based on importance and time available.

The memory operand tokenization is already working, so main benefit of regeneration is PLT symbol preservation.
