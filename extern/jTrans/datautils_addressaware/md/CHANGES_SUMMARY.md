# Summary of Changes to Address-Aware Data Generation

## Date: January 2, 2026

## Issues Fixed

### 1. Memory Operand Tokenization
**Problem:** Memory operands like `[rbp+var(0x8)]` were not properly tokenized - brackets, operators, and operands were concatenated without spaces, making it difficult for models to parse.

**Example of the issue:**
```
Before: mov rax [rbp+var(0x8)]
        ^^^ Hard to tokenize as individual symbols
```

**Solution:** Modified `process.py` line ~426-478 to:
- Keep delimiters `[`, `]`, `+`, `-`, `*` as separate tokens
- Add spaces between all tokens
- Change from `''.join(formatted_parts)` to `' '.join(formatted_parts)`

**Result:**
```
After:  mov rax [ rbp + var(0x8) ]
        ^^^ Each symbol is now separate and easy to tokenize
```

**Benefits:**
- Easier for transformer models to learn positional relationships
- Consistent with standard assembly tokenization practices
- Improves model's ability to understand memory access patterns

---

### 2. PLT Symbol Preservation
**Problem:** PLT (Procedure Linkage Table) symbols like `.printf`, `.malloc`, `.free` were not being preserved in the output. Instead, they were shown as generic address references.

**Example of the issue:**
```
Before: call daddr(0x1030:2.00000000:0.00000000:0.00000000)
        ^^^ Lost information about which library function is being called
```

**Solution:** Enhanced symbol detection in `process.py` to:
- Check if address is in `.plt` section
- Look up symbol name in symbol_map
- Preserve PLT symbol names (e.g., `.printf`, `.malloc`)
- Follows the approach from `cfg_hierarchical_icfg_ida.py`

**Result:**
```
After:  call .printf
After:  call .malloc
        ^^^ Clear indication of which library function is called
```

**Benefits:**
- Preserves semantic information about library function calls
- Helps model learn function call patterns
- Aligns with jTrans baseline format expectations

---

### 3. Log File Organization
**Problem:** Log files `*_ida.log` were mixed with data files `*_extract.pkl` and `*_addressaware.txt` in the same directory, making it hard to manage outputs.

**Solution:** 
- Added `LOGROOT` environment variable (default: `$SAVEROOT/logs/`)
- Modified `process.py` lines ~73, ~501, ~656 to save logs to separate directory
- Created `organize_logs.sh` script to move existing logs

**Result:**
```
Before:
/data/kun/jtransdata/addr_extract/
├── binary_extract.pkl
├── binary_addressaware.txt
└── binary_ida.log  ← Mixed with data

After:
/data/kun/jtransdata/addr_extract/
├── binary_extract.pkl
├── binary_addressaware.txt
└── logs/
    └── binary_ida.log  ← Organized separately
```

**Benefits:**
- Cleaner data directory structure
- Easier to manage/delete logs independently
- Better disk space management
- Simplified data file listings

---

## Files Modified

### 1. `/home/kun/Document/AAE/extern/jTrans/datautils_addressaware/process.py`
**Changes:**
- Lines ~73-77: Added `LOGROOT` variable and logging
- Lines ~426-478: Fixed operand tokenization to keep delimiters separate
- Lines ~437-443: Added explicit delimiter handling
- Lines ~501-505: Changed log path to use `LOGROOT`
- Lines ~656-658: Updated error logging to use `LOGROOT`

### 2. New Files Created

#### `/home/kun/Document/AAE/extern/jTrans/datautils_addressaware/FILE_FORMATS.md`
- Documents the purpose and structure of `.pkl`, `.txt`, and `.log` files
- Explains use cases for each file type

#### `/home/kun/Document/AAE/extern/jTrans/datautils_addressaware/LOG_SEPARATION_GUIDE.md`
- Instructions for using the new log separation feature
- Examples of environment variable configuration
- Commands for organizing existing logs

#### `/home/kun/Document/AAE/extern/jTrans/datautils_addressaware/organize_logs.sh`
- Executable script to move existing `*_ida.log` files to `logs/` subdirectory
- Provides summary statistics

#### `/home/kun/Document/AAE/extern/jTrans/datautils_addressaware/test_tokenization.py`
- Demonstrates the tokenization improvements
- Shows before/after comparisons
- Explains the changes

---

## Testing & Verification

### To test the changes:
```bash
# 1. Set environment variables
export SAVEROOT=/data/kun/jtransdata/addr_extract
export DATAROOT=/data/kun/jtransdata/binaries  
export LOGROOT=/data/kun/jtransdata/logs

# 2. Process a test binary
idat -A -S"process.py" /path/to/test_binary.strip

# 3. Check output format
head -5 /data/kun/jtransdata/addr_extract/test_binary_addressaware.txt

# 4. Verify log location
ls -lh /data/kun/jtransdata/logs/test_binary_ida.log

# 5. Look for PLT symbols
grep "\.printf\|\.malloc\|\.free" /data/kun/jtransdata/addr_extract/test_binary_addressaware.txt
```

### Expected changes in output:
1. **Memory operands:** `mov rax [ rbp + var(0x8) ]` instead of `mov rax [rbp+var(0x8)]`
2. **PLT symbols:** `call .printf` instead of `call daddr(0x1030:...)`
3. **Log location:** Files in `/data/kun/jtransdata/logs/` not mixed with data

---

## Organizing Existing Files

To organize your existing log files:
```bash
# Run the organize script
cd /home/kun/Document/AAE/extern/jTrans/datautils_addressaware
./organize_logs.sh /data/kun/jtransdata/addr_extract

# Or manually:
mkdir -p /data/kun/jtransdata/addr_extract/logs
mv /data/kun/jtransdata/addr_extract/*_ida.log /data/kun/jtransdata/addr_extract/logs/
```

---

## Backward Compatibility

✅ **Fully backward compatible:**
- Existing `.pkl` and `.txt` files remain valid
- Default `LOGROOT=$SAVEROOT/logs/` maintains output organization
- No changes to pickle format or data structure
- Scripts that only use `.pkl` or `.txt` files are unaffected

⚠️ **If you have scripts that depend on log file location:**
- Update paths from `$SAVEROOT/*_ida.log` to `$LOGROOT/*_ida.log`
- Or set `export LOGROOT=$SAVEROOT` to restore old behavior

---

## Performance Impact

- **Tokenization:** Minimal - adds space joining instead of concatenation
- **PLT lookup:** Minimal - uses existing symbol_map dictionary
- **Log writing:** None - just changes output directory

---

## Future Considerations

### Potential improvements:
1. **Vocabulary consistency:** Ensure tokenization matches pretraining vocabulary
2. **Symbol filtering:** Implement top-100 symbol filtering (as mentioned in comments)
3. **Memory operand normalization:** Consider standardizing `var(0x8)` format
4. **Batch processing:** Create parallel processing scripts for large datasets

### Monitoring:
- Check model performance on memory-intensive code
- Verify library function call prediction accuracy
- Compare with baseline jTrans format results

---

## Questions & Support

For issues or questions:
1. Check the test script: `python test_tokenization.py`
2. Review log files in `$LOGROOT/` directory
3. Compare output format with examples in this document
4. Verify environment variables are set correctly

---

## References

- Original jTrans code: `extern/jTrans/datautils/process.py`
- CFG generation: `data_generator/cfg_hierarchical_icfg_ida.py`
- Symbol handling precedent: Top-100 symbol filtering in `combine_and_split.sh`
