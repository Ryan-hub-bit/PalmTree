# Next Steps: Applying the Fixes

## Quick Start (for immediate testing)

```bash
# 1. Navigate to the directory
cd /home/kun/Document/AAE/extern/jTrans/datautils_addressaware

# 2. Set environment variables
export SAVEROOT=/data/kun/jtransdata/addr_extract_new  # Use new directory to compare
export DATAROOT=/data/kun/jtransdata/binaries
export LOGROOT=/data/kun/jtransdata/logs_new

# 3. Test with one binary
idat -A -S"process.py" /path/to/test_binary.strip

# 4. Check the output
head -20 /data/kun/jtransdata/addr_extract_new/test_binary_addressaware.txt

# 5. Look for improvements
# Should see: [ rbp + var(0x8) ] with spaces
# Should see: .printf .malloc .free symbols
grep "\[ .* + .* \]" /data/kun/jtransdata/addr_extract_new/test_binary_addressaware.txt | head -5
grep "\.printf\|\.malloc\|\.free" /data/kun/jtransdata/addr_extract_new/test_binary_addressaware.txt | head -5
```

---

## Option 1: Regenerate All Data (Recommended)

### Advantages:
- ✅ Consistent tokenization across entire dataset
- ✅ All PLT symbols properly captured
- ✅ Clean log organization

### Steps:

```bash
#!/bin/bash
# regenerate_all.sh

# Configuration
export SAVEROOT=/data/kun/jtransdata/addr_extract_v2  # New version
export DATAROOT=/data/kun/jtransdata/binaries
export LOGROOT=/data/kun/jtransdata/logs_v2

# Create directories
mkdir -p $SAVEROOT
mkdir -p $LOGROOT

# Count total binaries
TOTAL=$(find $DATAROOT -name "*.strip" | wc -l)
echo "Found $TOTAL binaries to process"

# Process all binaries
COUNTER=0
for binary in $(find $DATAROOT -name "*.strip"); do
    COUNTER=$((COUNTER + 1))
    echo "[$COUNTER/$TOTAL] Processing: $(basename $binary)"
    
    # Run IDA Pro
    idat -A -S"process.py" "$binary"
    
    # Check for errors
    BINARY_NAME=$(basename "$binary" .strip)
    if [ -f "$LOGROOT/${BINARY_NAME}_ida.log" ]; then
        if grep -q "ERROR" "$LOGROOT/${BINARY_NAME}_ida.log"; then
            echo "  ⚠️  ERRORS detected in log!"
        else
            echo "  ✓ Success"
        fi
    else
        echo "  ⚠️  No log file generated!"
    fi
done

echo ""
echo "Processing complete!"
echo "Data files: $SAVEROOT"
echo "Log files: $LOGROOT"
echo ""
echo "Summary:"
echo "  PKL files: $(find $SAVEROOT -name "*.pkl" | wc -l)"
echo "  TXT files: $(find $SAVEROOT -name "*.txt" | wc -l)"
echo "  LOG files: $(find $LOGROOT -name "*.log" | wc -l)"
```

**Run it:**
```bash
chmod +x regenerate_all.sh
./regenerate_all.sh
```

---

## Option 2: Keep Old Data, Organize Logs Only

### If you want to keep existing data but just organize logs:

```bash
# Run the organize script
cd /home/kun/Document/AAE/extern/jTrans/datautils_addressaware
./organize_logs.sh /data/kun/jtransdata/addr_extract

# This will:
# - Create logs/ subdirectory
# - Move all *_ida.log files to logs/
# - Show summary statistics
```

**Note:** This only organizes logs - it doesn't fix tokenization or add PLT symbols to existing files.

---

## Option 3: Incremental Update (Mixed Approach)

### For selective regeneration:

```bash
#!/bin/bash
# regenerate_problematic.sh

export SAVEROOT=/data/kun/jtransdata/addr_extract
export DATAROOT=/data/kun/jtransdata/binaries
export LOGROOT=/data/kun/jtransdata/logs

# Find binaries with specific patterns that need PLT symbols
# (e.g., binaries that call many library functions)
BINARIES_TO_REPROCESS=(
    "2bwm-git-2bwm-O0-f9579e061f6e200bc50fdae0d8f2a873"
    "9base-bc-O0-50b1dfb548c183eee821cd27b2ac6b88"
    # Add more as needed
)

for basename in "${BINARIES_TO_REPROCESS[@]}"; do
    binary="$DATAROOT/${basename}.strip"
    if [ -f "$binary" ]; then
        echo "Reprocessing: $basename"
        
        # Backup old files
        mv "$SAVEROOT/${basename}_extract.pkl" "$SAVEROOT/${basename}_extract.pkl.bak" 2>/dev/null
        mv "$SAVEROOT/${basename}_addressaware.txt" "$SAVEROOT/${basename}_addressaware.txt.bak" 2>/dev/null
        
        # Regenerate
        idat -A -S"process.py" "$binary"
        
        echo "  ✓ Done"
    else
        echo "  ⚠️  Binary not found: $binary"
    fi
done

echo "Incremental update complete!"
```

---

## Validation & Quality Checks

After regenerating, run these checks:

### 1. Tokenization Check
```bash
# Should show lines with spaced brackets
head -100 /data/kun/jtransdata/addr_extract_v2/*_addressaware.txt | grep "\[ .* \]" | head -10

# Example expected output:
# mov(0x1234:...) rax [ rbp + var(0x8) ]
# lea(0x1240:...) rdi [ rax + rbx * 4 ]
```

### 2. PLT Symbol Check
```bash
# Should find PLT symbols
grep -h "\.printf\|\.malloc\|\.free\|\.calloc\|\.memcpy" /data/kun/jtransdata/addr_extract_v2/*_addressaware.txt | head -20

# Example expected output:
# call(0x4010:...) .printf
# call(0x4020:...) .malloc
```

### 3. File Count Check
```bash
PKL_COUNT=$(find /data/kun/jtransdata/addr_extract_v2 -name "*.pkl" | wc -l)
TXT_COUNT=$(find /data/kun/jtransdata/addr_extract_v2 -name "*.txt" | wc -l)
LOG_COUNT=$(find /data/kun/jtransdata/logs_v2 -name "*.log" | wc -l)

echo "PKL files: $PKL_COUNT"
echo "TXT files: $TXT_COUNT"
echo "LOG files: $LOG_COUNT"

# All three should be equal
if [ "$PKL_COUNT" = "$TXT_COUNT" ] && [ "$TXT_COUNT" = "$LOG_COUNT" ]; then
    echo "✓ File counts match!"
else
    echo "⚠️  File count mismatch - some processing may have failed"
fi
```

### 4. Error Check
```bash
# Check for errors in logs
ERROR_LOGS=$(grep -l "ERROR" /data/kun/jtransdata/logs_v2/*_ida.log)
ERROR_COUNT=$(echo "$ERROR_LOGS" | grep -c "ida.log" || echo 0)

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  Found $ERROR_COUNT logs with errors:"
    echo "$ERROR_LOGS"
else
    echo "✓ No errors in logs!"
fi
```

### 5. Comparison Check (if keeping old data)
```bash
# Compare file sizes (new should be similar or slightly larger due to spaces)
OLD_SIZE=$(du -sh /data/kun/jtransdata/addr_extract/*.txt | awk '{sum+=$1} END {print sum}')
NEW_SIZE=$(du -sh /data/kun/jtransdata/addr_extract_v2/*.txt | awk '{sum+=$1} END {print sum}')

echo "Old total size: $OLD_SIZE"
echo "New total size: $NEW_SIZE"
echo "(New should be slightly larger due to added spaces)"
```

---

## Integration with Training Pipeline

### Update your data loading code:

```python
# Before
data_dir = "/data/kun/jtransdata/addr_extract"

# After
data_dir = "/data/kun/jtransdata/addr_extract_v2"  # Use new version
log_dir = "/data/kun/jtransdata/logs_v2"            # Separate logs
```

### Update tokenization if needed:

```python
# The new format uses spaces, so basic whitespace tokenization should work
tokens = instruction_text.split()

# Example:
# Old: "mov rax [rbp+var(0x8)]" → ["mov", "rax", "[rbp+var(0x8)]"]
# New: "mov rax [ rbp + var(0x8) ]" → ["mov", "rax", "[", "rbp", "+", "var(0x8)", "]"]
```

---

## Rollback Plan (if needed)

If something goes wrong:

```bash
# Option 1: Keep old data untouched by using new directory
# (Recommended approach above with addr_extract_v2)

# Option 2: If you overwrote old data, restore from backup
mv /data/kun/jtransdata/addr_extract_v2/* /data/kun/jtransdata/addr_extract/

# Option 3: If you have backups with .bak extension
for file in /data/kun/jtransdata/addr_extract/*.bak; do
    mv "$file" "${file%.bak}"
done
```

---

## Troubleshooting

### Issue: No PLT symbols appearing
**Cause:** Binary may not have .plt section, or symbols not in symbol_map
**Solution:** Check binary with `readelf -s binary | grep FUNC`

### Issue: Tokenization still looks wrong
**Cause:** Old version of process.py might be cached
**Solution:** 
```bash
# Force reload
rm -rf /tmp/ida_*
# Re-run IDA
```

### Issue: Logs not going to LOGROOT
**Cause:** Environment variable not set
**Solution:**
```bash
export LOGROOT=/data/kun/jtransdata/logs
echo $LOGROOT  # Verify it's set
```

### Issue: Processing very slow
**Cause:** IDA Pro processing is CPU intensive
**Solution:**
```bash
# Run multiple instances in parallel (use with caution)
ls /data/kun/jtransdata/binaries/*.strip | parallel -j 4 'idat -A -S"process.py" {}'
```

---

## Timeline Estimate

- **Single binary:** ~1-5 minutes (depending on size)
- **100 binaries:** ~2-8 hours
- **1000 binaries:** ~20-80 hours (run overnight or over weekend)
- **10000 binaries:** Several days (consider parallel processing)

---

## Final Checklist

Before considering the update complete:

- [ ] Tested on sample binary successfully
- [ ] Verified tokenization with spaces: `[ rbp + var(0x8) ]`
- [ ] Verified PLT symbols appear: `.printf`, `.malloc`
- [ ] Logs organized in separate directory
- [ ] File counts match (PKL = TXT = LOG)
- [ ] No errors in log files
- [ ] Old data backed up or new data in separate directory
- [ ] Training pipeline updated to use new data path
- [ ] Documentation updated with new format
- [ ] Team notified of format changes

---

## Questions?

Check these resources:
1. `VISUAL_COMPARISON.md` - See before/after examples
2. `CHANGES_SUMMARY.md` - Detailed technical changes  
3. `test_tokenization.py` - Test script for demonstrations
4. Log files in `$LOGROOT/` - Detailed processing logs

Good luck with the regeneration! 🚀
