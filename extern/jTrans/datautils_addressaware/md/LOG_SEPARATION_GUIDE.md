# Log File Separation Guide

## What Changed

The `process.py` script has been modified to save log files in a separate directory from the data files.

### File Outputs:

**Before:**
```
/data/kun/jtransdata/addr_extract/
├── binary_name_extract.pkl          # Pickle file
├── binary_name_addressaware.txt     # Text file
└── binary_name_ida.log              # Log file (mixed with data)
```

**After:**
```
/data/kun/jtransdata/addr_extract/
├── binary_name_extract.pkl          # Pickle file
├── binary_name_addressaware.txt     # Text file
└── logs/
    └── binary_name_ida.log          # Log file (in separate folder)
```

## Configuration

### Default Behavior
By default, logs are now saved to `<SAVEROOT>/logs/` directory.

### Custom Log Directory
You can specify a custom log directory using the `LOGROOT` environment variable:

```bash
# Option 1: Save logs to a completely separate directory
export LOGROOT=/data/kun/jtransdata/logs
idat -A -S"process.py" /path/to/binary

# Option 2: Save logs to a subdirectory of SAVEROOT (default)
export SAVEROOT=/data/kun/jtransdata/addr_extract
# Logs will automatically go to /data/kun/jtransdata/addr_extract/logs/

# Option 3: Custom location
export SAVEROOT=/data/kun/jtransdata/addr_extract
export LOGROOT=/home/kun/ida_processing_logs
idat -A -S"process.py" /path/to/binary
```

## Batch Processing Script

If you're using a batch processing script, update it like this:

```bash
#!/bin/bash

# Set directories
export DATAROOT=/data/kun/jtransdata/binaries
export SAVEROOT=/data/kun/jtransdata/addr_extract
export LOGROOT=/data/kun/jtransdata/logs  # Optional: specify custom log directory

# Process binaries
for binary in /path/to/binaries/*.strip; do
    echo "Processing: $binary"
    idat -A -S"process.py" "$binary"
done

echo "Data files saved to: $SAVEROOT"
echo "Log files saved to: $LOGROOT"
```

## Moving Existing Log Files

If you want to organize existing log files, you can run:

```bash
# Create logs directory
mkdir -p /data/kun/jtransdata/addr_extract/logs

# Move all log files
mv /data/kun/jtransdata/addr_extract/*_ida.log /data/kun/jtransdata/addr_extract/logs/

# Verify
ls -lh /data/kun/jtransdata/addr_extract/logs/
```

## File Structure Summary

### 1. **`*_extract.pkl`** (Pickle File)
- **Purpose**: Structured function metadata for ML models
- **Format**: Python pickle (binary)
- **Size**: Usually 10KB - 1MB depending on binary size
- **Contains**: Functions, assembly, CFG, raw bytes, features

### 2. **`*_addressaware.txt`** (Text File)  
- **Purpose**: Address-aware tokenized format for pretraining
- **Format**: Text file, one function per line
- **Size**: Usually 50KB - 5MB depending on binary size
- **Contains**: Hierarchical position tokens + instruction tokens

### 3. **`*_ida.log`** (Log File) → **Now in `logs/` subdirectory**
- **Purpose**: Processing logs and debugging info
- **Format**: Plain text log
- **Size**: Usually 1KB - 100KB
- **Contains**: Status messages, function counts, errors

## Benefits of Separate Log Directory

1. **Cleaner Data Directory**: Keep only actual data files (pkl/txt) in main directory
2. **Easier Management**: Can delete/archive all logs at once
3. **Better Organization**: Logs don't clutter data listings
4. **Disk Space**: Easy to identify and clean up log space
5. **Debugging**: All logs in one place for troubleshooting

## Quick Commands

```bash
# Check data files only (no logs)
ls /data/kun/jtransdata/addr_extract/*.pkl
ls /data/kun/jtransdata/addr_extract/*.txt

# Check log files
ls /data/kun/jtransdata/addr_extract/logs/*.log

# Count files
echo "PKL files: $(ls /data/kun/jtransdata/addr_extract/*.pkl | wc -l)"
echo "TXT files: $(ls /data/kun/jtransdata/addr_extract/*.txt | wc -l)"
echo "LOG files: $(ls /data/kun/jtransdata/addr_extract/logs/*.log | wc -l)"

# Clean up old logs (be careful!)
rm /data/kun/jtransdata/addr_extract/logs/*_ida.log

# Archive logs by date
tar -czf logs_$(date +%Y%m%d).tar.gz /data/kun/jtransdata/addr_extract/logs/
```
