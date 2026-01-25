# Fixed DataUtils for jTrans Feature Extraction

## What Was Fixed

1. **Python Path Issue**: IDA Pro was unable to find `binaryai` and `networkx` packages
   - Added `/home/kun/anaconda3/lib/python3.12/site-packages` to Python path in `process_fixed.py`
   
2. **Missing TVHEADLESS**: IDA Pro failed to run in batch mode
   - Added `TVHEADLESS=1` environment variable in `run_fixed.py`

3. **Environment Variables**: Paths were hardcoded and inconsistent
   - Made `SAVEROOT` and `DATAROOT` configurable via environment variables
   - Both `run_fixed.py` and `process_fixed.py` now use the same paths

4. **Error Handling**: Made `binaryai` optional
   - If `binaryai` is not available, extraction continues with `None` for BinaryAI features
   - Added better error messages and logging

5. **IDA Path**: Using 32-bit `idat` instead of `idat64`

## Requirements

Make sure these packages are installed in your base anaconda environment:

```bash
/home/kun/anaconda3/bin/python3 -m pip install binaryai networkx pyelftools
```

**Note**: The `binaryai.ida` module may not be available, but extraction will work without it (BinaryAI features will be None).

## How to Use

### Option 1: Test with a Few Binaries First

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_fixed

# Edit run_fixed.py and uncomment the prefixfilter line to test with specific binaries
# prefixfilter = ['usbutils-git-lsusb']

python3 run_fixed.py
```

### Option 2: Run on Full Dataset

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_fixed
python3 run_fixed.py
```

### Option 3: Test Single Binary

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_fixed

# Set environment variables
export SAVEROOT=/data/kun/jtransdata/extract
export DATAROOT=/data/kun/jtransdata/mini_train
export TVHEADLESS=1

# Strip a binary
strip -s /data/kun/jtransdata/mini_train/usbutils-git-lsusb-O1-de3c13729e00ddf9ed087671ef3aa783 \
      -o /tmp/test.strip

# Run IDA on it
./ida-pro-9.0/idat -Llog/test.log -c -A \
    -S$(pwd)/process_fixed.py \
    -oidb/test.idb \
    /tmp/test.strip

# Check results
ls -lh /data/kun/jtransdata/extract/*.pkl
cat log/test.log | tail -50
```

## Output

- **Pickle files**: `/data/kun/jtransdata/extract/*_extract.pkl`
- **Log files**: `./log/*.log`
- **IDB files**: `./idb/*.idb`
- **Stripped binaries**: `/data/kun/jtransdata/mini_train_strip/*.strip`

## Expected Behavior

When running successfully, you should see:

```
✓ strip succeeded: <filename>
→ Starting IDA: <filename>
...
[*] Features Extracting Done
[*] Processed N binaries
[*] Running pairdata to organize extracted features...
[SUMMARY] Generated N pickle files in /data/kun/jtransdata/extract
```

In the IDA logs (`log/*.log`), you should see:

```
[INFO] SAVEROOT=/data/kun/jtransdata/extract
[INFO] DATAROOT=/data/kun/jtransdata/mini_train
[*] Processing binary: <name>
[+] function_name_1
[+] function_name_2
...
[SUCCESS] Extracted N functions
[SUCCESS] Saved to: /data/kun/jtransdata/extract/<name>_extract.pkl
```

## Troubleshooting

### No pickle files generated

Check the log files:
```bash
grep -r "ERROR\|Exception\|SUCCESS" log/*.log | head -20
```

### IDA not running

Make sure TVHEADLESS is set:
```bash
TVHEADLESS=1 ./ida-pro-9.0/idat --help
```

### Import errors

Make sure packages are in base Python:
```bash
/home/kun/anaconda3/bin/python3 -c "import binaryai, networkx; print('OK')"
```

## Files

- `process_fixed.py` - Fixed IDA Python script that extracts features
- `run_fixed.py` - Fixed main driver script
- `README_FIXED.md` - This file
