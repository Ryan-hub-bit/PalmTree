#!/usr/bin/env python3
import os
import subprocess
import sys

# Set up environment
os.environ['SAVEROOT'] = '/data/kun/jtransdata/extract'
os.environ['DATAROOT'] = '/data/kun/jtransdata/mini_train'
os.environ['TVHEADLESS'] = '1'

# Get first binary
dataset_dir = '/data/kun/jtransdata/mini_train'
binaries = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)][:1]

if not binaries:
    print("No binaries found!")
    sys.exit(1)

target = binaries[0]
filename = os.path.basename(target)
print(f"Testing with: {filename}")

# Strip binary
strip_path = '/tmp/test_ida_strip'
os.makedirs(strip_path, exist_ok=True)
filename_strip = filename + '.strip'
ida_input = os.path.join(strip_path, filename_strip)

strip_cmd = ['strip', '-s', target, '-o', ida_input]
result = subprocess.run(strip_cmd, capture_output=True)
if result.returncode != 0:
    print(f"Strip failed: {result.stderr.decode()}")
    sys.exit(1)
print(f"✓ Stripped to: {ida_input}")

# Run IDA
ida_path = "./ida-pro-9.0/idat"
script_path = os.path.abspath("./process.py")
log_file = f'log/test_{filename}.log'
idb_file = f'idb/test_{filename}.idb'

os.makedirs('log', exist_ok=True)
os.makedirs('idb', exist_ok=True)

cmd = [ida_path, f'-L{log_file}', '-c', '-A', f'-S{script_path}', f'-o{idb_file}', ida_input]
print(f"Running: {' '.join(cmd)}")
print("Environment:")
print(f"  SAVEROOT={os.environ['SAVEROOT']}")
print(f"  DATAROOT={os.environ['DATAROOT']}")
print(f"  TVHEADLESS={os.environ['TVHEADLESS']}")

result = subprocess.run(cmd, capture_output=True, text=True)
print(f"\nIDA exit code: {result.returncode}")

if result.stdout:
    print(f"\nSTDOUT:\n{result.stdout}")
if result.stderr:
    print(f"\nSTDERR:\n{result.stderr}")

# Check results
print("\n=== Checking Results ===")
if os.path.exists(log_file):
    print(f"✓ Log file created: {log_file}")
    with open(log_file, 'r') as f:
        log_content = f.read()
        if 'process.py' in log_content:
            print("✓ process.py was executed")
            if '[+]' in log_content:
                print("✓ Functions were processed")
            else:
                print("✗ No functions found in log")
        else:
            print("✗ process.py was not found in log")
        
        # Check for errors
        if 'Error' in log_content or 'Exception' in log_content:
            print("\n⚠ Errors found in log:")
            for line in log_content.split('\n'):
                if 'Error' in line or 'Exception' in line:
                    print(f"  {line}")
else:
    print(f"✗ No log file at {log_file}")

# Check for pickle
import glob
pkl_pattern = f"/data/kun/jtransdata/extract/*{filename[:20]}*.pkl"
pkl_files = glob.glob(pkl_pattern)
if pkl_files:
    print(f"✓ Pickle file(s) created: {len(pkl_files)} file(s)")
    for pkl in pkl_files:
        print(f"  - {pkl}")
else:
    print(f"✗ No pickle files matching pattern: {pkl_pattern}")
