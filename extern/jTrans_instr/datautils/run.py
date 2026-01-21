#!/usr/bin/env python3
import os
import subprocess
import multiprocessing
import time
from util.pairdata import pairdata

# Configuration
ida_path = "./ida-pro-9.0/idat"  # 32-bit IDA
work_dir = os.path.abspath('.')
dataset_dir = '/data/kun/jtransdata/small_train'
strip_path = os.environ.get('STRIP_PATH', '/data/kun/jtransdata/small_train_strip')
script_path = os.path.abspath("./process.py")
SAVE_ROOT = "/data/kun/jtrans_instr/extract"

# Create all necessary directories
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(strip_path, exist_ok=True)
os.makedirs('./log', exist_ok=True)
os.makedirs('./idb', exist_ok=True)

print(f"[CONFIG] IDA path: {ida_path}")
print(f"[CONFIG] Dataset: {dataset_dir}")
print(f"[CONFIG] Strip path: {strip_path}")
print(f"[CONFIG] Save root: {SAVE_ROOT}")
print(f"[CONFIG] Script: {script_path}")

def run_ida_with_env(cmd, env):
    """Wrapper function to run IDA with custom environment variables"""
    return subprocess.call(cmd, env=env)

def getTarget(path, prefixfilter=None):
    """Get list of binaries to process, excluding IDA-generated files"""
    target = []
    # File extensions to skip
    skip_extensions = {'.i64', '.idb', '.id0', '.id1', '.id2', '.nam', '.til', '.txt', '.log', '.strip'}
    
    for root, dirs, files in os.walk(path):
        for file in files:
            # Skip files with IDA-generated extensions
            if any(file.endswith(ext) for ext in skip_extensions):
                continue
                
            if prefixfilter is None:
                target.append(os.path.join(root, file))
            else:
                for prefix in prefixfilter:
                    if file.startswith(prefix):
                        target.append(os.path.join(root, file))
    return target

if __name__ == '__main__':
    # prefixfilter = ['libcap-git-setcap']
    start = time.time()
    target_list = getTarget(dataset_dir)
    
    print(f"\n[*] Found {len(target_list)} binaries to process")
    print(f"[*] Starting parallel processing with 8 workers\n")

    pool = multiprocessing.Pool(processes=8)
    success_count = 0
    
    for target in target_list:
        filename = os.path.basename(target)
        filename_strip = filename + '.strip'
        ida_input = os.path.join(strip_path, filename_strip)

        # Run strip and check for errors
        strip_cmd = ['strip', '-s', target, '-o', ida_input]
        try:
            subprocess.run(strip_cmd, check=True, capture_output=True)
            print(f"✓ strip succeeded: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"✗ ERROR: strip failed for {filename}: {e}")
            continue

        # Start IDA in async pool with env vars
        cmd = [ida_path, f'-Llog/{filename}.log', '-c', '-A', f'-S{script_path}', f'-oidb/{filename}.i64', ida_input]
        print(f"→ Starting IDA: {filename}")
        
        # Set environment variables for the IDA subprocess
        env = os.environ.copy()
        env['SAVEROOT'] = SAVE_ROOT
        env['DATAROOT'] = dataset_dir
        env['TVHEADLESS'] = '1'  # Required for IDA batch mode
        
        # Use wrapper function to pass env correctly through multiprocessing
        pool.apply_async(run_ida_with_env, args=(cmd, env))
        success_count += 1
    
    pool.close()
    pool.join()
    
    print('\n[*] Features Extracting Done')
    print(f'[*] Processed {success_count} binaries')
    
    # Pair the data
    print('\n[*] Running pairdata to organize extracted features...')
    pairdata(SAVE_ROOT)
    
    end = time.time()
    print(f"[*] Total Time: {end - start:.2f} seconds")
    
    # Show summary
    import glob
    pkl_files = glob.glob(os.path.join(SAVE_ROOT, '*_extract.pkl'))
    print(f"\n[SUMMARY] Generated {len(pkl_files)} pickle files in {SAVE_ROOT}")
