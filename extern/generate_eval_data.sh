#!/bin/bash
#
# Generate evaluation data for baseline, jtrans_instr and addressaware
# Only generates the necessary JSON files for evaluation (func_blocks and ground_truth)
#
# Usage: ./generate_eval_data.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
BINARY_DIR="/data/kun/jtransdata/small_test"
EVAL_DIR="/data/kun/jtransdata/eval"
IDA_PATH="/home/kun/ida-pro-9.0/idat"
SCRIPT_BASE="/home/kun/Document/AAE/extern"

# Extract directories
BASELINE_EXTRACT_DIR="$EVAL_DIR/extract_baseline"
INSTR_EXTRACT_DIR="$EVAL_DIR/extract_instr"
ADDRAWARE_EXPORT_DIR="$EVAL_DIR/export_addraware"

echo "========================================================================"
echo "                    GENERATE EVALUATION DATA"
echo "========================================================================"
echo ""
echo "This script generates evaluation JSON files for:"
echo "  1. Baseline (func_blocks_baseline.json, ground_truth_baseline.json)"
echo "  2. jTrans_instr (func_blocks_instr.json, ground_truth_instr.json)"
echo "  3. Address-aware (func_blocks.json, ground_truth.json)"
echo "  4. Fair Pools (pool and query files for function similarity)"
echo ""
echo "Configuration:"
echo "  Binary directory:    $BINARY_DIR"
echo "  Output directory:    $EVAL_DIR"
echo "  IDA Pro path:        $IDA_PATH"
echo ""
echo "========================================================================"
echo ""

# Verify binary directory exists
if [ ! -d "$BINARY_DIR" ]; then
    print_error "Binary directory not found: $BINARY_DIR"
    exit 1
fi

# Create output directories
mkdir -p "$EVAL_DIR/baseline"
mkdir -p "$EVAL_DIR/jtrans_instr"
mkdir -p "$EVAL_DIR/addraware"
mkdir -p "$BASELINE_EXTRACT_DIR"
mkdir -p "$INSTR_EXTRACT_DIR"
mkdir -p "$ADDRAWARE_EXPORT_DIR"

# Count binaries
total_binaries=$(find "$BINARY_DIR" -maxdepth 1 -type f \
    ! -name "*.i64" ! -name "*.idb" ! -name "*.id0" ! -name "*.id1" ! -name "*.id2" \
    ! -name "*.nam" ! -name "*.til" ! -name "*.txt" ! -name "*.log" ! -name "*.strip" | wc -l)
echo "Found $total_binaries binaries to process"
echo ""

# ============================================================================
# STEP 1: Generate Baseline Evaluation Data
# ============================================================================
print_step "1/3" "Generating Baseline Evaluation Data"
echo ""

cd "$SCRIPT_BASE/jTrans/datautils"

print_warning "Running IDA extraction for baseline (this may take a while)..."
echo ""

# Create a temporary run script for baseline
cat > /tmp/run_baseline_eval.py << 'EOFPYTHON'
#!/usr/bin/env python3
import os
import subprocess
import multiprocessing
import time
import sys

# Get configuration from environment
ida_path = os.environ.get('IDA_PATH')
dataset_dir = os.environ.get('BINARY_DIR')
strip_path = os.environ.get('STRIP_PATH')
SAVE_ROOT = os.environ.get('EXTRACT_DIR')

# Add current directory to path for imports
sys.path.insert(0, '.')
# NOTE: pairdata is not needed for new evaluation pipeline

# Create necessary directories
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(strip_path, exist_ok=True)
os.makedirs('./log', exist_ok=True)
os.makedirs('./idb', exist_ok=True)

print(f"[CONFIG] IDA path: {ida_path}")
print(f"[CONFIG] Dataset: {dataset_dir}")
print(f"[CONFIG] Strip path: {strip_path}")
print(f"[CONFIG] Save root: {SAVE_ROOT}")

def run_ida_with_env(cmd, env):
    return subprocess.call(cmd, env=env)

def getTarget(path):
    target = []
    skip_extensions = {'.i64', '.idb', '.id0', '.id1', '.id2', '.nam', '.til', '.txt', '.log', '.strip'}
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.endswith(ext) for ext in skip_extensions):
                continue
            target.append(os.path.join(root, file))
    return target

if __name__ == '__main__':
    start = time.time()
    target_list = getTarget(dataset_dir)
    script_path = os.path.abspath("./process.py")
    
    print(f"\n[*] Found {len(target_list)} binaries to process")
    print(f"[*] Starting parallel processing with 8 workers\n")

    pool = multiprocessing.Pool(processes=8)
    success_count = 0
    
    for target in target_list:
        filename = os.path.basename(target)
        filename_strip = filename + '.strip'
        ida_input = os.path.join(strip_path, filename_strip)

        strip_cmd = ['strip', '-s', target, '-o', ida_input]
        try:
            subprocess.run(strip_cmd, check=True, capture_output=True)
            print(f"✓ strip: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"✗ strip failed: {filename}")
            continue

        cmd = [ida_path, f'-Llog/{filename}.log', '-c', '-A', f'-S{script_path}', f'-oidb/{filename}.i64', ida_input]
        
        env = os.environ.copy()
        env['SAVEROOT'] = SAVE_ROOT
        env['DATAROOT'] = dataset_dir
        env['TVHEADLESS'] = '1'
        
        pool.apply_async(run_ida_with_env, args=(cmd, env))
        success_count += 1
    
    pool.close()
    pool.join()
    
    end = time.time()
    print(f"[*] Time: {end - start:.2f}s")
    
    # Check if any pickle files were generated
    import glob
    pkl_files = glob.glob(os.path.join(SAVE_ROOT, '*_extract.pkl'))
    if len(pkl_files) == 0:
        print('[ERROR] No pickle files found! IDA extraction may have failed.')
        print('[ERROR] Check log files in ./log/ directory')
        sys.exit(1)
    
    print(f'[*] Found {len(pkl_files)} pickle files ready for JSON generation')
    # NOTE: pairdata() is NOT needed for new evaluation pipeline
    # The create_*_dataset.py scripts directly read pickle files
EOFPYTHON

# Run baseline extraction
export IDA_PATH="$IDA_PATH"
export BINARY_DIR="$BINARY_DIR"
export EXTRACT_DIR="$BASELINE_EXTRACT_DIR"
export STRIP_PATH="$EVAL_DIR/strip_baseline"

python3 /tmp/run_baseline_eval.py

# Generate baseline dataset (JSON only, no pretrain.txt)
echo ""
echo "Generating baseline JSON files..."
python3 create_baseline_dataset.py \
    "$BASELINE_EXTRACT_DIR" \
    "$EVAL_DIR/baseline" \
    --binary-dir "$BINARY_DIR"

print_success "Baseline evaluation data generated!"
echo "  - $EVAL_DIR/baseline/func_blocks_baseline.json"
echo "  - $EVAL_DIR/baseline/ground_truth_baseline.json"
echo ""

# ============================================================================
# STEP 2: Generate jTrans_instr Evaluation Data
# ============================================================================
print_step "2/3" "Generating jTrans_instr Evaluation Data"
echo ""

cd "$SCRIPT_BASE/jTrans_instr/datautils"

print_warning "Running IDA extraction for instruction-level..."
echo ""

# Create temporary run script for instruction-level
cat > /tmp/run_instr_eval.py << 'EOFPYTHON'
#!/usr/bin/env python3
import os
import subprocess
import multiprocessing
import time
import sys

ida_path = os.environ.get('IDA_PATH')
dataset_dir = os.environ.get('BINARY_DIR')
strip_path = os.environ.get('STRIP_PATH')
SAVE_ROOT = os.environ.get('EXTRACT_DIR')

sys.path.insert(0, '.')
# NOTE: pairdata is not needed for new evaluation pipeline

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(strip_path, exist_ok=True)
os.makedirs('./log', exist_ok=True)
os.makedirs('./idb', exist_ok=True)

print(f"[CONFIG] Dataset: {dataset_dir}")
print(f"[CONFIG] Save root: {SAVE_ROOT}")

def run_ida_with_env(cmd, env):
    return subprocess.call(cmd, env=env)

def getTarget(path):
    target = []
    skip_extensions = {'.i64', '.idb', '.id0', '.id1', '.id2', '.nam', '.til', '.txt', '.log', '.strip'}
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.endswith(ext) for ext in skip_extensions):
                continue
            target.append(os.path.join(root, file))
    return target

if __name__ == '__main__':
    start = time.time()
    target_list = getTarget(dataset_dir)
    script_path = os.path.abspath("./process_instr.py")
    
    print(f"\n[*] Found {len(target_list)} binaries")
    print(f"[*] Processing with 8 workers\n")

    pool = multiprocessing.Pool(processes=8)
    
    for target in target_list:
        filename = os.path.basename(target)
        filename_strip = filename + '.strip'
        ida_input = os.path.join(strip_path, filename_strip)

        strip_cmd = ['strip', '-s', target, '-o', ida_input]
        try:
            subprocess.run(strip_cmd, check=True, capture_output=True)
            print(f"✓ strip: {filename}")
        except subprocess.CalledProcessError:
            print(f"✗ strip failed: {filename}")
            continue

        cmd = [ida_path, f'-Llog/{filename}.log', '-c', '-A', f'-S{script_path}', f'-oidb/{filename}.i64', ida_input]
        
        env = os.environ.copy()
        env['SAVEROOT'] = SAVE_ROOT
        env['DATAROOT'] = dataset_dir
        env['TVHEADLESS'] = '1'
        
        pool.apply_async(run_ida_with_env, args=(cmd, env))
    
    pool.close()
    pool.join()
    
    end = time.time()
    print(f"[*] Time: {end - start:.2f}s")
    
    # Check if any pickle files were generated
    import glob
    pkl_files = glob.glob(os.path.join(SAVE_ROOT, '*_extract.pkl'))
    if len(pkl_files) == 0:
        print('[ERROR] No pickle files found! IDA extraction may have failed.')
        print('[ERROR] Check log files in ./log/ directory')
        sys.exit(1)
    
    print(f'[*] Found {len(pkl_files)} pickle files ready for JSON generation')
    # NOTE: pairdata() is NOT needed for new evaluation pipeline
    # The create_*_dataset.py scripts directly read pickle files
EOFPYTHON

# Run instruction-level extraction (reuse baseline strip directory)
export STRIP_PATH="$EVAL_DIR/strip_baseline"
export EXTRACT_DIR="$INSTR_EXTRACT_DIR"

python3 /tmp/run_instr_eval.py

# Generate instruction-level dataset (JSON only)
echo ""
echo "Generating instruction-level JSON files..."
python3 create_instr_dataset.py \
    "$INSTR_EXTRACT_DIR" \
    "$EVAL_DIR/jtrans_instr" \
    --binary-dir "$BINARY_DIR"

print_success "jTrans_instr evaluation data generated!"
echo "  - $EVAL_DIR/jtrans_instr/func_blocks_instr.json"
echo "  - $EVAL_DIR/jtrans_instr/ground_truth_instr.json"
echo ""

# ============================================================================
# STEP 3: Generate Address-aware Evaluation Data
# ============================================================================
print_step "3/3" "Generating Address-aware Evaluation Data"
echo ""

cd "$SCRIPT_BASE/jTrans/datautils_addraware"

print_warning "Exporting functions with address-aware format (using baseline stripped binaries)..."
echo ""

# Reuse baseline strip directory
STRIP_DIR="$EVAL_DIR/strip_baseline"

# Export functions using IDA
processed=0
failed=0

for stripped_binary in "$STRIP_DIR"/*.strip; do
    if [ -f "$stripped_binary" ]; then
        binary_name=$(basename "$stripped_binary" .strip)
        
        processed=$((processed + 1))
        echo "[$processed/$total_binaries] Processing: $binary_name"
        
        # Export using IDA on stripped binary
        export OUTPUT_DIR="$ADDRAWARE_EXPORT_DIR"
        if "$IDA_PATH" -A -S"function_export_ida.py" "$stripped_binary" > /dev/null 2>&1; then
            echo "  ✓ Completed"
        else
            echo "  ✗ Failed"
            failed=$((failed + 1))
        fi
    fi
done

echo ""
echo "Processed: $processed, Failed: $failed"
echo ""

# Create address-aware dataset (JSON only)
echo "Creating address-aware JSON files..."
python3 create_function_dataset.py \
    "$ADDRAWARE_EXPORT_DIR" \
    "$EVAL_DIR/addraware" \
    --binary-dir "$BINARY_DIR"

print_success "Address-aware evaluation data generated!"
echo "  - $EVAL_DIR/addraware/func_blocks.json"
echo "  - $EVAL_DIR/addraware/ground_truth.json"
echo ""

# ============================================================================
# STEP 4: Generate Fair Pools for Function Similarity Evaluation
# ============================================================================
print_step "4/4" "Generating Fair Pools for Function Similarity"
echo ""

cd "$SCRIPT_BASE/jTrans_instr"

print_warning "Generating fair pools for all three methods..."
echo ""

FAIR_POOL_DIR="$EVAL_DIR/fair_pools"
mkdir -p "$FAIR_POOL_DIR"

python3 generate_fair_pools_all.py \
    --func_blocks_baseline "$EVAL_DIR/baseline/func_blocks_baseline.json" \
    --func_blocks_addressaware "$EVAL_DIR/addraware/func_blocks.json" \
    --func_blocks_instr "$EVAL_DIR/jtrans_instr/func_blocks_instr.json" \
    --ground_truth_baseline "$EVAL_DIR/baseline/ground_truth_baseline.json" \
    --ground_truth_addressaware "$EVAL_DIR/addraware/ground_truth.json" \
    --ground_truth_instr "$EVAL_DIR/jtrans_instr/ground_truth_instr.json" \
    --output_dir "$FAIR_POOL_DIR" \
    --pool_sizes 100 1000 10000 \
    --min_token_length 30 \
    --seed 42

print_success "Fair pools generated!"
echo "  - $FAIR_POOL_DIR/pool_*.json"
echo "  - $FAIR_POOL_DIR/query_pool_*.json"
echo "  - $FAIR_POOL_DIR/pool_statistics.json"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "                        COMPLETE!"
echo "========================================================================"
echo ""
print_success "All evaluation data generated successfully!"
echo ""
echo "Evaluation data locations:"
echo ""
echo "1. Baseline:"
echo "   $EVAL_DIR/baseline/func_blocks_baseline.json"
echo "   $EVAL_DIR/baseline/ground_truth_baseline.json"
echo ""
echo "2. jTrans_instr:"
echo "   $EVAL_DIR/jtrans_instr/func_blocks_instr.json"
echo "   $EVAL_DIR/jtrans_instr/ground_truth_instr.json"
echo ""
echo "3. Address-aware:"
echo "   $EVAL_DIR/addraware/func_blocks.json"
echo "   $EVAL_DIR/addraware/ground_truth.json"
echo ""
echo "4. Fair Pools (for function similarity evaluation):"
echo "   $EVAL_DIR/fair_pools/"
echo "   - pool_100_*.json, pool_1000_*.json, pool_10000_*.json"
echo "   - query_pool_100_*.json, query_pool_1000_*.json, query_pool_10000_*.json"
echo "   - pool_statistics.json"
echo ""
echo "Usage:"
echo "  - Use func_blocks and ground_truth for training/finetuning evaluation"
echo "  - Use fair_pools for function similarity evaluation with fair comparison"
echo ""
echo "========================================================================"
