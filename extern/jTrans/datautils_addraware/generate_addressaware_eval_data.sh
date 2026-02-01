#!/bin/bash
#
# Generate addressaware evaluation dataset from small_test
# This creates func_blocks and ground_truth for evaluation purposes
#
# Usage: ./generate_addressaware_eval_data.sh
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
STRIPPED_DIR="/data/kun/jtrans/small_test_strip"
NONSTRIPPED_DIR="/data/kun/jtrans/small_test"
OUTPUT_DIR="/data/kun/jtrans/addressaware/eval"
IDA_PATH="/home/kun/ida-pro-9.0/idat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="${SCRIPT_DIR}/function_export_ida.py"
COMBINE_SCRIPT="${SCRIPT_DIR}/combine_function_files_pretrain.py"
DATASET_SCRIPT="${SCRIPT_DIR}/create_function_dataset.py"
EXTRACT_DIR="${OUTPUT_DIR}/extract"

# Validate inputs
if [ ! -d "$STRIPPED_DIR" ]; then
    print_error "Stripped binary directory not found: $STRIPPED_DIR"
    exit 1
fi

if [ ! -d "$NONSTRIPPED_DIR" ]; then
    print_error "Non-stripped binary directory not found: $NONSTRIPPED_DIR"
    exit 1
fi

if [ ! -f "$IDA_PATH" ]; then
    print_error "IDA Pro not found at: $IDA_PATH"
    exit 1
fi

echo "========================================================================"
echo "      ADDRESSAWARE EVALUATION DATASET GENERATION"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Stripped binaries:     $STRIPPED_DIR"
echo "  Non-stripped binaries: $NONSTRIPPED_DIR"
echo "  Output directory:      $OUTPUT_DIR"
echo "  Extract directory:     $EXTRACT_DIR"
echo "  IDA Pro path:          $IDA_PATH"
echo ""
echo "This will generate:"
echo "  - func_blocks_addr.json (evaluation)"
echo "  - ground_truth_addr.json (evaluation)"
echo ""
echo "========================================================================"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$EXTRACT_DIR"
mkdir -p "./log"
mkdir -p "./idb"

# ============================================================================
# STEP 1: Function Export using IDA Pro
# ============================================================================
print_step "1/3" "Exporting functions from binaries using IDA Pro"
echo ""

# Export for function_export_ida.py (it reads OUTPUT_DIR env variable)
export IDA_OUTPUT_DIR="$EXTRACT_DIR"
export IDA_PATH

# Count binaries efficiently
echo "Counting binaries..."
total_binaries=$(find "$STRIPPED_DIR" -maxdepth 1 -type f \
    ! -name "*.idb" ! -name "*.i64" ! -name "*.id0" ! -name "*.id1" ! -name "*.id2" \
    ! -name "*.nam" ! -name "*.til" ! -name "*.asm" \
    ! -name "*_functions.txt" ! -name "*_functions.pkl" \
    ! -name "*.strip" ! -name "*.log" ! -name "*.txt" \
    ! -name "*.o" ! -name "*.a" \
    | wc -l)

echo "Found $total_binaries binary files to process"
echo ""

if [ $total_binaries -eq 0 ]; then
    print_error "No binary files found in $STRIPPED_DIR"
    exit 1
fi

# Process binaries sequentially (IDA Pro handles one at a time)
processed=0
failed=0

find "$STRIPPED_DIR" -maxdepth 1 -type f \
    ! -name "*.idb" ! -name "*.i64" ! -name "*.id0" ! -name "*.id1" ! -name "*.id2" \
    ! -name "*.nam" ! -name "*.til" ! -name "*.asm" \
    ! -name "*_functions.txt" ! -name "*_functions.pkl" \
    ! -name "*.strip" ! -name "*.log" ! -name "*.txt" \
    ! -name "*.o" ! -name "*.a" \
    | while read -r binary; do
    binary_name=$(basename "$binary")
    processed=$((processed + 1))
    
    echo "[$processed/$total_binaries] Processing: $binary_name"
    
    # Run IDA Pro in batch mode
    log_file="./log/${binary_name}.log"
    if "$IDA_PATH" -A -S"$EXPORT_SCRIPT" "$binary" > "$log_file" 2>&1; then
        print_success "Completed: $binary_name"
    else
        print_warning "Failed: $binary_name (check $log_file)"
        failed=$((failed + 1))
    fi
done

echo ""
print_success "Function export complete: $processed processed, $failed failed"

# Check function files
function_count=$(find "$EXTRACT_DIR" -name "*_functions.txt" | wc -l)
if [ $function_count -eq 0 ]; then
    print_error "No function files created!"
    exit 1
fi
echo "Created $function_count function files"
echo ""

# ============================================================================
# STEP 2: Combine pickle files into pretrain format (optional for eval)
# ============================================================================
print_step "2/3" "Combining pickle files to generate addr_pretrain.txt"
echo ""

if python3 "$COMBINE_SCRIPT" \
    --input_dir "$EXTRACT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --top_n 100 \
    --sample_ratio 1.0 \
    --enable_smart_merge \
    --output_name addr_pretrain.txt; then
    print_success "Pretrain file generated"
    
    if [ -f "$OUTPUT_DIR/addr_pretrain.txt" ]; then
        line_count=$(wc -l < "$OUTPUT_DIR/addr_pretrain.txt")
        echo "  - $OUTPUT_DIR/addr_pretrain.txt ($line_count functions)"
    fi
else
    print_warning "Failed to combine pickle files (not critical for eval)"
fi
echo ""

# ============================================================================
# STEP 3: Generate function dataset (JSON format)
# ============================================================================
print_step "3/3" "Generating function dataset (JSON)"
echo ""

if python3 "$DATASET_SCRIPT" \
    "$EXTRACT_DIR" \
    "$OUTPUT_DIR" \
    --binary-dir "$NONSTRIPPED_DIR"; then
    print_success "Function dataset generated"
    
    if [ -f "$OUTPUT_DIR/func_blocks_addr.json" ]; then
        func_count=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_DIR/func_blocks_addr.json'))))" 2>/dev/null || echo "?")
        echo "  - $OUTPUT_DIR/func_blocks_addr.json ($func_count functions)"
    fi
    
    if [ -f "$OUTPUT_DIR/ground_truth_addr.json" ]; then
        pair_count=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/ground_truth_addr.json'))['total_pairs'])" 2>/dev/null || echo "?")
        echo "  - $OUTPUT_DIR/ground_truth_addr.json ($pair_count groups)"
    fi
else
    print_error "Failed to create function dataset"
    exit 1
fi

echo ""
echo "========================================================================"
print_success "EVALUATION DATASET GENERATION FINISHED"
echo "========================================================================"
echo ""
echo "Generated files in: $OUTPUT_DIR"
echo ""
echo "Evaluation files:"
echo "  - func_blocks_addr.json"
echo "  - ground_truth_addr.json"
echo ""
echo "Next steps:"
echo "  1. Create filtered pools: ./run_create_addressaware_filtered_pools.sh"
echo "  2. Evaluate: ./run_addressaware_pool_evaluation.sh"
echo ""
echo "========================================================================"
