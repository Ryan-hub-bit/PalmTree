#!/bin/bash
#
# Complete pipeline for function export and dataset creation
# 
# Usage: ./run_pipeline.sh <stripped_binary_dir> <non_stripped_binary_dir> [output_dir]
#
# Example:
#   ./run_pipeline.sh /data/kun/jtransdata/small_train_strip /data/kun/jtransdata/small_train
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
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

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <stripped_binary_dir> <non_stripped_binary_dir> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  stripped_binary_dir     : Directory containing stripped binaries for export"
    echo "  non_stripped_binary_dir : Directory containing non-stripped binaries for function names"
    echo "  output_dir              : (Optional) Output directory (default: function_exports in stripped dir)"
    echo ""
    echo "Example:"
    echo "  $0 /data/kun/jtransdata/small_train_strip /data/kun/jtransdata/small_train"
    echo ""
    echo "This script will:"
    echo "  1. Export functions from stripped binaries using IDA Pro"
    echo "  2. Combine and create train/val/test splits (generates top_symbols.txt)"
    echo "  3. Create function dataset with ground truth (uses same top_symbols.txt)"
    exit 1
fi

STRIPPED_DIR="$1"
NONSTRIPPED_DIR="$2"
OUTPUT_DIR="${3:-/data/kun/jtransdata/function_exports}"

# Configuration
IDA_PATH="/home/kun/ida-pro-9.0/idat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="${SCRIPT_DIR}/function_export_ida.py"
COMBINE_SCRIPT="${SCRIPT_DIR}/combine_function_files.py"
DATASET_SCRIPT="${SCRIPT_DIR}/create_function_dataset.py"

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
echo "                    FUNCTION DATASET PIPELINE"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Stripped binaries:     $STRIPPED_DIR"
echo "  Non-stripped binaries: $NONSTRIPPED_DIR"
echo "  Output directory:      $OUTPUT_DIR"
echo "  IDA Pro path:          $IDA_PATH"
echo ""
echo "========================================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# STEP 1: Export functions from stripped binaries using IDA Pro
# ============================================================================
print_step "1/3" "Exporting functions from stripped binaries"
echo ""

export OUTPUT_DIR
export IDA_PATH

# Count binaries to process
total_binaries=$(find "$STRIPPED_DIR" -maxdepth 1 -type f -executable | wc -l)
echo "Found $total_binaries executable files to process"
echo ""

if [ $total_binaries -eq 0 ]; then
    print_error "No executable files found in $STRIPPED_DIR"
    exit 1
fi

processed=0
failed=0

for binary in "$STRIPPED_DIR"/*; do
    # Check if it's an executable file
    if [ -f "$binary" ] && [ -x "$binary" ]; then
        binary_name=$(basename "$binary")
        processed=$((processed + 1))
        
        echo "[$processed/$total_binaries] Processing: $binary_name"
        
        # Run IDA Pro in batch mode
        if "$IDA_PATH" -A -S"$EXPORT_SCRIPT" "$binary" > /dev/null 2>&1; then
            print_success "Completed: $binary_name"
        else
            print_warning "Failed: $binary_name"
            failed=$((failed + 1))
        fi
    fi
done

echo ""
print_success "Function export complete: $processed processed, $failed failed"
echo ""

# ============================================================================
# STEP 2: Combine function files and create train/val/test splits
# ============================================================================
print_step "2/3" "Combining function files and creating train/val/test splits"
echo ""

# Create subdirectories for train/val/test organization if needed
# For now, we'll run combine on all exported files
echo "Running combine_function_files.py..."
echo ""

if python3 "$COMBINE_SCRIPT" \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --enable_smart_merge \
    --top_n 100 \
    --train_sample_ratio 1; then
    print_success "Train/val/test splits created"
    
    if [ -f "$OUTPUT_DIR/top_symbols.txt" ]; then
        symbol_count=$(grep -v '^#' "$OUTPUT_DIR/top_symbols.txt" | wc -l)
        echo "  - Top symbols file: $OUTPUT_DIR/top_symbols.txt ($symbol_count symbols)"
    fi
    
    if [ -f "$OUTPUT_DIR/train.txt" ]; then
        train_count=$(wc -l < "$OUTPUT_DIR/train.txt")
        echo "  - Train file: $OUTPUT_DIR/train.txt ($train_count functions)"
    fi
    
    if [ -f "$OUTPUT_DIR/val.txt" ]; then
        val_count=$(wc -l < "$OUTPUT_DIR/val.txt")
        echo "  - Val file: $OUTPUT_DIR/val.txt ($val_count functions)"
    fi
    
    if [ -f "$OUTPUT_DIR/test.txt" ]; then
        test_count=$(wc -l < "$OUTPUT_DIR/test.txt")
        echo "  - Test file: $OUTPUT_DIR/test.txt ($test_count functions)"
    fi
else
    print_error "Failed to combine function files"
    exit 1
fi

echo ""

# ============================================================================
# STEP 3: Create function dataset with ground truth
# ============================================================================
print_step "3/3" "Creating function dataset with ground truth"
echo ""

echo "Running create_function_dataset.py..."
echo ""

if python3 "$DATASET_SCRIPT" \
    "$OUTPUT_DIR" \
    "$OUTPUT_DIR" \
    --binary-dir "$NONSTRIPPED_DIR" \
    --symbol-file "$OUTPUT_DIR/top_symbols.txt"; then
    print_success "Function dataset created"
    
    if [ -f "$OUTPUT_DIR/func_blocks_addr.json" ]; then
        func_count=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_DIR/func_blocks_addr.json'))))")
        echo "  - Function blocks: $OUTPUT_DIR/func_blocks_addr.json ($func_count functions)"
    fi
    
    if [ -f "$OUTPUT_DIR/ground_truth_addr.json" ]; then
        pair_count=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/ground_truth_addr.json'))['total_pairs'])")
        echo "  - Ground truth: $OUTPUT_DIR/ground_truth_addr.json ($pair_count function groups)"
    fi
else
    print_error "Failed to create function dataset"
    exit 1
fi

echo ""
echo "========================================================================"
print_success "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Output files in: $OUTPUT_DIR"
echo ""
echo "Train/Val/Test splits:"
echo "  - train.txt"
echo "  - val.txt"
echo "  - test.txt"
echo ""
echo "Function similarity dataset:"
echo "  - func_blocks_addr.json"
echo "  - ground_truth_addr.json"
echo ""
echo "Shared vocabulary:"
echo "  - top_symbols.txt"
echo ""
echo "You can now use these files for training!"
echo "========================================================================"
