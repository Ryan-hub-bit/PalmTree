#!/bin/bash
#
# Combine address-aware text files and split into train/test
# Uses smart_merge approach to keep only top 100 symbol names (like .printf, .malloc, etc.)
#
# Usage:
#   bash combine_and_split.sh /path/to/extract/dir /path/to/output/dir
#

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  input_dir  : Directory containing *_addressaware.txt files"
    echo "  output_dir : Directory where train/test splits will be saved"
    echo ""
    echo "Example:"
    echo "  $0 /data/kun/jtransdata/extract ./data"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Verify input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Input directory: $INPUT_DIR"
echo "[INFO] Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Combine all address-aware text files
COMBINED_FILE="$OUTPUT_DIR/combined_all.txt"
echo "[INFO] Step 1: Combining address-aware text files (*_addressaware.txt)..."

# Remove old combined file if exists
rm -f "$COMBINED_FILE"

file_count=0
for file in "$INPUT_DIR"/*_addressaware.txt; do
    if [ -f "$file" ]; then
        cat "$file" >> "$COMBINED_FILE"
        file_count=$((file_count + 1))
        if [ $((file_count % 100)) -eq 0 ]; then
            echo "    [+] Processed $file_count files..."
        fi
    fi
done

if [ $file_count -eq 0 ]; then
    echo "[ERROR] No *_addressaware.txt files found in $INPUT_DIR"
    exit 1
fi

echo "[INFO] Combined $file_count files"

# Step 2: Find top 100 symbol names (following smart_merge.py approach)
echo ""
echo "[INFO] Step 2: Analyzing symbol names (pattern: .symbol_name)..."

# Create Python script to find top 100 symbols and replace others with .plt
cat > "$OUTPUT_DIR/process_symbols.py" << 'PYTHON_SCRIPT'
import sys
import re
from collections import Counter

def main():
    if len(sys.argv) != 3:
        print("Usage: python process_symbols.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Regex to match symbol names (following cfg_hierarchical_icfg_ida.py)
    # Matches . followed by a letter or underscore, then any word characters
    pattern = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)')
    
    print("[INFO] Counting symbol frequencies...")
    counts = Counter()
    
    with open(input_file, 'r', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"    Processed {line_num} lines...")
            counts.update(pattern.findall(line))
    
    # Get top 100 most common symbols
    top_100 = set(name for name, count in counts.most_common(100))
    
    print(f"[INFO] Found {len(counts)} unique symbols")
    print(f"[INFO] Keeping top 100 symbols, replacing others with .plt")
    print(f"[INFO] Top 10 symbols: {[f'.{s}' for s, _ in counts.most_common(10)]}")
    
    # Save top 100 symbols list
    top_list_file = output_file.rsplit('.', 1)[0] + '_top100_symbols.txt'
    with open(top_list_file, 'w') as f:
        for symbol in sorted(top_100):
            f.write(f".{symbol}\n")
    print(f"[INFO] Saved top 100 symbols to: {top_list_file}")
    
    print("[INFO] Processing file and replacing symbols...")
    with open(input_file, 'r', errors='ignore') as in_f, \
         open(output_file, 'w') as out_f:
        for line_num, line in enumerate(in_f, 1):
            if line_num % 100000 == 0:
                print(f"    Processed {line_num} lines...")
            
            # Replace function: keep if in top_100, else replace with .plt
            def replace_func(match):
                symbol = match.group(1)
                return f".{symbol}" if symbol in top_100 else ".plt"
            
            new_line = pattern.sub(replace_func, line)
            out_f.write(new_line)
    
    print("[SUCCESS] Symbol replacement complete!")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Run the Python script
PROCESSED_FILE="$OUTPUT_DIR/combined_processed.txt"
python3 "$OUTPUT_DIR/process_symbols.py" "$COMBINED_FILE" "$PROCESSED_FILE"

if [ ! -f "$PROCESSED_FILE" ]; then
    echo "[ERROR] Symbol processing failed!"
    exit 1
fi

if [ ! -f "$PROCESSED_FILE" ]; then
    echo "[ERROR] Symbol processing failed!"
    exit 1
fi

# Step 3: Split into train and test
echo ""
echo "[INFO] Step 3: Splitting into train (90%) and test (10%)..."

# Count total lines
total_lines=$(wc -l < "$PROCESSED_FILE")
echo "[INFO] Total lines: $total_lines"

# Calculate split (90% train, 10% test)
train_lines=$((total_lines * 9 / 10))
test_lines=$((total_lines - train_lines))

echo "[INFO] Train lines: $train_lines"
echo "[INFO] Test lines: $test_lines"

# Split into train and test
TRAIN_FILE="$OUTPUT_DIR/addressaware_train.txt"
TEST_FILE="$OUTPUT_DIR/addressaware_test.txt"

echo "[INFO] Creating train/test files..."
head -n "$train_lines" "$PROCESSED_FILE" > "$TRAIN_FILE"
tail -n "$test_lines" "$PROCESSED_FILE" > "$TEST_FILE"

echo ""
echo "================================"
echo "SUCCESS! Files created:"
echo "================================"
echo "  Train: $TRAIN_FILE ($train_lines lines)"
echo "  Test:  $TEST_FILE ($test_lines lines)"
echo "  Top 100 symbols: $OUTPUT_DIR/combined_processed_top100_symbols.txt"
echo "================================"

echo "  Top 100 symbols: $OUTPUT_DIR/combined_processed_top100_symbols.txt"
echo "================================"

# Show sample
echo ""
echo "Sample from train file (first 2 lines):"
echo "--------------------------------"
head -n 2 "$TRAIN_FILE"
echo ""
echo "================================"
echo "Token format reference:"
echo "================================"
echo "Following cfg_hierarchical_icfg_ida.py token processing:"
echo ""
echo "  Opcode with position:"
echo "    push(0x401000:0.12345678:0.23456789:0.34567890)"
echo ""
echo "  Code address (in .text):"
echo "    address(0x401050:0.12345678:0.23456789:0.34567890)"
echo "    - Position 1: Function position in binary"
echo "    - Position 2: Basic block position in function"
echo "    - Position 3: Instruction position in basic block"
echo ""
echo "  Data address (in .data, .rodata, .bss):"
echo "    daddr(0x600000:0.50000000:0.25000000:0.00000000)"
echo "    - Position 1: Section position in binary"
echo "    - Position 2: Address position in section"
echo "    - Position 3: 0.0 (no BB context)"
echo ""
echo "  PLT symbols (top 100 kept, others replaced with .plt):"
echo "    .printf, .malloc, .free, etc. (top 100)"
echo "    .plt (for less common symbols)"
echo ""
echo "  Other tokens:"
echo "    var(0x10)  - Stack variables"
echo "    imm        - Immediate values"
echo "    disp       - Displacement operands"
echo "================================"
echo ""