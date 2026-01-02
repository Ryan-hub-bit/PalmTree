#!/bin/bash
#
# Test script to verify PLT symbol preservation works
# This processes ONE test binary to confirm the fixes
#

set -e

# Configuration
IDA_PATH="${IDA_PATH:-/home/kun/ida-pro-9.0/idat}"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/process.py"
TEST_OUTPUT="/tmp/test_plt_output"
TEST_LOGS="/tmp/test_plt_logs"

# Check if test binary is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_test_binary>"
    echo ""
    echo "Example:"
    echo "  $0 /data/kun/unstripped_binary_O0/someprogram.strip"
    echo ""
    echo "This script will:"
    echo "  1. Process the binary with updated process.py"
    echo "  2. Check if PLT symbols (.printf, .malloc, etc) appear in output"
    echo "  3. Show example call instructions"
    exit 1
fi

TEST_BINARY="$1"

if [ ! -f "$TEST_BINARY" ]; then
    echo "[ERROR] Binary not found: $TEST_BINARY"
    exit 1
fi

BINARY_NAME=$(basename "$TEST_BINARY")

# Clean previous test output
rm -rf "$TEST_OUTPUT" "$TEST_LOGS"
mkdir -p "$TEST_OUTPUT" "$TEST_LOGS"

echo "=========================================="
echo "PLT Symbol Preservation Test"
echo "=========================================="
echo "Binary:     $BINARY_NAME"
echo "Output:     $TEST_OUTPUT"
echo "Logs:       $TEST_LOGS"
echo ""

# Set environment variables
export SAVEROOT="$TEST_OUTPUT"
export DATAROOT="$(dirname "$TEST_BINARY")"
export LOGROOT="$TEST_LOGS"
export TVHEADLESS=1

echo "[1/3] Processing binary with IDA Pro..."
"$IDA_PATH" -L"$TEST_LOGS/${BINARY_NAME}_ida.log" -c -A -S"$SCRIPT_PATH" "$TEST_BINARY" >/dev/null 2>&1

# Check outputs
PICKLE_FILE="$TEST_OUTPUT/${BINARY_NAME}_extract.pkl"
TEXT_FILE="$TEST_OUTPUT/${BINARY_NAME}_addressaware.txt"

if [ ! -f "$PICKLE_FILE" ] || [ ! -f "$TEXT_FILE" ]; then
    echo "[ERROR] Failed to generate output files!"
    echo "Check log: $TEST_LOGS/${BINARY_NAME}_ida.log"
    exit 1
fi

echo "[SUCCESS] Generated output files"
echo ""

echo "[2/3] Searching for PLT symbols..."
PLT_SYMBOLS=$(grep -oh "\.\(printf\|malloc\|free\|calloc\|realloc\|memcpy\|memset\|strlen\|strcmp\|strcpy\|fopen\|fclose\|exit\|puts\|putchar\|getchar\|scanf\|fprintf\|sprintf\)" "$TEXT_FILE" | sort | uniq)

if [ -z "$PLT_SYMBOLS" ]; then
    echo "[WARNING] No PLT symbols found!"
    echo ""
    echo "This might mean:"
    echo "  - Binary doesn't call standard library functions"
    echo "  - Binary is statically linked"
    echo "  - PLT detection needs adjustment"
else
    echo "[SUCCESS] Found PLT symbols:"
    echo "$PLT_SYMBOLS" | sed 's/^/  /'
    echo ""
    echo "Count: $(echo "$PLT_SYMBOLS" | wc -l) unique PLT symbols"
fi

echo ""
echo "[3/3] Sample call instructions:"
grep "call(" "$TEXT_FILE" | head -10

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo "Files generated:"
echo "  $PICKLE_FILE"
echo "  $TEXT_FILE"
echo "  $TEST_LOGS/${BINARY_NAME}_ida.log"
echo ""
echo "To verify:"
echo "  grep '\.printf\|\.malloc\|\.free' $TEXT_FILE"
