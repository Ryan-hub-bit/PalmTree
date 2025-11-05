#!/bin/bash
#
# Test script to generate CFG and DFG data for ghex binary only
#
# Usage: ./generate_ghex_test.sh
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  PalmTree Test - Generate ghex binary only${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Configuration
SEG_LEN=2
GHEX_BINARY="/home/kun/smallbinary/ghex"
OUTPUT_DIR="/tmp/ghex_test_output"
SCRIPT_DIR="/home/kun/Document/PalmTree/src/data_generator"

# Create output directories
mkdir -p "${OUTPUT_DIR}/cfg"
mkdir -p "${OUTPUT_DIR}/dfg"

echo "Configuration:"
echo "  Binary: ${GHEX_BINARY}"
echo "  Output: ${OUTPUT_DIR}"
echo "  SEG_LEN: ${SEG_LEN}"
echo ""

# Check if binary exists
if [ ! -f "${GHEX_BINARY}" ]; then
    echo -e "${RED}Error: ghex binary not found at ${GHEX_BINARY}${NC}"
    exit 1
fi

# Generate CFG
echo -e "${BLUE}Generating CFG data...${NC}"
cd "${SCRIPT_DIR}"
python3 -c "
import sys
sys.path.insert(0, '.')
import cfg_address
import __main__
cfg_address.SEG_LEN = ${SEG_LEN}
__main__.OUTPUT_DIR = '${OUTPUT_DIR}/cfg'
cfg_address.process_file('${GHEX_BINARY}')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CFG generation completed${NC}"
    CFG_FILE=$(ls ${OUTPUT_DIR}/cfg/ghex_cfg_${SEG_LEN}_inline.txt 2>/dev/null)
    if [ -f "${CFG_FILE}" ]; then
        LINE_COUNT=$(wc -l < "${CFG_FILE}")
        FILE_SIZE=$(ls -lh "${CFG_FILE}" | awk '{print $5}')
        echo "  File: ${CFG_FILE}"
        echo "  Lines: ${LINE_COUNT}"
        echo "  Size: ${FILE_SIZE}"
    fi
else
    echo -e "${RED}✗ CFG generation failed${NC}"
    exit 1
fi
echo ""

# Generate DFG
echo -e "${BLUE}Generating DFG data...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
import dfg_address
import __main__
dfg_address.SEG_LEN = ${SEG_LEN}
__main__.OUTPUT_DIR = '${OUTPUT_DIR}/dfg'
dfg_address.process_file('${GHEX_BINARY}')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ DFG generation completed${NC}"
    DFG_FILE=$(ls ${OUTPUT_DIR}/dfg/ghex_dfg_${SEG_LEN}_inline.txt 2>/dev/null)
    if [ -f "${DFG_FILE}" ]; then
        LINE_COUNT=$(wc -l < "${DFG_FILE}")
        FILE_SIZE=$(ls -lh "${DFG_FILE}" | awk '{print $5}')
        echo "  File: ${DFG_FILE}"
        echo "  Lines: ${LINE_COUNT}"
        echo "  Size: ${FILE_SIZE}"
    fi
else
    echo -e "${RED}✗ DFG generation failed${NC}"
    exit 1
fi
echo ""

# Check for duplicate hex issue
echo -e "${YELLOW}Checking for duplicate hex values issue...${NC}"
CFG_FILE="${OUTPUT_DIR}/cfg/ghex_cfg_${SEG_LEN}_inline.txt"

if [ -f "${CFG_FILE}" ]; then
    # Count lines with duplicate pattern
    DUP_COUNT=$(grep -c "\[ [a-z0-9]* [+-] 0x.* 0x.* " "${CFG_FILE}" 2>/dev/null || echo "0")
    TOTAL_LINES=$(wc -l < "${CFG_FILE}")
    
    echo "  Total lines: ${TOTAL_LINES}"
    echo "  Lines with duplicate hex pattern: ${DUP_COUNT}"
    
    if [ "${DUP_COUNT}" -gt 0 ]; then
        echo -e "${RED}  ✗ DUPLICATE ISSUE FOUND!${NC}"
        echo ""
        echo "  Sample lines with duplicates:"
        grep "\[ [a-z0-9]* [+-] 0x.* 0x.* " "${CFG_FILE}" | head -3
    else
        echo -e "${GREEN}  ✓ No duplicate hex values found!${NC}"
    fi
fi

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}✓ Test generation complete!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo "Output directory: ${OUTPUT_DIR}"
