#!/bin/bash
#
# Script to generate CFG and DFG data using cfg_address.py and dfg_address.py
#
# Usage: ./generate_data.sh
#

# =============================================================================
# Configuration - Modify these variables as needed
# =============================================================================

# Segment length (number of instructions per line)
SEG_LEN=8

# Binary folder containing binaries to process
BIN_FOLDER="/home/kun/smallbinary"

# Output directories
CFG_OUTPUT_DIR="/home/kun/Document/PalmTree/data/cfg"
DFG_OUTPUT_DIR="/home/kun/Document/PalmTree/data/dfg"

# Path to generator scripts
SCRIPT_DIR="/home/kun/Document/PalmTree/src/data_generator"
CFG_SCRIPT="${SCRIPT_DIR}/cfg_address.py"
DFG_SCRIPT="${SCRIPT_DIR}/dfg_address.py"

# Combined output files (optional - set to "" to disable)
COMBINED_CFG="${CFG_OUTPUT_DIR}/all_cfg_combined.txt"
COMBINED_DFG="${DFG_OUTPUT_DIR}/all_dfg_combined.txt"

# =============================================================================
# Script execution - No need to modify below this line
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  PalmTree Data Generation Script${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Configuration:"
echo "  SEG_LEN:        ${SEG_LEN}"
echo "  BIN_FOLDER:     ${BIN_FOLDER}"
echo "  CFG_OUTPUT_DIR: ${CFG_OUTPUT_DIR}"
echo "  DFG_OUTPUT_DIR: ${DFG_OUTPUT_DIR}"
echo ""

# Check if binary folder exists
if [ ! -d "${BIN_FOLDER}" ]; then
    echo -e "${RED}Error: Binary folder ${BIN_FOLDER} does not exist!${NC}"
    exit 1
fi

# Check if scripts exist
if [ ! -f "${CFG_SCRIPT}" ]; then
    echo -e "${RED}Error: CFG script ${CFG_SCRIPT} not found!${NC}"
    exit 1
fi

if [ ! -f "${DFG_SCRIPT}" ]; then
    echo -e "${RED}Error: DFG script ${DFG_SCRIPT} not found!${NC}"
    exit 1
fi

# Create output directories
mkdir -p "${CFG_OUTPUT_DIR}"
mkdir -p "${DFG_OUTPUT_DIR}"

# Count binaries
BINARY_COUNT=$(find "${BIN_FOLDER}" -type f ! -name "*.txt" | wc -l)
echo -e "${GREEN}Found ${BINARY_COUNT} binaries to process${NC}"
echo ""

# Generate CFG data
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Generating CFG Data${NC}"
echo -e "${BLUE}============================================================${NC}"
cd "${SCRIPT_DIR}"
python3 "${CFG_SCRIPT}" "${SEG_LEN}" "${BIN_FOLDER}" "${CFG_OUTPUT_DIR}"
CFG_EXIT=$?

if [ ${CFG_EXIT} -eq 0 ]; then
    echo -e "${GREEN}✓ CFG generation completed successfully${NC}"
else
    echo -e "${RED}✗ CFG generation failed with exit code ${CFG_EXIT}${NC}"
fi
echo ""

# Generate DFG data
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Generating DFG Data${NC}"
echo -e "${BLUE}============================================================${NC}"
python3 "${DFG_SCRIPT}" "${SEG_LEN}" "${BIN_FOLDER}" "${DFG_OUTPUT_DIR}"
DFG_EXIT=$?

if [ ${DFG_EXIT} -eq 0 ]; then
    echo -e "${GREEN}✓ DFG generation completed successfully${NC}"
else
    echo -e "${RED}✗ DFG generation failed with exit code ${DFG_EXIT}${NC}"
fi
echo ""

# Combine outputs if requested
if [ -n "${COMBINED_CFG}" ] && [ ${CFG_EXIT} -eq 0 ]; then
    echo -e "${YELLOW}Combining CFG outputs...${NC}"
    cat "${CFG_OUTPUT_DIR}"/*_cfg_${SEG_LEN}_inline.txt > "${COMBINED_CFG}" 2>/dev/null || true
    if [ -f "${COMBINED_CFG}" ]; then
        LINE_COUNT=$(wc -l < "${COMBINED_CFG}")
        echo -e "${GREEN}✓ Combined CFG: ${COMBINED_CFG} (${LINE_COUNT} lines)${NC}"
    fi
fi

if [ -n "${COMBINED_DFG}" ] && [ ${DFG_EXIT} -eq 0 ]; then
    echo -e "${YELLOW}Combining DFG outputs...${NC}"
    cat "${DFG_OUTPUT_DIR}"/*_dfg_${SEG_LEN}_inline.txt > "${COMBINED_DFG}" 2>/dev/null || true
    if [ -f "${COMBINED_DFG}" ]; then
        LINE_COUNT=$(wc -l < "${COMBINED_DFG}")
        echo -e "${GREEN}✓ Combined DFG: ${COMBINED_DFG} (${LINE_COUNT} lines)${NC}"
    fi
fi

# Summary
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}============================================================${NC}"

CFG_FILE_COUNT=$(find "${CFG_OUTPUT_DIR}" -name "*_cfg_${SEG_LEN}_inline.txt" -type f | wc -l)
DFG_FILE_COUNT=$(find "${DFG_OUTPUT_DIR}" -name "*_dfg_${SEG_LEN}_inline.txt" -type f | wc -l)

echo "CFG files generated: ${CFG_FILE_COUNT}"
echo "DFG files generated: ${DFG_FILE_COUNT}"
echo ""
echo "Output locations:"
echo "  CFG: ${CFG_OUTPUT_DIR}"
echo "  DFG: ${DFG_OUTPUT_DIR}"

if [ -n "${COMBINED_CFG}" ] && [ -f "${COMBINED_CFG}" ]; then
    echo "  Combined CFG: ${COMBINED_CFG}"
fi

if [ -n "${COMBINED_DFG}" ] && [ -f "${COMBINED_DFG}" ]; then
    echo "  Combined DFG: ${COMBINED_DFG}"
fi

echo ""
if [ ${CFG_EXIT} -eq 0 ] && [ ${DFG_EXIT} -eq 0 ]; then
    echo -e "${GREEN}✓ All operations completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some operations failed. Check output above for details.${NC}"
    exit 1
fi
