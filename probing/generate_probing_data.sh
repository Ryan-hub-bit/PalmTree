#!/bin/bash

# Generate Inline Format Test Data from Binaries
# Uses the existing data generation pipeline to create test data
# with inline address format for probing experiments.

echo "========================================================================="
echo "Generate Inline Format Test Data for Probing"
echo "========================================================================="
echo ""

# Configuration
BINARY_DIR="/home/kun/testbinary"
OUTPUT_DIR="/home/kun/Document/PalmTree/probing/test_data"
SRC_DIR="/home/kun/Document/PalmTree/src/data_generator"

# Parameters
SEG_LEN=2  # 2 instructions per line (for probing)
MAX_SAMPLES=5000  # Limit samples per binary

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Configuration:"
echo "  Binary Directory: ${BINARY_DIR}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Segment Length:   ${SEG_LEN} instructions"
echo "  Max Samples:      ${MAX_SAMPLES} per binary"
echo ""

# Check if binaries exist
if [ ! -d "${BINARY_DIR}" ]; then
    echo "ERROR: Binary directory not found: ${BINARY_DIR}"
    exit 1
fi

BINARY_COUNT=$(ls "${BINARY_DIR}" 2>/dev/null | wc -l)
if [ "${BINARY_COUNT}" -eq 0 ]; then
    echo "ERROR: No binaries found in ${BINARY_DIR}"
    exit 1
fi

echo "Found ${BINARY_COUNT} binaries:"
ls -1 "${BINARY_DIR}/"
echo ""

# Process each binary
echo "========================================================================="
echo "Processing Binaries"
echo "========================================================================="
echo ""

cfg_files=()
dfg_files=()

for binary_path in "${BINARY_DIR}"/*; do
    if [ ! -f "${binary_path}" ]; then
        continue
    fi
    
    binary_name=$(basename "${binary_path}")
    echo "Processing: ${binary_name}"
    
    # Output file names
    cfg_output="${OUTPUT_DIR}/${binary_name}_cfg_inline.txt"
    dfg_output="${OUTPUT_DIR}/${binary_name}_dfg_inline.txt"
    
    # Generate CFG data with inline addresses
    echo "  [1/2] Generating CFG data..."
    cd "${SRC_DIR}" || exit 1
    
    python3 cfg_address.py \
        --binary "${binary_path}" \
        --output "${cfg_output}" \
        --seg_len ${SEG_LEN} \
        2>&1 | grep -v "^INFO" | head -20
    
    if [ -f "${cfg_output}" ]; then
        line_count=$(wc -l < "${cfg_output}")
        echo "  ✓ CFG: ${line_count} lines → ${cfg_output}"
        cfg_files+=("${cfg_output}")
    else
        echo "  ✗ CFG generation failed"
    fi
    
    # Generate DFG data with inline addresses
    echo "  [2/2] Generating DFG data..."
    
    python3 dfg_address.py \
        --binary "${binary_path}" \
        --output "${dfg_output}" \
        --seg_len ${SEG_LEN} \
        2>&1 | grep -v "^INFO" | head -20
    
    if [ -f "${dfg_output}" ]; then
        line_count=$(wc -l < "${dfg_output}")
        echo "  ✓ DFG: ${line_count} lines → ${dfg_output}"
        dfg_files+=("${dfg_output}")
    else
        echo "  ✗ DFG generation failed"
    fi
    
    echo ""
done

# Combine all outputs
echo "========================================================================="
echo "Combining Outputs"
echo "========================================================================="
echo ""

if [ ${#cfg_files[@]} -gt 0 ]; then
    combined_cfg="${OUTPUT_DIR}/all_cfg_probing.txt"
    cat "${cfg_files[@]}" > "${combined_cfg}"
    total_lines=$(wc -l < "${combined_cfg}")
    echo "✓ Combined CFG: ${total_lines} lines → ${combined_cfg}"
else
    echo "✗ No CFG files to combine"
fi

if [ ${#dfg_files[@]} -gt 0 ]; then
    combined_dfg="${OUTPUT_DIR}/all_dfg_probing.txt"
    cat "${dfg_files[@]}" > "${combined_dfg}"
    total_lines=$(wc -l < "${combined_dfg}")
    echo "✓ Combined DFG: ${total_lines} lines → ${combined_dfg}"
else
    echo "✗ No DFG files to combine"
fi

echo ""
echo "========================================================================="
echo "Summary"
echo "========================================================================="
echo ""

if [ -f "${OUTPUT_DIR}/all_cfg_probing.txt" ] && [ -f "${OUTPUT_DIR}/all_dfg_probing.txt" ]; then
    echo "✓ Test data generation complete!"
    echo ""
    echo "Output files:"
    echo "  CFG: ${OUTPUT_DIR}/all_cfg_probing.txt"
    echo "  DFG: ${OUTPUT_DIR}/all_dfg_probing.txt"
    echo ""
    echo "To run probing experiment with this data:"
    echo ""
    echo "  cd /home/kun/Document/PalmTree/probing"
    echo "  python probe_positions.py \\"
    echo "    --addressaware_model ../addressaware/output_addressaware_new/best_model.pt \\"
    echo "    --baseline_model ../addressaware/output_baseline_new/best_model.pt \\"
    echo "    --test_cfg ./test_data/all_cfg_probing.txt \\"
    echo "    --test_dfg ./test_data/all_dfg_probing.txt \\"
    echo "    --vocab ../pre-trained_model/palmtree/vocab \\"
    echo "    --output_dir ./results"
    echo ""
else
    echo "✗ Data generation incomplete. Please check errors above."
    exit 1
fi
