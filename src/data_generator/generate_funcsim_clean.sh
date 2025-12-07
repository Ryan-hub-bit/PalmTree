#!/bin/bash
#
# Generate Function Similarity JSON with CLEAN instruction format using IDA Pro
# 
# This script processes all binaries in the funcsim dataset and generates
# JSON files with clean, raw disassembly (no transformations).
#
# Directory structure expected:
#   /data/kun/funcsim_dataset/
#     openssl_O0/          <- contains: openssl, libcrypto.so.3, libssl.so.3, etc.
#     openssl_O1/          <- contains: openssl, libcrypto.so.3, libssl.so.3, etc.
#     openssl_O2/          <- contains: openssl, libcrypto.so.3, libssl.so.3, etc.
#     openssl_O3/          <- contains: openssl, libcrypto.so.3, libssl.so.3, etc.
#
# Output structure:
#   /data/kun/funcsim_clean/
#     openssl/             <- output directory
#       openssl.json
#       libcrypto.so.3.json
#       libssl.so.3.json
#       ...
#
# Usage: ./generate_funcsim_clean.sh [dataset_path]
# Example: ./generate_funcsim_clean.sh /data/kun/funcsim_dataset
#

set -e

# Configuration
IDA_PATH="/home/kun/ida-pro-9.0/idat"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IDA_SCRIPT="${SCRIPT_DIR}/generate_funcsim_clean_ida.py"
DATASET_PATH="${1:-/data/kun/funcsim_dataset}"
OUTPUT_BASE="/data/kun/funcsim_clean"

# Optimization levels to process
OPT_LEVELS=("O0" "O1" "O2" "O3")

echo "================================================"
echo "Generate Function Similarity JSON (Clean Format)"
echo "================================================"
echo "Dataset path: ${DATASET_PATH}"
echo "IDA script: ${IDA_SCRIPT}"
echo "Output base: ${OUTPUT_BASE}"
echo "================================================"

# Find all project directories by looking at folder patterns
# Folders are named like: project_O0, project_O1, etc.
# Extract unique project names (e.g., "openssl" from "openssl_O0")
PROJECTS=$(ls -d ${DATASET_PATH}/*_O0 2>/dev/null | xargs -n1 basename | sed 's/_O0$//' | sort -u)

if [ -z "$PROJECTS" ]; then
    echo "[ERROR] No project directories found in ${DATASET_PATH}"
    exit 1
fi

echo "[INFO] Found projects:"
for project in $PROJECTS; do
    echo "  - $project"
done
echo ""

# Process each project
for PROJECT_NAME in $PROJECTS; do
    echo "================================================"
    echo "[INFO] Processing project: ${PROJECT_NAME}"
    echo "================================================"
    
    # Create output directory: /data/kun/funcsim_clean/{project}/
    OUTPUT_DIR="${OUTPUT_BASE}/${PROJECT_NAME}"
    mkdir -p "${OUTPUT_DIR}"
    
    # Find all binary files that exist in ALL optimization levels
    echo "[INFO] Finding common binaries across all optimization levels..."
    
    # Get list of files from O0 (excluding .i64 IDA database files)
    O0_DIR="${DATASET_PATH}/${PROJECT_NAME}_O0"
    if [ ! -d "$O0_DIR" ]; then
        echo "[WARNING] O0 directory not found: $O0_DIR"
        continue
    fi
    
    # Find binaries that exist in all 4 optimization levels (only ELF files)
    COMMON_BINARIES=""
    for binary_file in $(ls "$O0_DIR"); do
        binary_path="${O0_DIR}/${binary_file}"
        
        # Skip non-regular files
        if [ ! -f "$binary_path" ]; then
            continue
        fi
        
        # Only process ELF files (skip .i64 IDA databases, .log files, etc.)
        if ! file "$binary_path" 2>/dev/null | grep -q "ELF"; then
            echo "[SKIP] Not an ELF file: $binary_file"
            continue
        fi
        
        # Check if this binary exists in all optimization levels
        exists_in_all=true
        for opt in "${OPT_LEVELS[@]}"; do
            opt_dir="${DATASET_PATH}/${PROJECT_NAME}_${opt}"
            if [ ! -f "${opt_dir}/${binary_file}" ]; then
                exists_in_all=false
                break
            fi
        done
        
        if $exists_in_all; then
            COMMON_BINARIES="${COMMON_BINARIES} ${binary_file}"
        fi
    done
    
    if [ -z "$COMMON_BINARIES" ]; then
        echo "[WARNING] No common binaries found for project: ${PROJECT_NAME}"
        continue
    fi
    
    echo "[INFO] Common binaries found:"
    for binary in $COMMON_BINARIES; do
        echo "    - $binary"
    done
    echo ""
    
    # Process each binary
    for BINARY_NAME in $COMMON_BINARIES; do
        # Create JSON filename (replace special chars)
        JSON_NAME=$(echo "$BINARY_NAME" | sed 's/[^a-zA-Z0-9._-]/_/g')
        OUTPUT_FILE="${OUTPUT_DIR}/${JSON_NAME}.json"
        
        echo "------------------------------------------------"
        echo "[INFO] Processing binary: ${BINARY_NAME}"
        echo "[INFO] Output file: ${OUTPUT_FILE}"
        echo "------------------------------------------------"
        
        # Remove existing JSON to start fresh
        if [ -f "${OUTPUT_FILE}" ]; then
            echo "[INFO] Removing existing JSON to regenerate: ${OUTPUT_FILE}"
            rm -f "${OUTPUT_FILE}"
        fi
        
        # Process each optimization level
        for opt in "${OPT_LEVELS[@]}"; do
            BINARY_PATH="${DATASET_PATH}/${PROJECT_NAME}_${opt}/${BINARY_NAME}"
            
            if [ ! -f "${BINARY_PATH}" ]; then
                echo "[WARNING] Binary not found: ${BINARY_PATH}"
                continue
            fi
            
            echo "[INFO] Processing ${opt}: ${BINARY_PATH}"
            
            # Set environment variable for output directory
            export OUTPUT_DIR="${OUTPUT_DIR}"
            
            # Run IDA in batch mode
            LOG_FILE="/tmp/ida_funcsim_clean_${PROJECT_NAME}_${JSON_NAME}_${opt}.log"
            
            "${IDA_PATH}" -A -S"${IDA_SCRIPT}" -L"${LOG_FILE}" "${BINARY_PATH}" || {
                echo "[WARNING] IDA failed for ${BINARY_PATH}, check ${LOG_FILE}"
                continue
            }
            
            echo "[INFO] Finished processing ${opt}"
        done
        
        # Post-process: Remove functions that don't appear in all 4 optimization levels
        if [ -f "${OUTPUT_FILE}" ]; then
            echo "[INFO] Post-processing: Filtering functions without all four opt levels..."
            
            python3 << PYEOF
import json

try:
    with open('${OUTPUT_FILE}', 'r') as f:
        data = json.load(f)
except:
    print("[WARNING] Could not read JSON file")
    exit(0)

opt_levels = ['O0', 'O1', 'O2', 'O3']

# Filter to keep only functions with all 4 optimization levels
filtered_data = {}
removed_count = 0

for func_name, opt_data in data.items():
    has_all = all(opt in opt_data for opt in opt_levels)
    if has_all:
        filtered_data[func_name] = opt_data
    else:
        removed_count += 1

# Save filtered JSON
with open('${OUTPUT_FILE}', 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"[INFO] Kept {len(filtered_data)} functions with all 4 opt levels")
print(f"[INFO] Removed {removed_count} incomplete functions")
PYEOF
            
            echo "[DONE] ${JSON_NAME}.json created"
        fi
        
        echo ""
    done
done

echo "================================================"
echo "[DONE] All projects and binaries processed"
echo "================================================"

# Post-process: Remove short functions and common functions across all JSON files
echo ""
echo "=== Post-processing ==="
for PROJECT_NAME in $PROJECTS; do
    OUTPUT_DIR="${OUTPUT_BASE}/${PROJECT_NAME}"
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "[INFO] Post-processing ${PROJECT_NAME}..."
        python3 "${SCRIPT_DIR}/postprocess_funcsim.py" "${OUTPUT_DIR}"
    fi
done

# Show summary
echo ""
echo "=== Final Output Summary ==="
for PROJECT_NAME in $PROJECTS; do
    OUTPUT_DIR="${OUTPUT_BASE}/${PROJECT_NAME}"
    if [ -d "${OUTPUT_DIR}" ]; then
        echo ""
        echo "Project: ${PROJECT_NAME}/"
        for json_file in $(ls "${OUTPUT_DIR}"/*.json 2>/dev/null); do
            if [ -f "$json_file" ]; then
                FUNC_COUNT=$(python3 -c "import json; print(len(json.load(open('${json_file}'))))" 2>/dev/null || echo "?")
                FILE_SIZE=$(du -h "${json_file}" | cut -f1)
                echo "  $(basename $json_file): ${FUNC_COUNT} functions, ${FILE_SIZE}"
            fi
        done
    fi
done

echo ""
echo "=== All Done ==="
