#!/bin/bash
#
# Complete Function Similarity Data Generation Pipeline
#
# This script runs the entire funcsim data generation pipeline:
# 1. generate_funcsim.sh - Extract function data from binaries using IDA Pro
# 2. combine_funcsim.sh - Combine and deduplicate the extracted data
# 3. check_special_tokens.py - Check for tokens not in vocabulary
# 4. generate_pairs.sh - Generate function similarity pairs (with filtering)
#
# Usage: ./run_funcsim_pipeline.sh [--skip-ida] [--skip-combine] [--skip-check] [--skip-pairs]
#
# Options:
#   --skip-ida      Skip IDA extraction (use existing JSON files)
#   --skip-combine  Skip combine/deduplication step
#   --skip-check    Skip vocabulary check step
#   --skip-pairs    Skip pairs generation step
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/data/kun/funcsim_match"
VOCAB_FILE="/home/kun/Document/PalmTree/strupos/vocab_mapping.txt"

# Parse arguments
SKIP_IDA=false
SKIP_COMBINE=false
SKIP_CHECK=false
SKIP_PAIRS=false

for arg in "$@"; do
    case $arg in
        --skip-ida)
            SKIP_IDA=true
            shift
            ;;
        --skip-combine)
            SKIP_COMBINE=true
            shift
            ;;
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        --skip-pairs)
            SKIP_PAIRS=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_funcsim_pipeline.sh [--skip-ida] [--skip-combine] [--skip-check] [--skip-pairs]"
            echo ""
            echo "Options:"
            echo "  --skip-ida      Skip IDA extraction (use existing JSON files)"
            echo "  --skip-combine  Skip combine/deduplication step"
            echo "  --skip-check    Skip vocabulary check step"
            echo "  --skip-pairs    Skip pairs generation step"
            exit 0
            ;;
        *)
            echo "[WARNING] Unknown argument: $arg"
            ;;
    esac
done

# Print banner
echo "================================================================================"
echo "                  Function Similarity Data Generation Pipeline"
echo "================================================================================"
echo ""
echo "Pipeline Steps:"
echo "  1. IDA Extraction:    $([ "$SKIP_IDA" = true ] && echo "SKIP" || echo "RUN")"
echo "  2. Combine/Dedup:     $([ "$SKIP_COMBINE" = true ] && echo "SKIP" || echo "RUN")"
echo "  3. Vocab Check:       $([ "$SKIP_CHECK" = true ] && echo "SKIP" || echo "RUN")"
echo "  4. Generate Pairs:    $([ "$SKIP_PAIRS" = true ] && echo "SKIP" || echo "RUN")"
echo ""
echo "Directories:"
echo "  Script Dir: $SCRIPT_DIR"
echo "  Data Dir:   $DATA_DIR"
echo "  Vocab File: $VOCAB_FILE"
echo "================================================================================"
echo ""

# Track timing
PIPELINE_START=$(date +%s)

# =============================================================================
# STEP 1: Generate funcsim data from binaries using IDA Pro
# =============================================================================
if [ "$SKIP_IDA" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 1/4: Extracting function data from binaries (IDA Pro)"
    echo "================================================================================"
    STEP_START=$(date +%s)
    
    if [ -f "${SCRIPT_DIR}/generate_funcsim.sh" ]; then
        bash "${SCRIPT_DIR}/generate_funcsim.sh"
    else
        echo "[ERROR] generate_funcsim.sh not found at ${SCRIPT_DIR}"
        exit 1
    fi
    
    STEP_END=$(date +%s)
    echo "[TIMING] Step 1 completed in $((STEP_END - STEP_START)) seconds"
else
    echo ""
    echo "================================================================================"
    echo "STEP 1/4: SKIPPED (--skip-ida)"
    echo "================================================================================"
fi

# =============================================================================
# STEP 2: Combine and deduplicate the extracted data
# =============================================================================
if [ "$SKIP_COMBINE" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 2/4: Combining and deduplicating function data"
    echo "================================================================================"
    STEP_START=$(date +%s)
    
    if [ -f "${SCRIPT_DIR}/combine_funcsim.sh" ]; then
        bash "${SCRIPT_DIR}/combine_funcsim.sh"
    else
        echo "[ERROR] combine_funcsim.sh not found at ${SCRIPT_DIR}"
        exit 1
    fi
    
    STEP_END=$(date +%s)
    echo "[TIMING] Step 2 completed in $((STEP_END - STEP_START)) seconds"
else
    echo ""
    echo "================================================================================"
    echo "STEP 2/4: SKIPPED (--skip-combine)"
    echo "================================================================================"
fi

# =============================================================================
# STEP 3: Check for tokens not in vocabulary
# =============================================================================
if [ "$SKIP_CHECK" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 3/4: Checking for out-of-vocabulary tokens"
    echo "================================================================================"
    STEP_START=$(date +%s)
    
    if [ -f "${SCRIPT_DIR}/check_special_tokens.py" ]; then
        python3 "${SCRIPT_DIR}/check_special_tokens.py" "$DATA_DIR" "$VOCAB_FILE"
    else
        echo "[ERROR] check_special_tokens.py not found at ${SCRIPT_DIR}"
        exit 1
    fi
    
    STEP_END=$(date +%s)
    echo "[TIMING] Step 3 completed in $((STEP_END - STEP_START)) seconds"
else
    echo ""
    echo "================================================================================"
    echo "STEP 3/4: SKIPPED (--skip-check)"
    echo "================================================================================"
fi

# =============================================================================
# STEP 4: Generate function similarity pairs
# =============================================================================
if [ "$SKIP_PAIRS" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 4/4: Generating function similarity pairs"
    echo "================================================================================"
    STEP_START=$(date +%s)
    
    if [ -f "${SCRIPT_DIR}/generate_pairs.sh" ]; then
        bash "${SCRIPT_DIR}/generate_pairs.sh"
    else
        echo "[ERROR] generate_pairs.sh not found at ${SCRIPT_DIR}"
        exit 1
    fi
    
    STEP_END=$(date +%s)
    echo "[TIMING] Step 4 completed in $((STEP_END - STEP_START)) seconds"
else
    echo ""
    echo "================================================================================"
    echo "STEP 4/4: SKIPPED (--skip-pairs)"
    echo "================================================================================"
fi

# =============================================================================
# Summary
# =============================================================================
PIPELINE_END=$(date +%s)
TOTAL_TIME=$((PIPELINE_END - PIPELINE_START))
TOTAL_MINUTES=$((TOTAL_TIME / 60))
TOTAL_SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "================================================================================"
echo "                         Pipeline Complete!"
echo "================================================================================"
echo ""
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "Output files in $DATA_DIR:"
echo "  - *_funcsim.json           (per-binary function data)"
echo "  - combined_deduplicated.json (all functions deduplicated)"
echo "  - missing_tokens_report.json (vocabulary analysis)"
echo "  - function_blocks.json     (function ID -> instructions)"
echo "  - funcsim_pairs.json       (function similarity ground truth)"
echo ""
echo "================================================================================"
