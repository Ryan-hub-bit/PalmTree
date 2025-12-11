#!/bin/bash
#
# Complete Function Similarity Pipeline
#
# This script:
# 1. Fine-tunes AddressAwareBERT on function similarity task using MLM pre-trained model
# 2. Evaluates the fine-tuned model on test set
#
# Pre-trained model: output/mlm/best_bert.pt
# Data: /data/kun/funcsim_match/
#
# Usage: ./run_funcsim_pipeline.sh [--skip-train] [--skip-eval]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# Parse arguments
SKIP_TRAIN=false
SKIP_EVAL=false

for arg in "$@"; do
    case $arg in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_funcsim_pipeline.sh [--skip-train] [--skip-eval]"
            echo ""
            echo "Options:"
            echo "  --skip-train    Skip training (use existing model)"
            echo "  --skip-eval     Skip evaluation"
            echo ""
            echo "Pre-trained BERT: output/mlm/best_bert.pt"
            echo "Data directory: /data/kun/funcsim_match/"
            exit 0
            ;;
        *)
            echo "[WARNING] Unknown argument: $arg"
            ;;
    esac
done

# Print banner
echo "================================================================================"
echo "                  Function Similarity Fine-tuning Pipeline"
echo "================================================================================"
echo ""
echo "Pipeline Steps:"
echo "  1. Train:    $([ "$SKIP_TRAIN" = true ] && echo "SKIP" || echo "RUN")"
echo "  2. Evaluate: $([ "$SKIP_EVAL" = true ] && echo "SKIP" || echo "RUN")"
echo ""
echo "Pre-trained BERT: ${PROJECT_ROOT}/output/mlm/best_bert.pt"
echo "Data directory:   /data/kun/funcsim_match/"
echo "Output directory: ${PROJECT_ROOT}/output/funcsim/"
echo "================================================================================"
echo ""

# Track timing
PIPELINE_START=$(date +%s)

# =============================================================================
# STEP 1: Fine-tune for function similarity
# =============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 1/2: Fine-tuning AddressAwareBERT for Function Similarity"
    echo "================================================================================"
    STEP_START=$(date +%s)
    
    # Check if pre-trained model exists
    if [ ! -f "${PROJECT_ROOT}/output/mlm/best_bert.pt" ]; then
        echo "[ERROR] Pre-trained BERT not found: ${PROJECT_ROOT}/output/mlm/best_bert.pt"
        echo "[INFO] Please train the MLM model first"
        exit 1
    fi
    
    # Check if data files exist
    if [ ! -f "/data/kun/funcsim_match/function_blocks.json" ]; then
        echo "[ERROR] Function blocks not found: /data/kun/funcsim_match/function_blocks.json"
        echo "[INFO] Please run: cd ${PROJECT_ROOT}/src/data_generator/funcsim && ./run_funcsim_pipeline.sh"
        exit 1
    fi
    
    if [ ! -f "/data/kun/funcsim_match/funcsim_pairs.json" ]; then
        echo "[ERROR] Funcsim pairs not found: /data/kun/funcsim_match/funcsim_pairs.json"
        echo "[INFO] Please run: cd ${PROJECT_ROOT}/src/data_generator/funcsim && ./run_funcsim_pipeline.sh"
        exit 1
    fi
    
    # Run training
    bash "${SCRIPT_DIR}/run_funcsim_train.sh"
    
    STEP_END=$(date +%s)
    echo ""
    echo "[TIMING] Training completed in $((STEP_END - STEP_START)) seconds"
else
    echo ""
    echo "================================================================================"
    echo "STEP 1/2: SKIPPED (--skip-train)"
    echo "================================================================================"
fi

# =============================================================================
# STEP 2: Evaluate on test set
# =============================================================================
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 2/2: Evaluating on Test Set"
    echo "================================================================================"
    STEP_START=$(date +%s)
    
    # Check if model exists
    if [ ! -f "${PROJECT_ROOT}/output/funcsim/best_model.pt" ]; then
        echo "[ERROR] Trained model not found: ${PROJECT_ROOT}/output/funcsim/best_model.pt"
        echo "[INFO] Please train the model first (remove --skip-train)"
        exit 1
    fi
    
    # Check if test indices exist
    if [ ! -f "${PROJECT_ROOT}/output/funcsim/test_indices.json" ]; then
        echo "[ERROR] Test indices not found: ${PROJECT_ROOT}/output/funcsim/test_indices.json"
        echo "[INFO] Test indices are created during training"
        exit 1
    fi
    
    # Run evaluation
    bash "${SCRIPT_DIR}/run_eval.sh"
    
    STEP_END=$(date +%s)
    echo ""
    echo "[TIMING] Evaluation completed in $((STEP_END - STEP_START)) seconds"
else
    echo ""
    echo "================================================================================"
    echo "STEP 2/2: SKIPPED (--skip-eval)"
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

if [ "$SKIP_TRAIN" = false ]; then
    echo "Model saved to:"
    echo "  - ${PROJECT_ROOT}/output/funcsim/best_model.pt"
    echo "  - ${PROJECT_ROOT}/output/funcsim/checkpoint_latest.pt"
    echo ""
fi

if [ "$SKIP_EVAL" = false ]; then
    echo "Evaluation results saved to:"
    echo "  - ${PROJECT_ROOT}/output/funcsim/test_results.json"
    echo ""
fi

echo "Logs saved to:"
echo "  - ${PROJECT_ROOT}/log/funcsim/"
echo ""
echo "================================================================================"
