#!/bin/bash
# Run experiment 4 (MLM + NSP-CFG) standalone
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
source "${SCRIPT_DIR}/common_run_env.sh"

START_TIME=$(date +%s)
run_experiment 4 "MLM + NSP-CFG" true true false false false
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Experiment 4 finished in ${ELAPSED}s"
