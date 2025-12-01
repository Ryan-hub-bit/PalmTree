#!/bin/bash
# Run experiment 5 (MLM + NSP-CFG + Address) standalone
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
source "${SCRIPT_DIR}/common_run_env.sh"

START_TIME=$(date +%s)
run_experiment 5 "MLM + NSP-CFG + Address" true true false false true
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Experiment 5 finished in ${ELAPSED}s"
