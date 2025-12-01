#!/bin/bash
# Run experiment 7 (MLM + NSP-DFG + Address) standalone
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
source "${SCRIPT_DIR}/common_run_env.sh"

START_TIME=$(date +%s)
run_experiment 7 "MLM + NSP-DFG + Address" true false true false true
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Experiment 7 finished in ${ELAPSED}s"
