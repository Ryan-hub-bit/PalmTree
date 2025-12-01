#!/bin/bash
# Run experiment 6 (MLM + NSP-DFG) standalone
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
source "${SCRIPT_DIR}/common_run_env.sh"

START_TIME=$(date +%s)
run_experiment 6 "MLM + NSP-DFG" true false true false false
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Experiment 6 finished in ${ELAPSED}s"
