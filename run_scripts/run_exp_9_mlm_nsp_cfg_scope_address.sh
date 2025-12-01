#!/bin/bash
# Run experiment 9 (MLM + NSP-CFG + Scope + Address) standalone
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
source "${SCRIPT_DIR}/common_run_env.sh"

START_TIME=$(date +%s)
run_experiment 9 "MLM + NSP-CFG + Scope + Address" true true false true true
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Experiment 9 finished in ${ELAPSED}s"
