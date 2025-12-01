#!/bin/bash
# Run experiment 8 (MLM + NSP-CFG + Scope) standalone
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
source "${SCRIPT_DIR}/common_run_env.sh"

START_TIME=$(date +%s)
run_experiment 8 "MLM + NSP-CFG + Scope" true true false true false
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Experiment 8 finished in ${ELAPSED}s"
