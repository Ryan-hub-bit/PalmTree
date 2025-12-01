#!/bin/bash

# Automated Ablation Study Runner
# Runs all experiments in a specific order to evaluate different task combinations
# 
# Order of experiments:
# 1. MLM + Address
# 2. MLM + Scope (no address)
# 3. MLM + Scope + Address
# 4. MLM + NSP-CFG
# 5. MLM + NSP-CFG + Address
# 6. MLM + NSP-DFG
# 7. MLM + NSP-DFG + Address
# 8. MLM + NSP-CFG + NSP-DFG + Scope
# 9. MLM + NSP-CFG + NSP-DFG + Scope + Address

set -e # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================"
echo "AUTOMATED ABLATION STUDY"
echo "========================================"
echo ""
echo "This script will run 9 experiments:"
echo "  1. MLM + Address"
echo "  2. MLM + Scope"
echo "  3. MLM + Scope + Address"
echo "  4. MLM + NSP-CFG"
echo "  5. MLM + NSP-CFG + Address"
echo "  6. MLM + NSP-DFG"
echo "  7. MLM + NSP-DFG + Address"
echo "  8. MLM + NSP-CFG + NSP-DFG + Scope"
echo "  9. MLM + NSP-CFG + NSP-DFG + Scope + Address"
echo ""
echo "========================================"
echo ""

# Source the common run environment (variables + run_experiment function)
source ../run_scripts/common_run_env.sh

# Record start time
START_TIME=$(date +%s)
echo "Start time: $(date)"
echo ""

## Run experiments 4-9 using per-experiment scripts so they can be run individually
bash ../run_scripts/run_exp_4_mlm_nsp_cfg.sh
bash ../run_scripts/run_exp_5_mlm_nsp_cfg_address.sh
bash ../run_scripts/run_exp_6_mlm_nsp_dfg.sh
bash ../run_scripts/run_exp_7_mlm_nsp_dfg_address.sh
bash ../run_scripts/run_exp_8_mlm_nsp_cfg_scope.sh
bash ../run_scripts/run_exp_9_mlm_nsp_cfg_scope_address.sh

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${GREEN}========================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================${NC}"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "End time: $(date)"
echo ""
echo "Results saved in:"
echo "  - Models: ../output/"
echo "  - Logs: ../log/"
echo ""
echo "Experiments completed:"
echo "  ✓ 1. MLM + Address"
echo "  ✓ 2. MLM + Scope"
echo "  ✓ 3. MLM + Scope + Address"
echo "  ✓ 4. MLM + NSP-CFG"
echo "  ✓ 5. MLM + NSP-CFG + Address"
echo "  ✓ 6. MLM + NSP-DFG"
echo "  ✓ 7. MLM + NSP-DFG + Address"
echo "  ✓ 8. MLM + NSP-CFG + NSP-DFG + Scope"
echo "  ✓ 9. MLM + NSP-CFG + NSP-DFG + Scope + Address"
echo ""
