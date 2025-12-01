#!/bin/bash
# Wrapper script to generate both CFG and DFG with IDA Pro batch mode
# Automatically cleans up IDA-generated database files

IDA_PATH="/home/kun/ida-pro-9.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_SCRIPT="$SCRIPT_DIR/cfg_hierarchical_icfg_ida.py"
DFG_SCRIPT="$SCRIPT_DIR/dfg_hierarchical_idfg_ida.py"

# Default parameters
SEG_LEN=${1:-8}
BIN_FOLDER=${2:-"/data/kun/addressbin_test"}
OUTPUT_DIR=${3:-"/work/kliu14/txtdataset/test_cdfg"}

export OUTPUT_DIR
export SEG_LEN

echo "=========================================="
echo "IDA Pro CFG + DFG Generation"
echo "=========================================="
echo "IDA Path: $IDA_PATH"
echo "CFG Script: $CFG_SCRIPT"
echo "DFG Script: $DFG_SCRIPT"
echo "SEG_LEN: $SEG_LEN"
echo "Binary Folder: $BIN_FOLDER"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create log file
LOG_FILE="$OUTPUT_DIR/ida_generation_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
echo ""

# Function to log message (both to console and file)
log() {
  echo "$1" | tee -a "$LOG_FILE"
}

# Function to clean up IDA database files
cleanup_ida_files() {
  local binary="$1"
  rm -f "${binary}.idb" "${binary}.i64" "${binary}.id0" "${binary}.id1" "${binary}.id2" "${binary}.nam" "${binary}.til" 2>/dev/null
}

# Count total binaries
total=$(find "$BIN_FOLDER" -maxdepth 1 -type f ! -name "*.txt" ! -name "*.idb" ! -name "*.i64" ! -name "*.id0" ! -name "*.id1" ! -name "*.id2" ! -name "*.nam" ! -name "*.til" | wc -l)

if [ "$total" -eq 0 ]; then
  log "[ERROR] No binary files found in $BIN_FOLDER"
  exit 1
fi

log "[INFO] Found $total binaries to process"
log ""

# Process each binary file
count=0

for binary in "$BIN_FOLDER"/*; do
  # Skip non-files
  if [ ! -f "$binary" ]; then
    continue
  fi

  basename=$(basename "$binary")

  # Skip text files and IDA database files
  if [[ "$basename" == *.txt ]] ||
    [[ "$basename" == *.idb ]] ||
    [[ "$basename" == *.i64 ]] ||
    [[ "$basename" == *.id0 ]] ||
    [[ "$basename" == *.id1 ]] ||
    [[ "$basename" == *.id2 ]] ||
    [[ "$basename" == *.nam ]] ||
    [[ "$basename" == *.til ]]; then
    continue
  fi

  count=$((count + 1))
  log "======================================================================"
  log "Processing $count/$total: $basename"
  log "======================================================================"

  # Start timing
  start_time=$(date +%s)

  # Check if output files already exist
  cfg_file="$OUTPUT_DIR/${basename}_cfg_${SEG_LEN}_inline.txt"
  dfg_file="$OUTPUT_DIR/${basename}_dfg_${SEG_LEN}_inline.txt"

  cfg_exists=false
  dfg_exists=false

  if [ -f "$cfg_file" ]; then
    cfg_exists=true
    log "[INFO] CFG file already exists, skipping CFG generation"
  fi

  if [ -f "$dfg_file" ]; then
    dfg_exists=true
    log "[INFO] DFG file already exists, skipping DFG generation"
  fi

  # Skip if both files exist
  if [ "$cfg_exists" = true ] && [ "$dfg_exists" = true ]; then
    log "[SKIP] Both CFG and DFG already generated for $basename"
    log ""
    continue
  fi

  # ==========================================
  # Step 1: Generate CFG (if not exists)
  # ==========================================
  cfg_time=0
  if [ "$cfg_exists" = false ]; then
    log "[INFO] Generating CFG for $basename..."
    cfg_start=$(date +%s)
    "$IDA_PATH/idat" -A -S"$CFG_SCRIPT" "$binary" >/dev/null 2>&1
    cfg_end=$(date +%s)
    cfg_time=$((cfg_end - cfg_start))

    if [ $? -eq 0 ]; then
      log "[SUCCESS] CFG generation completed in ${cfg_time}s"
    else
      log "[WARNING] CFG generation may have issues (${cfg_time}s)"
    fi

    # Clean up IDA database files after CFG
    cleanup_ida_files "$binary"
  fi

  # ==========================================
  # Step 2: Generate DFG (if not exists)
  # ==========================================
  dfg_time=0
  if [ "$dfg_exists" = false ]; then
    log "[INFO] Generating DFG for $basename..."
    dfg_start=$(date +%s)
    "$IDA_PATH/idat" -A -S"$DFG_SCRIPT" "$binary" >/dev/null 2>&1
    dfg_end=$(date +%s)
    dfg_time=$((dfg_end - dfg_start))

    if [ $? -eq 0 ]; then
      log "[SUCCESS] DFG generation completed in ${dfg_time}s"
    else
      log "[WARNING] DFG generation may have issues (${dfg_time}s)"
    fi

    # Clean up IDA database files after DFG
    cleanup_ida_files "$binary"
  fi

  # Verify output files were created
  cfg_file="$OUTPUT_DIR/${basename}_cfg_${SEG_LEN}_inline.txt"
  dfg_file="$OUTPUT_DIR/${basename}_dfg_${SEG_LEN}_inline.txt"

  if [ -f "$cfg_file" ]; then
    cfg_lines=$(wc -l <"$cfg_file")
    log "  → CFG: $cfg_lines sequences"
  else
    log "  → CFG: [MISSING]"
  fi

  if [ -f "$dfg_file" ]; then
    dfg_lines=$(wc -l <"$dfg_file")
    log "  → DFG: $dfg_lines sequences"
  else
    log "  → DFG: [MISSING]"
  fi

  # Calculate and display total time
  end_time=$(date +%s)
  total_time=$((end_time - start_time))
  log "  → Total time: ${total_time}s (CFG: ${cfg_time}s, DFG: ${dfg_time}s)"

  log ""
done

log "======================================================================"
log "All binaries processed!"
log "======================================================================"
log "Output location: $OUTPUT_DIR"
log ""

# Summary statistics
cfg_count=$(find "$OUTPUT_DIR" -name "*_cfg_${SEG_LEN}_inline.txt" -type f | wc -l)
dfg_count=$(find "$OUTPUT_DIR" -name "*_dfg_${SEG_LEN}_inline.txt" -type f | wc -l)
total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)

log "Summary:"
log "  CFG files generated: $cfg_count"
log "  DFG files generated: $dfg_count"
log "  Total output size: ${total_size:-N/A}"
log "======================================================================"
log "Log saved to: $LOG_FILE"
