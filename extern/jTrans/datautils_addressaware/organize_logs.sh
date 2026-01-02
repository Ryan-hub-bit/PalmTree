#!/bin/bash
# Script to organize existing log files into a separate directory

set -e  # Exit on error

# Default directory (current location based on your terminal output)
TARGET_DIR="${1:-/data/kun/jtransdata/addr_extract}"

echo "=================================================="
echo "Log File Organization Script"
echo "=================================================="
echo "Target directory: $TARGET_DIR"
echo ""

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory does not exist: $TARGET_DIR"
    exit 1
fi

# Create logs subdirectory
LOG_DIR="$TARGET_DIR/logs"
mkdir -p "$LOG_DIR"
echo "Created logs directory: $LOG_DIR"

# Count existing files
PKL_COUNT=$(find "$TARGET_DIR" -maxdepth 1 -name "*_extract.pkl" | wc -l)
TXT_COUNT=$(find "$TARGET_DIR" -maxdepth 1 -name "*_addressaware.txt" | wc -l)
LOG_COUNT=$(find "$TARGET_DIR" -maxdepth 1 -name "*_ida.log" | wc -l)

echo ""
echo "Current file counts:"
echo "  PKL files: $PKL_COUNT"
echo "  TXT files: $TXT_COUNT"
echo "  LOG files (to move): $LOG_COUNT"
echo ""

# Move log files if any exist
if [ $LOG_COUNT -gt 0 ]; then
    echo "Moving log files to $LOG_DIR..."
    mv "$TARGET_DIR"/*_ida.log "$LOG_DIR/"
    echo "✓ Moved $LOG_COUNT log files"
else
    echo "No log files found to move"
fi

# Verify the move
MOVED_COUNT=$(find "$LOG_DIR" -name "*_ida.log" | wc -l)
REMAINING_COUNT=$(find "$TARGET_DIR" -maxdepth 1 -name "*_ida.log" | wc -l)

echo ""
echo "=================================================="
echo "Organization Complete!"
echo "=================================================="
echo "Log files in logs directory: $MOVED_COUNT"
echo "Log files remaining in main directory: $REMAINING_COUNT"
echo ""
echo "Directory structure:"
echo "  Data files: $TARGET_DIR/*.{pkl,txt}"
echo "  Log files:  $LOG_DIR/*.log"
echo ""

# Show disk usage
echo "Disk usage summary:"
echo "-------------------"
PKL_SIZE=$(du -sh "$TARGET_DIR"/*.pkl 2>/dev/null | awk '{sum+=$1} END {print sum}' || echo "0")
TXT_SIZE=$(du -sh "$TARGET_DIR"/*.txt 2>/dev/null | awk '{sum+=$1} END {print sum}' || echo "0")
LOG_SIZE=$(du -sh "$LOG_DIR"/*.log 2>/dev/null | awk '{sum+=$1} END {print sum}' || echo "0")

echo "  PKL files: $(du -sh "$TARGET_DIR"/*.pkl 2>/dev/null | awk '{s+=$1} END {print s}' || echo 0) (total)"
echo "  TXT files: $(du -sh "$TARGET_DIR"/*.txt 2>/dev/null | awk '{s+=$1} END {print s}' || echo 0) (total)"  
echo "  LOG files: $(du -sh "$LOG_DIR" 2>/dev/null | cut -f1 || echo 0)"

echo ""
echo "✓ Done!"
