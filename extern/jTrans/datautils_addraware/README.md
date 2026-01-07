# Address-Aware Function Export for jTrans

This directory contains scripts to export entire functions from binary files with hierarchical position encoding and address-aware features.

## Files

- **cfg_function_export_ida.py**: IDA Pro script that exports entire functions with hierarchical position encoding
- **run_ida_function_export.sh**: Shell script to run the IDA Pro processing
- **combine_function_files.py**: Python script to merge function files from train/val/test directories
- **run_combine_functions.sh**: Shell script to run the combination process
- **output/**: Directory where processed function data is saved

## Features

### Hierarchical Position Encoding

The script uses a 3-level hierarchical position encoding scheme:

#### For CODE addresses (instructions):
- **Position 1**: Function's position in binary = `(func_start - min_addr) / (max_addr - min_addr)`
- **Position 2**: Basic block's position in function = `(bb_start - func_start) / (func_end - func_start)`
- **Position 3**: Instruction's position in basic block = `(inst_addr - bb_start) / (bb_end - bb_start)`

#### For DATA addresses (not in text section):
- **Position 1**: Section's position in binary = `(section_start - min_addr) / (max_addr - min_addr)`
- **Position 2**: Address position inside section = `(addr - section_start) / (section_end - section_start)`
- **Position 3**: 0.0 (no BB context for data)

### Address-Aware Features

- **address(0xXXXX:pos1:pos2:pos3)**: Code/data addresses with hierarchical positions
- **var(0xXX)**: Stack variables (e.g., local variables, function arguments)
- **imm**: Immediate values (constants that aren't addresses)
- **disp**: Displacement operands (memory offsets)

### Output Format

Each line represents one complete function:
```
opcode1(0xaddr1:pos1:pos2:pos3) operand1 operand2	opcode2(0xaddr2:pos1:pos2:pos3) operand1 operand2	...
```

Instructions within a function are separated by tabs (`\t`).

## Usage

### Basic Usage

```bash
# Make the script executable
chmod +x run_ida_function_export.sh

# Process a single binary
./run_ida_function_export.sh /path/to/binary

# Process a binary with custom output directory
./run_ida_function_export.sh /path/to/binary /custom/output/dir
```

### Batch Processing

To process multiple binaries:

```bash
# Process all binaries in a directory
for binary in /path/to/binaries/*; do
    ./run_ida_function_export.sh "$binary"
done
```

## Smart Merge (Combining Function Files)

After processing multiple binaries, you can combine the function export files from different splits (train/val/test) into unified files.

### Directory Structure for Merging

The merge script expects this structure:
```
/data/kun/dataset/
├── train_functions/    # Put all training binary outputs here
│   ├── binary1_functions.txt
│   ├── binary2_functions.txt
│   └── ...
├── val_functions/      # Put all validation binary outputs here
│   └── ...
└── test_functions/     # Put all test binary outputs here
    └── ...
```

### Running the Merge

```bash
# Make the merge script executable
chmod +x run_combine_functions.sh

# Run the merge
./run_combine_functions.sh
```

### Merge Configuration

The merge script:
- **Maintains 0.8:0.1:0.1 ratio** for train:val:test splits
- **Samples from each directory separately** (no cross-contamination)
- **Uses chunked random sampling** for fair representation
- **Default train sampling**: 100% (use all functions, can be adjusted in combine_function_files.py)
- **Smart merge enabled**: Keeps only top 100 symbols, replaces others with `.plt` to reduce vocabulary size

### Smart Merge Feature

The smart_merge feature reduces vocabulary size by:
1. **Analyzing all files** to find the top 100 most common symbols (e.g., `.plt`, `.got`, `.text`, etc.)
2. **Keeping top symbols** unchanged in the output
3. **Replacing rare symbols** with `.plt` to prevent vocabulary explosion
4. **Saving symbol list** to `top_symbols.txt` for reference

This is especially important for function similarity tasks where rare symbols like library-specific PLT entries can create noise.

### Output Files

After merging, you'll get:
- `/data/kun/dataset/train_functions.txt` - Combined training functions
- `/data/kun/dataset/val_functions.txt` - Combined validation functions  
- `/data/kun/dataset/test_functions.txt` - Combined test functions
- `/data/kun/dataset/top_symbols.txt` - List of top 100 symbols kept (if smart_merge enabled)

### Customizing Merge Paths

Edit the paths in [combine_function_files.py](combine_function_files.py):
```python
# Directory paths (lines 109-111)
train_dir = "/data/kun/dataset/train_functions"
val_dir = "/data/kun/dataset/val_functions"
test_dir = "/data/kun/dataset/test_functions"

# Output paths (line 114)
output_dir = "/data/kun/dataset"

# Train sampling ratio (line 126)
TRAIN_SAMPLE_RATIO = 1.0  # 1.0 = use all functions, 0.03 = 3% sampling

# Smart merge settings (lines 129-130)
TOP_N_SYMBOLS = 100  # Number of top symbols to keep
ENABLE_SMART_MERGE = True  # Set to False to disable symbol replacement
```

### Output

The script creates:
- `<binary_name>_functions.txt`: One function per line with all instructions
- `ida_processing.log`: Processing log with debug information

## Requirements

- IDA Pro 9.0 installed at `/home/kun/ida-pro-9.0/`
- Conda environment with NetworkX: `palmtree` at `/home/kun/anaconda3/envs/palmtree/`
- Python packages: `networkx`

## Differences from PalmTree Version

This version differs from the PalmTree CFG generator in the following ways:

1. **No Sequence Chunking**: Exports entire functions instead of fixed-length random walk sequences
2. **One Function Per Line**: Each line contains a complete function (all basic blocks in order)
3. **No ICFG Walks**: Doesn't perform random walks across the inter-procedural CFG
4. **Simplified Output**: Focused on whole-function export for function similarity tasks

## Example Output

```
mov(0x1000:0.10000000:0.00000000:0.00000000) [ rbp ] imm	push(0x1001:0.10000000:0.00000000:0.50000000) rbp	...
call(0x2000:0.50000000:0.00000000:0.00000000) address(0x3000:0.75000000:0.00000000:0.00000000)	ret(0x2005:0.50000000:0.00000000:1.00000000)
```

Each line represents a complete function with all its instructions.
