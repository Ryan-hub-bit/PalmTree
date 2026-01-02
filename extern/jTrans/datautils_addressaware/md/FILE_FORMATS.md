# File Format Documentation

## Overview
The IDA Pro processing script (`process.py`) generates three types of files for each binary:

### 1. `*_extract.pkl` - Pickle File (Binary Format)
**Purpose**: Stores structured function metadata extracted from the binary for baseline analysis

**Content**:
- Dictionary with function names as keys
- Each function contains:
  - `func`: Function entry address
  - `asm`: Assembly instruction list
  - `raw`: Raw bytes of the function
  - `cfg`: Control Flow Graph structure
  - `bai`: BinaryAI features (if available)

**Use Case**: 
- Used by machine learning models for function similarity detection
- Compatible with baseline jTrans format
- Can be loaded with `pickle.load()`

**Example Structure**:
```python
{
    'main': {
        'func': 0x401000,
        'asm': ['push rbp', 'mov rbp, rsp', ...],
        'raw': b'\x55\x48\x89\xe5...',
        'cfg': {...},
        'bai': {...}
    },
    'printf': { ... }
}
```

### 2. `*_addressaware.txt` - Text File (Address-Aware Format)
**Purpose**: Human-readable format with hierarchical position encoding for pretraining

**Content**:
- One function per line
- Tokenized instructions with address information
- Hierarchical position encoding (binary, function, basic block, instruction levels)
- Symbol names preserved for PLT functions

**Format**:
```
<BINARY_POS> <FUNC_POS> <BB_POS> <INST_POS> instruction_tokens...
```

**Use Case**:
- Training transformer models with address awareness
- Preserves spatial relationships between instructions
- Supports hierarchical attention mechanisms

### 3. `*_ida.log` - Log File
**Purpose**: Processing logs from IDA Pro execution

**Content**:
- Processing status messages
- Function count and statistics
- Error messages (if any)
- File paths and configuration info

**Location**: Currently saved in the same directory as output files (SAVEROOT)

---

## Redirecting Log Files to a Separate Folder

To organize your outputs better, you can modify the log file location in `process.py`.
