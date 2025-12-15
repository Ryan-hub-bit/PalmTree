# Example Data Format for Address-Aware PalmTree

## Standard PalmTree Data Format

**File: cfg_train.txt**
```
mov rax rbx	push rbp
call symbol	mov rsp rbp
lea rax var(0x10)	add rax 0x8
```

**File: dfg_train.txt**
```
mov rax rbx	xor rax rax
lea rcx var(0x20)	call address
add rax rcx	ret
```

**Characteristics:**
- Simple token sequences
- No position information
- `var(0x10)` treated as a single token
- Tab-separated sequences (for NSP task)

---

## Address-Aware PalmTree Data Format

**File: cfg_train.txt**
```
mov(0x401000:0.000000:0.000000:0.000000) rax rbx	push(0x401003:0.000015:0.012500:0.166667) rbp
call(0x401004:0.000020:0.016667:0.222222) symbol(0x402000:0.000500:0.000000:0.000000)	mov(0x401009:0.000045:0.037500:0.388889) rsp rbp
lea(0x40100c:0.000060:0.050000:0.555556) rax var(0x10)	add(0x401010:0.000080:0.066667:0.777778) rax 0x8
```

**File: dfg_train.txt**
```
mov(0x401000:0.000000:0.000000:0.000000) rax rbx	xor(0x401003:0.000015:0.012500:0.000000) rax rax
lea(0x401005:0.000025:0.020833:0.000000) rcx var(0x20)	call(0x401009:0.000045:0.037500:0.000000) address(0x402000:0.000500:0.000000:0.000000)
add(0x40100e:0.000070:0.058333:0.000000) rax rcx	ret(0x401010:0.000080:0.066667:0.000000)
```

**Characteristics:**
- Each token has inline address information: `token(addr:bnorm:fnorm:bbnorm)`
- Position information is normalized to [0.0, 1.0]
- `var(0x10)` and `var(0x20)` remain as tokens (offsets extracted by parser)
- Tab-separated sequences (for NSP task)

---

## Format Breakdown

### Token with Address Info

```
mov(0x401000:0.000000:0.000000:0.000000)
│   │        │        │        │
│   │        │        │        └─ Basic block position (bbnorm)
│   │        │        └────────── Function position (fnorm)
│   │        └─────────────────── Binary position (bnorm)
│   └──────────────────────────── Instruction address (hexadecimal)
└────────────────────────────────── Token (opcode)
```

### Token WITHOUT Address Info

```
rax             # Register - no address info
var(0x10)       # Variable - offset extracted by parser
0x8             # Immediate - no address info
```

---

## Position Normalization Examples

### Binary-level normalization (bnorm)

If binary has address range `[0x400000, 0x410000]` (65536 bytes):

```
Instruction at 0x401000:
bnorm = (0x401000 - 0x400000) / (0x410000 - 0x400000)
      = 0x1000 / 0x10000
      = 4096 / 65536
      = 0.0625
```

### Function-level normalization (fnorm)

If function starts at `0x401000` and ends at `0x401080` (128 bytes):

```
Instruction at 0x401020:
fnorm = (0x401020 - 0x401000) / (0x401080 - 0x401000)
      = 0x20 / 0x80
      = 32 / 128
      = 0.25
```

### Basic block-level normalization (bbnorm)

If basic block starts at `0x401020` and ends at `0x401038` (24 bytes):

```
Instruction at 0x401028:
bbnorm = (0x401028 - 0x401020) / (0x401038 - 0x401020)
       = 0x8 / 0x18
       = 8 / 24
       = 0.333333
```

---

## Variable Offset Examples

### Stack Variables

```
var(0x0)   → offset = 0   (base pointer)
var(0x8)   → offset = 8   (first local variable)
var(0x10)  → offset = 16  (second local variable)
var(0x20)  → offset = 32  (third local variable)
```

The offset value (e.g., 16 for `var(0x10)`) is used in the sinusoidal encoding:
```python
encoding[:, 0::2] = sin(offset * div_term)  # Even indices
encoding[:, 1::2] = cos(offset * div_term)  # Odd indices
```

---

## Real Example: Simple Function

### Assembly Code
```asm
0x401000:  push   rbp               ; Function prologue
0x401001:  mov    rbp, rsp
0x401004:  sub    rsp, 0x20
0x401008:  mov    QWORD PTR [rbp-0x8], rdi
0x40100c:  mov    QWORD PTR [rbp-0x10], rsi
0x401010:  mov    rax, QWORD PTR [rbp-0x8]
0x401014:  add    rax, QWORD PTR [rbp-0x10]
0x401018:  leave
0x401019:  ret
```

### Binary Info
- Binary address range: `[0x400000, 0x410000]` (65536 bytes)
- Function address range: `[0x401000, 0x40101a]` (26 bytes)

### Address-Aware Format (CFG)

Assuming basic block covers the entire function:

```
push(0x401000:0.0625:0.000000:0.000000) rbp	mov(0x401001:0.062515:0.038462:0.038462) rbp rsp
sub(0x401004:0.062576:0.153846:0.153846) rsp 0x20	mov(0x401008:0.062652:0.307692:0.307692) address(0x401008:0.062652:0.307692:0.307692) var(0x8)
mov(0x40100c:0.062729:0.461538:0.461538) address(0x40100c:0.062729:0.461538:0.461538) var(0x10)	mov(0x401010:0.062805:0.615385:0.615385) rax address(0x401010:0.062805:0.615385:0.615385)
add(0x401014:0.062882:0.769231:0.769231) rax address(0x401014:0.062882:0.769231:0.769231)	leave(0x401018:0.062958:0.923077:0.923077)
ret(0x401019:0.062973:1.000000:1.000000)
```

**Position calculations:**

```
0x401000: bnorm = (0x401000 - 0x400000) / 0x10000 = 0.0625
          fnorm = (0x401000 - 0x401000) / 0x1a = 0.0
          bbnorm = (0x401000 - 0x401000) / 0x1a = 0.0

0x401008: bnorm = (0x401008 - 0x400000) / 0x10000 = 0.062652
          fnorm = (0x401008 - 0x401000) / 0x1a = 0.307692
          bbnorm = (0x401008 - 0x401000) / 0x1a = 0.307692

0x401019: bnorm = (0x401019 - 0x400000) / 0x10000 = 0.062973
          fnorm = (0x401019 - 0x401000) / 0x1a = 1.0
          bbnorm = (0x401019 - 0x401000) / 0x1a = 1.0
```

---

## Token Type Summary

| Token Type | Example | Has Address Info? | Has Var Offset? |
|-----------|---------|-------------------|-----------------|
| Opcode | `mov(...)` | ✓ Yes | ✗ No |
| Register | `rax` | ✗ No | ✗ No |
| Address | `address(0x401000:...)` | ✓ Yes | ✗ No |
| Symbol | `symbol(0x402000:...)` | ✓ Yes | ✗ No |
| String | `string(0x403000:...)` | ✓ Yes | ✗ No |
| Variable | `var(0x10)` | ✗ No | ✓ Yes (offset=16) |
| Immediate | `0x8` | ✗ No | ✗ No |
| Special | `[SOS]`, `[EOS]` | ✗ No | ✗ No |

---

## Generating This Data

Use the hierarchical data generators:

```bash
# Generate CFG with address info
python data_generator/cfg_hierarchical_icfg_ida.py \
    --input /path/to/binary \
    --output cfg_train.txt

# Generate DFG with address info  
python data_generator/dfg_hierarchical_idfg_ida.py \
    --input /path/to/binary \
    --output dfg_train.txt
```

These scripts automatically compute normalized positions and embed them in the output format.

---

## Validation

To verify your data format is correct:

```python
import re

def validate_line(line):
    """Check if line has proper address-aware format."""
    parts = line.strip().split('\t')
    for sequence in parts:
        tokens = sequence.split()
        for token in tokens:
            # Check for opcode with address
            if re.match(r'[a-zA-Z_]\w*\(0x[0-9a-fA-F]+:[\d.]+:[\d.]+:[\d.]+\)', token):
                print(f"✓ Valid opcode: {token}")
            # Check for operand with address
            elif re.match(r'(address|symbol|string)\(0x[0-9a-fA-F]+:[\d.]+:[\d.]+:[\d.]+\)', token):
                print(f"✓ Valid address operand: {token}")
            # Check for var
            elif re.match(r'var\(0x[0-9a-fA-F]+\)', token):
                print(f"✓ Valid var: {token}")
            # Plain token (register, immediate, etc.)
            else:
                print(f"✓ Plain token: {token}")

# Test with your data
with open('cfg_train.txt', 'r') as f:
    validate_line(f.readline())
```

Expected output should show all tokens properly parsed with their address information.
