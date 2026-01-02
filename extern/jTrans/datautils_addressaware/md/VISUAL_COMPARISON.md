# Visual Comparison: Before & After Fixes

## Memory Operand Tokenization

### BEFORE (Concatenated)
```assembly
mov(0x1234:...) rax [rbp+var(0x8)]
                    ^^^^^^^^^^^^^^^^
                    Hard to parse - no spaces!

lea(0x1240:...) rdi [rax+rbx*4+0x10]
                    ^^^^^^^^^^^^^^^^^
                    All stuck together

mov(0x1250:...) esi [rsp-0x20]
                    ^^^^^^^^^^^
                    No token boundaries
```

### AFTER (Space-separated)
```assembly
mov(0x1234:...) rax [ rbp + var(0x8) ]
                    ^ ^^^ ^ ^^^^^^^^^^ ^
                    Clear token boundaries!

lea(0x1240:...) rdi [ rax + rbx * 4 + 0x10 ]
                    ^ ^^^ ^ ^^^ ^ ^ ^ ^^^^^^ ^
                    Each element is separate

mov(0x1250:...) esi [ rsp - 0x20 ]
                    ^ ^^^ ^ ^^^^^^ ^
                    Easy to tokenize
```

**Token sequence after fix:**
```
[, rbp, +, var(0x8), ]
[, rax, +, rbx, *, 4, +, 0x10, ]
[, rsp, -, 0x20, ]
```

---

## PLT Symbol Handling

### BEFORE (Generic addresses)
```assembly
call(0x4010:...) daddr(0x1030:2.00000000:0.00000000:0.00000000)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 Lost information! Which function is this?

call(0x4020:...) daddr(0x1040:2.00000000:0.00000000:0.00000000)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 Another unknown library call

jmp(0x4030:...) daddr(0x1050:2.00000000:0.00000000:0.00000000)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                What are we jumping to?
```

### AFTER (Named PLT symbols)
```assembly
call(0x4010:...) .printf
                 ^^^^^^^
                 Clear! This calls printf

call(0x4020:...) .malloc
                 ^^^^^^^
                 Allocating memory

jmp(0x4030:...) .free
                ^^^^^
                Freeing memory
```

**Common PLT symbols you'll see:**
```
.printf    - formatted output
.malloc    - memory allocation
.free      - memory deallocation
.calloc    - zero-initialized allocation
.realloc   - resize allocation
.memcpy    - memory copy
.memset    - memory set
.strlen    - string length
.strcmp    - string compare
.strcpy    - string copy
.fopen     - file open
.fclose    - file close
.exit      - program exit
```

---

## File Organization

### BEFORE (Mixed)
```
/data/kun/jtransdata/addr_extract/
├── 2bwm-O0-f957.._extract.pkl          [DATA]
├── 2bwm-O0-f957.._addressaware.txt     [DATA]
├── 2bwm-O0-f957.._ida.log              [LOG] ← mixed!
├── 2bwm-O1-f912.._extract.pkl          [DATA]
├── 2bwm-O1-f912.._addressaware.txt     [DATA]
├── 2bwm-O1-f912.._ida.log              [LOG] ← mixed!
├── 9base-bc-O0-50b1.._extract.pkl      [DATA]
├── 9base-bc-O0-50b1.._addressaware.txt [DATA]
├── 9base-bc-O0-50b1.._ida.log          [LOG] ← mixed!
└── ... (hundreds more mixed files)

Problem: Hard to list just data or just logs!
```

### AFTER (Organized)
```
/data/kun/jtransdata/addr_extract/
├── 2bwm-O0-f957.._extract.pkl          [DATA]
├── 2bwm-O0-f957.._addressaware.txt     [DATA]
├── 2bwm-O1-f912.._extract.pkl          [DATA]
├── 2bwm-O1-f912.._addressaware.txt     [DATA]
├── 9base-bc-O0-50b1.._extract.pkl      [DATA]
├── 9base-bc-O0-50b1.._addressaware.txt [DATA]
└── logs/                                [LOGS]
    ├── 2bwm-O0-f957.._ida.log
    ├── 2bwm-O1-f912.._ida.log
    ├── 9base-bc-O0-50b1.._ida.log
    └── ...

Clean! Easy to list, easy to manage!
```

**Quick commands:**
```bash
# List only data files
ls /data/kun/jtransdata/addr_extract/*.{pkl,txt}

# List only logs
ls /data/kun/jtransdata/addr_extract/logs/*.log

# Delete old logs without touching data
rm /data/kun/jtransdata/addr_extract/logs/*

# Archive logs by date
tar -czf logs_$(date +%Y%m%d).tar.gz /data/kun/jtransdata/addr_extract/logs/
```

---

## Complete Example: Before vs After

### BEFORE
```assembly
push(0x1000:...) rbp
mov(0x1001:...) rbp rsp
sub(0x1004:...) rsp imm
mov(0x100b:...) dword ptr[rbp-0x4] imm
mov(0x1012:...) edi [rbp-0x4]
call(0x1015:...) daddr(0x2030:2.00000000:0.00000000:0.00000000)
lea(0x101a:...) rax [rbp-0x8]
mov(0x101e:...) rsi rax
lea(0x1021:...) rdi daddr(0x3000:0.75:0.5:0.25)
call(0x1028:...) daddr(0x2040:2.00000000:0.00000000:0.00000000)
mov(0x102d:...) eax dword ptr[rbp-0x4]
```

### AFTER
```assembly
push(0x1000:...) rbp
mov(0x1001:...) rbp rsp
sub(0x1004:...) rsp imm
mov(0x100b:...) dword ptr [ rbp - 0x4 ] imm
                ^^^^^^^^^^ ^ ^^^ ^ ^^^^^ ^
mov(0x1012:...) edi [ rbp - 0x4 ]
                    ^ ^^^ ^ ^^^^^ ^
call(0x1015:...) .malloc
                 ^^^^^^^
lea(0x101a:...) rax [ rbp - 0x8 ]
                    ^ ^^^ ^ ^^^^^ ^
mov(0x101e:...) rsi rax
lea(0x1021:...) rdi daddr(0x3000:0.75:0.5:0.25)
call(0x1028:...) .printf
                 ^^^^^^^
mov(0x102d:...) eax dword ptr [ rbp - 0x4 ]
                ^^^^^^^^^^ ^ ^^^ ^ ^^^^^ ^
```

**Key improvements visible:**
1. ✅ `[ rbp - 0x4 ]` instead of `[rbp-0x4]` - easier tokenization
2. ✅ `.malloc` instead of `daddr(0x2030:...)` - semantic information preserved
3. ✅ `.printf` instead of `daddr(0x2040:...)` - function calls are clear

---

## Impact on Model Training

### Tokenization Benefits
- **Better positional encoding:** Model can learn relationships between `[`, register, `+`, offset, `]`
- **Consistent vocabulary:** Each symbol has its own token
- **Easier attention:** Model can attend to specific parts of memory operands

### Symbol Benefits
- **Semantic understanding:** Model learns `.malloc` allocates, `.free` deallocates
- **Transfer learning:** Knowledge about common library functions transfers across binaries
- **Function signatures:** Model can learn typical calling patterns for each library function

### Organization Benefits
- **Cleaner datasets:** Only data files in main directory
- **Faster loading:** No need to filter out log files when reading data
- **Better debugging:** Logs easily accessible when needed, out of the way otherwise

---

## Verification Checklist

After regenerating data, check:

- [ ] Memory operands have spaces: `[ rbp + var(0x8) ]`
- [ ] PLT symbols visible: `.printf`, `.malloc`, etc.
- [ ] Logs in separate directory: `logs/*.log`
- [ ] Data files unchanged: same `.pkl` structure
- [ ] No errors in log files
- [ ] File counts match: #pkl = #txt, #log files present
