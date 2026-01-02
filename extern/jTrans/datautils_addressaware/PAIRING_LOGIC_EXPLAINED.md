# Function Pairing Logic Explained

## How We Identify Same-Source Functions

### The Two-Key System

Your address-aware format uses **TWO identifiers** to match functions:

```
┌─────────────────────────────────────────────────────────────────┐
│                    IDENTIFIER 1: PROJECT NAME                   │
│                   (extracted from filename)                     │
└─────────────────────────────────────────────────────────────────┘

Filename: 2bwm-git-2bwm-O0-f9579e061f6e200bc50fdae0d8f2a873_extract.pkl
          └──────┬──────┘ │  └───────────┬──────────────┘
            PROJECT      OPT          HASH (unique per build)
          
Project Name = "2bwm-git-2bwm"  ← Everything BEFORE the opt level


┌─────────────────────────────────────────────────────────────────┐
│                   IDENTIFIER 2: FUNCTION NAME                   │
│                  (key in the .pkl dictionary)                   │
└─────────────────────────────────────────────────────────────────┘

.pkl file contents:
{
  'focusnext': {...},        ← Function name from debug symbols
  'updateclientlist': {...},
  'changescreen': {...},
  ...
}
```

---

## Complete Example: The `focusnext` Function

### Step 1: Same Binary Compiled with Different Optimizations

```
Source Code (C):
┌──────────────────────────┐
│ void focusnext(void) {   │
│   // switch focus to     │
│   // next window         │
│ }                        │
└──────────────────────────┘
          │
          ├─ gcc -O0 ────> 2bwm-git-2bwm-O0-hash1.strip
          ├─ gcc -O1 ────> 2bwm-git-2bwm-O1-hash2.strip
          ├─ gcc -O2 ────> 2bwm-git-2bwm-O2-hash3.strip
          ├─ gcc -O3 ────> 2bwm-git-2bwm-O3-hash4.strip
          └─ gcc -Os ────> 2bwm-git-2bwm-Os-hash5.strip
```

### Step 2: IDA Pro Extracts Functions

Each binary is analyzed by IDA Pro and generates an `_extract.pkl` file:

```
2bwm-git-2bwm-O0-hash1_extract.pkl:
{
  'focusnext': {
    'func': 0x461e,
    'asm': ['endbr64', 'push rbp', 'mov rbp, rsp', ...],  # 15 instructions
    ...
  },
  'updateclientlist': {...},
  'changescreen': {...},
  ...
}

2bwm-git-2bwm-O1-hash2_extract.pkl:
{
  'focusnext': {
    'func': 0x4584,
    'asm': ['endbr64', 'movzx ecx, cs:curws', ...],  # 83 instructions (inlined!)
    ...
  },
  'updateclientlist': {...},
  ...
}

2bwm-git-2bwm-O2-hash3_extract.pkl:
{
  'focusnext': {
    'func': 0x7020,
    'asm': ['endbr64', 'push rbp', 'lea rcx, wslist', ...],  # 120 instructions
    ...
  },
  'updateclientlist': {...},
  ...
}
```

### Step 3: Pairing Algorithm

```python
# 1. Group files by PROJECT
proj2files = {
    '2bwm-git-2bwm': {
        'O0': '2bwm-git-2bwm-O0-hash1_extract.pkl',
        'O1': '2bwm-git-2bwm-O1-hash2_extract.pkl',
        'O2': '2bwm-git-2bwm-O2-hash3_extract.pkl',
        'O3': '2bwm-git-2bwm-O3-hash4_extract.pkl',
        'Os': '2bwm-git-2bwm-Os-hash5_extract.pkl',
    },
    '9base-bc': {
        'O0': '9base-bc-O0-hash1_extract.pkl',
        'O1': '9base-bc-O1-hash2_extract.pkl',
        ...
    }
}

# 2. For each project, load all .pkl files
for project in ['2bwm-git-2bwm', '9base-bc', ...]:
    pkl_data = {}
    for opt in ['O0', 'O1', 'O2', 'O3', 'Os']:
        pkl_data[opt] = load_pickle(proj2files[project][opt])
    
    # 3. Find INTERSECTION of function names across all opt levels
    func_names_O0 = set(pkl_data['O0'].keys())  # {'focusnext', 'updateclientlist', ...}
    func_names_O1 = set(pkl_data['O1'].keys())  # {'focusnext', 'updateclientlist', ...}
    func_names_O2 = set(pkl_data['O2'].keys())  # {'focusnext', 'changescreen', ...}
    ...
    
    common_functions = func_names_O0 & func_names_O1 & func_names_O2 & func_names_O3 & func_names_Os
    # Result: {'focusnext', 'updateclientlist', ...}  (72 functions for 2bwm)
    
    # 4. Create function groups
    for func_name in common_functions:
        function_group = [
            pkl_data['O0'][func_name],  # Same function
            pkl_data['O1'][func_name],  # Same function
            pkl_data['O2'][func_name],  # Same function
            pkl_data['O3'][func_name],  # Same function
            pkl_data['Os'][func_name],  # Same function
        ]
        yield function_group
```

---

## Visualization of Function Groups

```
┌───────────────────────────────────────────────────────────────────────────┐
│                       FUNCTION GROUP: focusnext                           │
│                    (Same source code, 5 variants)                         │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  2bwm-git-2bwm + focusnext + O0  →  0x461e   15 instructions             │
│  2bwm-git-2bwm + focusnext + O1  →  0x4584   83 instructions             │
│  2bwm-git-2bwm + focusnext + O2  →  0x7020  120 instructions             │
│  2bwm-git-2bwm + focusnext + O3  →  0x8b20  120 instructions             │
│  2bwm-git-2bwm + focusnext + Os  →  0x5e30   71 instructions             │
│                                                                           │
│  ⚡ These are POSITIVE PAIRS for contrastive learning                    │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                   FUNCTION GROUP: updateclientlist                        │
│                    (Different source code, 5 variants)                    │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  2bwm-git-2bwm + updateclientlist + O0  →  0x3a1c   45 instructions      │
│  2bwm-git-2bwm + updateclientlist + O1  →  0x3e80   52 instructions      │
│  2bwm-git-2bwm + updateclientlist + O2  →  0x5fd0   67 instructions      │
│  2bwm-git-2bwm + updateclientlist + O3  →  0x7a90   68 instructions      │
│  2bwm-git-2bwm + updateclientlist + Os  →  0x4d10   59 instructions      │
│                                                                           │
│  ❌ NEGATIVE PAIRS with focusnext group                                  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Triplet Loss Training Example

```
Random selection for one training sample:

┌─────────────┐
│   ANCHOR    │  focusnext @ O0  (15 instructions)
└─────────────┘  "I am the reference"
       │
       ├── POSITIVE (same function, different opt)
       │   ┌─────────────┐
       │   │  POSITIVE   │  focusnext @ O2  (120 instructions)
       │   └─────────────┘  "I should be CLOSE to anchor"
       │
       └── NEGATIVE (different function)
           ┌─────────────┐
           │  NEGATIVE   │  updateclientlist @ O1  (52 instructions)
           └─────────────┘  "I should be FAR from anchor"

Loss = max(0, distance(anchor, positive) - distance(anchor, negative) + margin)

Goal: distance(anchor, positive) < distance(anchor, negative)
      Same function at different opts should have similar embeddings!
```

---

## Why This Works

### ✅ Function Names are Preserved

Even in **stripped binaries**, IDA Pro can recover function names because:

1. **Debug symbols** in unstripped binaries (if available)
2. **DWARF information** (if present)
3. **Dynamic symbols** (for exported functions)
4. **Pattern matching** (IDA's FLIRT signatures)
5. **Your data appears to have symbols** (you see real function names like `focusnext`, not `sub_461E`)

### ✅ Filename Encodes Metadata

```
2bwm-git-2bwm-O2-78b862c439faaf310297881d8cb2ef6a_extract.pkl
│              │                                │
│              │                                └─ Unique hash (changes per build)
│              └─ Optimization level (stable identifier)
└─ Project+Binary name (stable identifier)
```

The optimization level at position `-2` is a **consistent pattern** across all your files.

---

## Summary: The Matching Criteria

Two functions are identified as **the same source code** if and only if:

| Criterion | Source | Must Match? |
|-----------|--------|-------------|
| **Project Name** | Filename (parts[:-2]) | ✅ YES |
| **Function Name** | .pkl dict key | ✅ YES |
| **Optimization Level** | Filename (parts[-2]) | ❌ NO (this is what we vary!) |
| **Hash** | Filename (parts[-1]) | ❌ NO (different per build) |
| **Address** | func_data['func'] | ❌ NO (changes with opt) |
| **Assembly** | func_data['asm'] | ❌ NO (changes with opt) |

**Result**: Functions with same project name + function name across different optimization levels form a **positive pair group** for contrastive learning.

---

## Your Data Has This Information! ✅

```bash
$ python generate_paired_data.py /data/kun/jtransdata/addr_extract/ -o paired_data.pkl

Total projects found: 13
2bwm-git-2bwm: 72 common functions across 5 opt levels
9base-bc: 125 common functions across 5 opt levels
...
Total function groups: 2232
```

You're ready to fine-tune! 🚀
