#!/usr/bin/env python3
"""
Demo: Show how JUMP_ADDR_X encoding works with the converter.
Creates a mock pickle file and converts it.
"""

import pickle
import networkx as nx
import os

# Create a mock function with jumps
mock_function = {
    'main': {
        'func': 0x401000,
        'asm': [
            'push rbp',                    # 0
            'mov rbp, rsp',                # 1
            'sub rsp, 0x10',              # 2
            'cmp eax, 0',                  # 3
            'je loc_401020',               # 4 - jump to position 7
            'mov eax, 1',                  # 5
            'jmp loc_401030',              # 6 - jump to position 10
            'mov eax, 0',                  # 7 - target of je
            'inc eax',                     # 8
            'test eax, eax',               # 9
            'leave',                       # 10 - target of jmp
            'ret'                          # 11
        ],
        'raw': b'\x55\x48\x89\xe5...',
        'cfg': None,  # Will create below
        'bai': None
    }
}

# Create CFG with proper jump mappings
cfg = nx.DiGraph()

# Basic blocks:
# BB1: instructions 0-4 (ends with je)
# BB2: instructions 5-6 (ends with jmp)  
# BB3: instructions 7-9 (fallthrough/jump target)
# BB4: instructions 10-11 (final block)

# Add nodes with their assembly instructions
cfg.add_node(0x401000, asm=mock_function['main']['asm'][0:5])   # BB1
cfg.add_node(0x401010, asm=mock_function['main']['asm'][5:7])   # BB2
cfg.add_node(0x401020, asm=mock_function['main']['asm'][7:10])  # BB3
cfg.add_node(0x401030, asm=mock_function['main']['asm'][10:12]) # BB4

# Add edges (control flow)
cfg.add_edge(0x401000, 0x401010)  # BB1 -> BB2 (fallthrough)
cfg.add_edge(0x401000, 0x401020)  # BB1 -> BB3 (je jump)
cfg.add_edge(0x401010, 0x401030)  # BB2 -> BB4 (jmp)
cfg.add_edge(0x401020, 0x401030)  # BB3 -> BB4 (fallthrough)

mock_function['main']['cfg'] = cfg

# Save mock pickle
os.makedirs('/tmp/test_pkl', exist_ok=True)
with open('/tmp/test_pkl/mock_extract.pkl', 'wb') as f:
    pickle.dump(mock_function, f)

print("✅ Created mock pickle file with jumps")
print("\nOriginal assembly:")
for i, asm in enumerate(mock_function['main']['asm']):
    print(f"  {i:2d}: {asm}")

print("\nExpected conversion:")
print("  - Instruction 4 (je): should become 'je JUMP_ADDR_7'")
print("  - Instruction 6 (jmp): should become 'jmp JUMP_ADDR_10'")
print("\nNow run:")
print("  python3 convert_pkl_to_text.py /tmp/test_pkl /tmp/demo_output.txt")
