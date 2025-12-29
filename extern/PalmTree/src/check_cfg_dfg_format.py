"""
Check CFG/DFG data format to understand is_next logic
"""

def check_data_format():
    print("="*80)
    print("Checking CFG/DFG Data Format")
    print("="*80)
    
    # Check CFG file
    print("\nCFG file format:")
    print("-"*80)
    with open('/data/kun/palmtreedata/cfg_train_2.txt', 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            parts = line.strip().split('\t')
            print(f"Line {i+1}: {len(parts)} parts")
            for j, part in enumerate(parts):
                # Show first 100 chars
                preview = part[:100] + ('...' if len(part) > 100 else '')
                print(f"  Part {j+1}: {preview}")
            print()
    
    # Check DFG file
    print("\n" + "="*80)
    print("DFG file format:")
    print("-"*80)
    try:
        with open('/data/kun/palmtreedata/dfg_train_2.txt', 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                parts = line.strip().split('\t')
                print(f"Line {i+1}: {len(parts)} parts")
                for j, part in enumerate(parts):
                    preview = part[:100] + ('...' if len(part) > 100 else '')
                    print(f"  Part {j+1}: {preview}")
                print()
    except FileNotFoundError:
        print("  DFG file not found")
    
    print("\n" + "="*80)
    print("Analysis:")
    print("="*80)
    print("""
Expected format for original PalmTree:
  - Each line contains TWO sequences separated by TAB
  - Line format: sequence1 \\t sequence2
  - sequence1 and sequence2 are consecutive in the control/data flow
  
is_next logic:
  - is_next=1: Use (sequence1, sequence2) from the same line
  - is_next=0: Use (sequence1, random_sequence2) from different lines
  
This trains the model to predict if two sequences are consecutive
in the control flow (CFG) or data flow (DFG).
""")

if __name__ == '__main__':
    check_data_format()
