import os
import re
from collections import Counter

# Configuration
directory = "/data/kun/palmtreedata"
cfg_files = [f for f in os.listdir(directory) if f.endswith("_cfg_2_inline.txt")]
dfg_files = [f for f in os.listdir(directory) if f.endswith("_dfg_2_inline.txt")]

# Regex: Matches . followed by a letter, then any word characters
pattern = re.compile(r'\.([a-zA-Z][a-zA-Z0-9_]*)')

def get_top_100(file_list):
    counts = Counter()
    print("Counting frequencies...")
    for fname in file_list:
        with open(os.path.join(directory, fname), 'r', errors='ignore') as f:
            for line in f:
                counts.update(pattern.findall(line))
    return set(name for name, count in counts.most_common(100))

def merge_with_replace(file_list, output_name, top_set):
    print(f"Merging into {output_name}...")
    with open(os.path.join(directory, output_name), 'w') as out_f:
        for fname in file_list:
            with open(os.path.join(directory, fname), 'r', errors='ignore') as in_f:
                for line in in_f:
                    # Function to replace if not in top_set
                    def replace_func(match):
                        s = match.group(1)
                        return f".{s}" if s in top_set else ".plt"
                    
                    new_line = pattern.sub(replace_func, line)
                    out_f.write(new_line)

# Execute for CFG
top_100_cfg = get_top_100(cfg_files)
merge_with_replace(cfg_files, "cfg_train_2.txt", top_100_cfg)

# Execute for DFG
top_100_dfg = get_top_100(dfg_files)
merge_with_replace(dfg_files, "dfg_train_2.txt", top_100_dfg)

print("Process Complete!")