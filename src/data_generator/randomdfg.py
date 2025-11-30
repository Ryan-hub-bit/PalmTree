import os
import math

root = "/data/kun/dataset/test_cdfg"
out_dfg = "/data/kun/dataset/test_dfg.txt"

def evenly_sample(lines, ratio=0.02):
    n = len(lines)
    k = max(1, int(n * ratio))
    if k <= 1:
        return lines[:1]
    step = n / k
    idx = [int(i * step) for i in range(k)]
    return [lines[i] for i in idx]

dfg_results = []
dfg_count = 0

print("Starting DFG sampling...\n")

for dirpath, _, files in os.walk(root):
    for name in files:
        path = os.path.join(dirpath, name)

        # Process DFG files only
        if name.endswith("_dfg_8_inline.txt"):
            with open(path, "r") as f:
                lines = f.readlines()
            sampled = evenly_sample(lines, 0.02)
            dfg_results.extend(sampled)
            dfg_count += len(sampled)

            print(f"[DFG] {path}")
            print(f"     Total lines: {len(lines)}")
            print(f"     Sampled:     {len(sampled)}")
            print(f"     Running DFG total: {dfg_count}\n")

# write to output file
with open(out_dfg, "w") as f:
    f.writelines(dfg_results)

print("=====================================")
print("Done!")
print(f"Final → {out_dfg}  ({len(dfg_results)} lines)")
print("=====================================")
