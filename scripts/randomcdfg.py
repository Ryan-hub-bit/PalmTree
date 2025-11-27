import os
import math

root = "/data/kun/dataset/val_cdfg"
out_cfg = "/data/kun/dataset/val_cfg.txt"
out_dfg = "/data/kun/dataset/val_dfg.txt"

def evenly_sample(lines, ratio=0.02):
    n = len(lines)
    k = max(1, int(n * ratio))
    if k <= 1:
        return lines[:1]
    step = n / k
    idx = [int(i * step) for i in range(k)]
    return [lines[i] for i in idx]

cfg_results = []
dfg_results = []

cfg_count = 0
dfg_count = 0

print("Starting sampling...\n")

for dirpath, _, files in os.walk(root):
    for name in files:
        path = os.path.join(dirpath, name)

        # Process CFG files
        if name.endswith("_cfg_8_inline.txt"):
            with open(path, "r") as f:
                lines = f.readlines()
            sampled = evenly_sample(lines, 0.02)
            cfg_results.extend(sampled)
            cfg_count += len(sampled)

            print(f"[CFG] {path}")
            print(f"     Total lines: {len(lines)}")
            print(f"     Sampled:     {len(sampled)}")
            print(f"     Running CFG total: {cfg_count}\n")

        # Process DFG files
        elif name.endswith("_dfg_8_inline.txt"):
            with open(path, "r") as f:
                lines = f.readlines()
            sampled = evenly_sample(lines, 0.02)
            dfg_results.extend(sampled)
            dfg_count += len(sampled)

            print(f"[DFG] {path}")
            print(f"     Total lines: {len(lines)}")
            print(f"     Sampled:     {len(sampled)}")
            print(f"     Running DFG total: {dfg_count}\n")

# write to output files
with open(out_cfg, "w") as f:
    f.writelines(cfg_results)

with open(out_dfg, "w") as f:
    f.writelines(dfg_results)

print("=====================================")
print("Done!")
print(f"Final → {out_cfg}  ({len(cfg_results)} lines)")
print(f"Final → {out_dfg}  ({len(dfg_results)} lines)")
print("=====================================")
