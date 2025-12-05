import os

data_dir = "/data/kun/dataset"
files = {
    "train": os.path.join(data_dir, "train_scope.txt"),
    "val":   os.path.join(data_dir, "val_scope.txt"),
    "test":  os.path.join(data_dir, "test_scope.txt")
}

def get_line_count(filepath):
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

# 1. Get Counts
counts = {k: get_line_count(v) for k, v in files.items()}
print(f"Current counts: {counts}")

# 2. Calculate Limiting Base Unit
# Train contributes 8 parts, Val/Test contribute 1 part
bases = [
    counts["train"] // 8,
    counts["val"],
    counts["test"]
]
limit_base = min(bases)
print(f"Limiting base unit: {limit_base}")

# 3. Define Targets
targets = {
    "train": limit_base * 8,
    "val":   limit_base,
    "test":  limit_base
}
print(f"Target counts: {targets}")

# 4. Truncate files
for key, filepath in files.items():
    target = targets[key]
    
    # Read all lines
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Write back only the target amount
    with open(filepath, 'w') as f:
        f.writelines(lines[:target])

print("Files truncated successfully.")