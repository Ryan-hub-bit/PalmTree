# Fair Pool Generation

## Purpose

Generates pool and query JSON files that ensure fair comparison between baseline and address-aware models by matching entries based on (binary, function_name, opt).

## What It Creates

For each pool size (100, 1000, 10000), it generates:

### Pool Files
- `pool_100.json`, `pool_1000.json`, `pool_10000.json`

Each entry contains:
```json
{
  "id": 0,
  "binary": "binary_name",
  "function_name": "func_name",
  "opt": "O0",
  "baseline_func_id": "123",
  "addressaware_func_id": "456"
}
```

### Query Files  
- `query_pool_100_O0_vs_O1.json`
- `query_pool_100_O0_vs_O2.json`
- `query_pool_100_O0_vs_O3.json`
- (and similar for pool sizes 1000 and 10000)

Same structure as pool files, but filtered to only anchor optimization level (e.g., O0 queries only).

### Statistics File
- `pool_statistics.json`

Contains summary info:
- Total common entries found
- Number of unique binaries
- Number of unique functions
- Optimization levels available

## Usage

### 1. Edit the script with your paths

```bash
vim run_generate_fair_pools.sh
```

Update these variables:
```bash
FUNC_BLOCKS_BASELINE="path/to/baseline/func_blocks.json"
FUNC_BLOCKS_ADDRESSAWARE="path/to/addressaware/func_blocks.json"
GROUND_TRUTH_BASELINE="path/to/baseline/ground_truth.json"
GROUND_TRUTH_ADDRESSAWARE="path/to/addressaware/ground_truth.json"
```

### 2. Run the script

```bash
./run_generate_fair_pools.sh
```

Or directly:

```bash
python generate_fair_pools.py \
    --func_blocks_baseline /path/to/baseline/func_blocks.json \
    --func_blocks_addressaware /path/to/addressaware/func_blocks.json \
    --ground_truth_baseline /path/to/baseline/ground_truth.json \
    --ground_truth_addressaware /path/to/addressaware/ground_truth.json \
    --output_dir ./fair_pools \
    --pool_sizes 100 1000 10000 \
    --seed 42
```

## Output Structure

```
fair_pools/
├── pool_100.json              # Pool of 100 matched entries
├── pool_1000.json             # Pool of 1000 matched entries
├── pool_10000.json            # Pool of 10000 matched entries
├── query_pool_100_O0_vs_O1.json    # O0 queries for pool 100
├── query_pool_100_O0_vs_O2.json
├── query_pool_100_O0_vs_O3.json
├── query_pool_1000_O0_vs_O1.json   # O0 queries for pool 1000
├── query_pool_1000_O0_vs_O2.json
├── query_pool_1000_O0_vs_O3.json
├── query_pool_10000_O0_vs_O1.json  # O0 queries for pool 10000
├── query_pool_10000_O0_vs_O2.json
├── query_pool_10000_O0_vs_O3.json
└── pool_statistics.json       # Statistics about the pools
```

## How It Works

1. **Load ground truth files** from both baseline and address-aware datasets
2. **Extract metadata** (binary, function_name, opt) for each function
3. **Find common entries** that exist in BOTH datasets
4. **Match by metadata**: Same binary + function name + opt = matched pair
5. **Create pool files** with baseline_func_id and addressaware_func_id
6. **Create query files** by filtering pools to anchor optimization level
7. **Generate statistics** about the common data

## Example Output

```
============================================================
SUMMARY
============================================================
Total matched entries: 15234
Unique binaries: 50
Unique functions: 3045
Optimization levels: ['O0', 'O1', 'O2', 'O3']

Generated pools: [100, 1000, 10000]
  pool_100.json: 100 entries
    query_pool_100_O0_vs_O1.json: ~25 queries
    query_pool_100_O0_vs_O2.json: ~25 queries
    query_pool_100_O0_vs_O3.json: ~25 queries
  pool_1000.json: 1000 entries
    query_pool_1000_O0_vs_O1.json: ~250 queries
    query_pool_1000_O0_vs_O2.json: ~250 queries
    query_pool_1000_O0_vs_O3.json: ~250 queries
  pool_10000.json: 10000 entries
    query_pool_10000_O0_vs_O1.json: ~2500 queries
    query_pool_10000_O0_vs_O2.json: ~2500 queries
    query_pool_10000_O0_vs_O3.json: ~2500 queries
============================================================
```

## What This Solves

✅ **Same queries**: Both models query the same (binary, function_name, opt)  
✅ **Same pools**: Both models have the same pool members  
✅ **Deterministic**: Fixed seed ensures reproducible pools  
✅ **Traceable**: ID mapping allows verification and debugging  
✅ **Fair comparison**: Eliminates dataset differences from evaluation  

## Next Steps

After generating these pool files, you can:

1. Use them in a modified evaluation script that loads pre-defined pools instead of sampling
2. Verify the pool quality by inspecting pool_statistics.json
3. Ensure both models are tested on identical queries and candidate sets
