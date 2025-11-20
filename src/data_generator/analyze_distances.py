"""
Analyze actual distance distributions for operand addresses.
This helps determine appropriate clipping bounds for fnorm/bbnorm.
"""

from binaryninja import load
import sys
import os
from collections import defaultdict
import numpy as np

def analyze_binary_distances(fpath):
    """
    Analyze distances for all operand addresses in a binary.
    Returns statistics for function-level and BB-level distances.
    """
    print(f"[INFO] Analyzing: {fpath}")
    bv = load(fpath)
    if bv is None:
        print(f"[WARN] Could not load {fpath}")
        return None
    
    # Collect all instruction addresses and their contexts
    addr_to_context = {}  # addr -> (func_start, func_end, bb_start, bb_end)
    
    for func in bv.functions:
        func_start = func.start
        func_end = func.start + func.total_bytes
        
        for block in func:
            bb_start = block.start
            bb_end = block.start
            
            for inst in block:
                curr_addr = bb_end
                bb_end += inst[1]
                addr_to_context[curr_addr] = (func_start, func_end, bb_start, bb_end)
    
    # Analyze distances for operand addresses
    func_distances = []
    bb_distances = []
    
    intra_func_distances = []
    inter_func_distances = []
    data_distances = []
    
    for func in bv.functions:
        for block in func:
            curr_addr = block.start
            for inst in block:
                # Get current context
                if curr_addr not in addr_to_context:
                    curr_addr += inst[1]
                    continue
                
                curr_func_start, curr_func_end, curr_bb_start, curr_bb_end = addr_to_context[curr_addr]
                func_size = curr_func_end - curr_func_start
                bb_size = curr_bb_end - curr_bb_start
                
                # Check all operands for address references
                for token in inst[0]:
                    if token.type == 8:  # Integer constant (potential address)
                        target = token.value
                        
                        # Skip small immediates
                        if target < 0x1000:
                            continue
                        
                        # Calculate distances
                        if func_size > 0:
                            func_dist = (target - curr_func_start) / float(func_size)
                            func_distances.append(func_dist)
                            
                            # Categorize
                            if target in addr_to_context:
                                tgt_func_start, _, _, _ = addr_to_context[target]
                                if tgt_func_start == curr_func_start:
                                    intra_func_distances.append(func_dist)
                                else:
                                    inter_func_distances.append(func_dist)
                            else:
                                data_distances.append(func_dist)
                        
                        if bb_size > 0:
                            bb_dist = (target - curr_bb_start) / float(bb_size)
                            bb_distances.append(bb_dist)
                
                curr_addr += inst[1]
    
    # Compute statistics
    stats = {}
    
    if func_distances:
        func_arr = np.array(func_distances)
        stats['func'] = {
            'count': len(func_arr),
            'min': float(np.min(func_arr)),
            'max': float(np.max(func_arr)),
            'mean': float(np.mean(func_arr)),
            'median': float(np.median(func_arr)),
            'std': float(np.std(func_arr)),
            'p95': float(np.percentile(func_arr, 95)),
            'p99': float(np.percentile(func_arr, 99)),
            'p99_9': float(np.percentile(func_arr, 99.9)),
        }
    
    if bb_distances:
        bb_arr = np.array(bb_distances)
        stats['bb'] = {
            'count': len(bb_arr),
            'min': float(np.min(bb_arr)),
            'max': float(np.max(bb_arr)),
            'mean': float(np.mean(bb_arr)),
            'median': float(np.median(bb_arr)),
            'std': float(np.std(bb_arr)),
            'p95': float(np.percentile(bb_arr, 95)),
            'p99': float(np.percentile(bb_arr, 99)),
            'p99_9': float(np.percentile(bb_arr, 99.9)),
        }
    
    # Breakdown by category
    if intra_func_distances:
        intra_arr = np.array(intra_func_distances)
        stats['intra_func'] = {
            'count': len(intra_arr),
            'mean': float(np.mean(intra_arr)),
            'median': float(np.median(intra_arr)),
            'p95': float(np.percentile(intra_arr, 95)),
            'p99': float(np.percentile(intra_arr, 99)),
        }
    
    if inter_func_distances:
        inter_arr = np.array(inter_func_distances)
        stats['inter_func'] = {
            'count': len(inter_arr),
            'mean': float(np.mean(inter_arr)),
            'median': float(np.median(inter_arr)),
            'p95': float(np.percentile(inter_arr, 95)),
            'p99': float(np.percentile(inter_arr, 99)),
        }
    
    if data_distances:
        data_arr = np.array(data_distances)
        stats['data'] = {
            'count': len(data_arr),
            'mean': float(np.mean(data_arr)),
            'median': float(np.median(data_arr)),
            'p95': float(np.percentile(data_arr, 95)),
            'p99': float(np.percentile(data_arr, 99)),
        }
    
    return stats


def print_stats(stats):
    """Print statistics in a readable format."""
    print("\n" + "="*80)
    print("FUNCTION-LEVEL DISTANCE STATISTICS")
    print("="*80)
    
    if 'func' in stats:
        s = stats['func']
        print(f"Total samples: {s['count']}")
        print(f"Range: [{s['min']:.2f}, {s['max']:.2f}]")
        print(f"Mean: {s['mean']:.2f}, Median: {s['median']:.2f}, Std: {s['std']:.2f}")
        print(f"95th percentile: {s['p95']:.2f}")
        print(f"99th percentile: {s['p99']:.2f}")
        print(f"99.9th percentile: {s['p99_9']:.2f}")
    
    print("\n" + "-"*80)
    print("BREAKDOWN BY TYPE")
    print("-"*80)
    
    if 'intra_func' in stats:
        s = stats['intra_func']
        print(f"\nIntra-function jumps: {s['count']} samples")
        print(f"  Mean: {s['mean']:.2f}, Median: {s['median']:.2f}")
        print(f"  95th percentile: {s['p95']:.2f}, 99th percentile: {s['p99']:.2f}")
    
    if 'inter_func' in stats:
        s = stats['inter_func']
        print(f"\nInter-function calls: {s['count']} samples")
        print(f"  Mean: {s['mean']:.2f}, Median: {s['median']:.2f}")
        print(f"  95th percentile: {s['p95']:.2f}, 99th percentile: {s['p99']:.2f}")
    
    if 'data' in stats:
        s = stats['data']
        print(f"\nData references: {s['count']} samples")
        print(f"  Mean: {s['mean']:.2f}, Median: {s['median']:.2f}")
        print(f"  95th percentile: {s['p95']:.2f}, 99th percentile: {s['p99']:.2f}")
    
    print("\n" + "="*80)
    print("BASIC BLOCK-LEVEL DISTANCE STATISTICS")
    print("="*80)
    
    if 'bb' in stats:
        s = stats['bb']
        print(f"Total samples: {s['count']}")
        print(f"Range: [{s['min']:.2f}, {s['max']:.2f}]")
        print(f"Mean: {s['mean']:.2f}, Median: {s['median']:.2f}, Std: {s['std']:.2f}")
        print(f"95th percentile: {s['p95']:.2f}")
        print(f"99th percentile: {s['p99']:.2f}")
        print(f"99.9th percentile: {s['p99_9']:.2f}")
    
    print("\n" + "="*80)
    print("RECOMMENDED CLIPPING BOUNDS")
    print("="*80)
    
    if 'func' in stats:
        # Recommend based on 99th percentile
        func_clip = max(2.0, stats['func']['p99'])
        print(f"Function-level (fnorm): [-{func_clip:.1f}, +{func_clip:.1f}]")
        print(f"  (Covers 99% of data, clips {100-99:.1f}% outliers)")
    
    if 'bb' in stats:
        bb_clip = max(5.0, stats['bb']['p99'])
        print(f"BB-level (bbnorm): [-{bb_clip:.1f}, +{bb_clip:.1f}]")
        print(f"  (Covers 99% of data, clips {100-99:.1f}% outliers)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_distances.py <binary_path>")
        sys.exit(1)
    
    binary_path = sys.argv[1]
    
    if os.path.isfile(binary_path):
        # Single file
        stats = analyze_binary_distances(binary_path)
        if stats:
            print_stats(stats)
    elif os.path.isdir(binary_path):
        # Directory of binaries
        all_stats = defaultdict(list)
        
        file_count = 0
        for parent, _, files in os.walk(binary_path):
            for f in files:
                if f.endswith('.txt'):
                    continue
                full = os.path.join(parent, f)
                stats = analyze_binary_distances(full)
                if stats:
                    for key in stats:
                        for metric in stats[key]:
                            all_stats[f"{key}_{metric}"].append(stats[key][metric])
                    file_count += 1
                
                if file_count >= 10:  # Analyze first 10 binaries
                    break
            if file_count >= 10:
                break
        
        # Aggregate statistics
        print("\n" + "="*80)
        print(f"AGGREGATED STATISTICS ({file_count} binaries)")
        print("="*80)
        
        for key in ['func_p99', 'func_p99_9', 'bb_p99', 'bb_p99_9']:
            if key in all_stats and all_stats[key]:
                values = all_stats[key]
                print(f"\n{key}:")
                print(f"  Mean across binaries: {np.mean(values):.2f}")
                print(f"  Median across binaries: {np.median(values):.2f}")
                print(f"  Max across binaries: {np.max(values):.2f}")


if __name__ == '__main__':
    main()
