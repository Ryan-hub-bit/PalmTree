#!/usr/bin/env python3
"""
Generate paired data from address-aware format for contrastive learning.
This script creates function pairs where:
- Positive pairs: Same function at different optimization levels
- Negative pairs: Different functions

Output format compatible with FunctionDataset_CL_Load
"""

import os
import pickle
import argparse
from collections import defaultdict
from functools import reduce
from pathlib import Path
from tqdm import tqdm


def parse_filename(filename):
    """
    Parse filename to extract project name and optimization level.
    
    Format: {project}-{binary}-{OptLevel}-{hash}_extract.pkl
    Example: 2bwm-git-2bwm-O0-f9579e061f6e200bc50fdae0d8f2a873_extract.pkl
    
    Returns:
        tuple: (project_name, opt_level) or (None, None) if parsing fails
    """
    if not filename.endswith('_extract.pkl'):
        return None, None
    
    # Remove suffix
    base = filename.replace('_extract.pkl', '')
    parts = base.split('-')
    
    if len(parts) < 3:
        return None, None
    
    # Optimization level is second to last part
    opt_level = parts[-2]
    
    # Validate opt level
    valid_opts = ['O0', 'O1', 'O2', 'O3', 'Os', 'Ofast', 'Og']
    if opt_level not in valid_opts:
        return None, None
    
    # Project name is everything except last 2 parts (opt and hash)
    project = '-'.join(parts[:-2])
    
    return project, opt_level


def load_paired_data(data_path, opt_levels=None, min_opts=2):
    """
    Load paired data from address-aware format.
    
    Args:
        data_path: Path to directory containing _extract.pkl files
        opt_levels: List of optimization levels to include (None = all)
        min_opts: Minimum number of optimization levels required for a project
        
    Returns:
        dict: {project: {func_name: {opt_level: func_data}}}
    """
    if opt_levels is None:
        opt_levels = ['O0', 'O1', 'O2', 'O3', 'Os']
    
    print(f"Loading data from: {data_path}")
    print(f"Target optimization levels: {opt_levels}")
    
    # Group files by project and optimization level
    proj2pickle = defaultdict(dict)
    
    for filename in os.listdir(data_path):
        if not filename.endswith('_extract.pkl'):
            continue
        
        project, opt = parse_filename(filename)
        if project is None or opt not in opt_levels:
            continue
        
        pkl_path = os.path.join(data_path, filename)
        proj2pickle[project][opt] = pkl_path
        print(f"  Found: {project} [{opt}]")
    
    print(f"\nTotal projects found: {len(proj2pickle)}")
    
    # Load and pair functions
    paired_data = {}
    total_functions = 0
    
    for project, pickle_path_dict in tqdm(proj2pickle.items(), desc="Processing projects"):
        if len(pickle_path_dict) < min_opts:
            print(f"  Skipping {project}: only {len(pickle_path_dict)} opt levels")
            continue
        
        # Load all pkl files for this project
        tmp_pickle_dict = {}
        function_lists = []
        
        for opt, pkl_path in pickle_path_dict.items():
            with open(pkl_path, 'rb') as f:
                pkl = pickle.load(f)
                tmp_pickle_dict[opt] = pkl
                function_lists.append(set(pkl.keys()))
        
        # Find functions that exist in ALL optimization levels
        common_functions = reduce(lambda x, y: x & y, function_lists)
        
        if len(common_functions) == 0:
            print(f"  Skipping {project}: no common functions across opt levels")
            continue
        
        # Store paired data
        paired_data[project] = {}
        for func_name in common_functions:
            paired_data[project][func_name] = {}
            for opt, pkl in tmp_pickle_dict.items():
                paired_data[project][func_name][opt] = pkl[func_name]
        
        total_functions += len(common_functions)
        print(f"  {project}: {len(common_functions)} common functions across {len(pickle_path_dict)} opt levels")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Projects with paired data: {len(paired_data)}")
    print(f"  Total function groups: {total_functions}")
    print(f"{'='*60}")
    
    return paired_data


def generate_triplet_dataset(paired_data, opt_levels, output_path):
    """
    Generate dataset in format compatible with FunctionDataset_CL_Load.
    
    Format: List of function groups, where each group contains the same function
            at different optimization levels.
    
    Args:
        paired_data: Output from load_paired_data()
        opt_levels: List of optimization levels
        output_path: Path to save output pickle file
    """
    dataset = []
    metadata = []
    
    print("\nGenerating triplet dataset...")
    
    for project, functions in tqdm(paired_data.items(), desc="Processing functions"):
        for func_name, opt_data in functions.items():
            # Create function group (same function at different opts)
            func_group = []
            func_meta = {
                'project': project,
                'function': func_name,
                'opts': {}
            }
            
            for opt in opt_levels:
                if opt in opt_data:
                    func_group.append(opt_data[opt])
                    func_meta['opts'][opt] = len(func_group) - 1
            
            if len(func_group) >= 2:  # Need at least 2 variants for pairing
                dataset.append(func_group)
                metadata.append(func_meta)
    
    # Save dataset
    output = {
        'functions': dataset,
        'metadata': metadata,
        'opt_levels': opt_levels
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"  Total function groups: {len(dataset)}")
    print(f"  Optimization levels: {opt_levels}")
    
    return dataset, metadata


def print_statistics(paired_data, opt_levels):
    """Print detailed statistics about the paired data."""
    
    print("\n" + "="*60)
    print("DETAILED STATISTICS")
    print("="*60)
    
    # Per-project statistics
    print("\nPer-Project Statistics:")
    print(f"{'Project':<30} {'Functions':<10} {'Opt Levels'}")
    print("-" * 60)
    
    for project in sorted(paired_data.keys()):
        num_funcs = len(paired_data[project])
        # Check which opt levels exist for first function
        sample_func = list(paired_data[project].values())[0]
        opts = sorted([opt for opt in opt_levels if opt in sample_func])
        opts_str = ','.join(opts)
        print(f"{project:<30} {num_funcs:<10} {opts_str}")
    
    # Optimization level coverage
    print("\nOptimization Level Coverage:")
    opt_coverage = defaultdict(int)
    
    for project, functions in paired_data.items():
        for func_name, opt_data in functions.items():
            for opt in opt_levels:
                if opt in opt_data:
                    opt_coverage[opt] += 1
    
    for opt in sorted(opt_coverage.keys()):
        print(f"  {opt}: {opt_coverage[opt]} functions")
    
    # Function completeness (how many have all opt levels)
    print("\nFunction Completeness:")
    completeness_counts = defaultdict(int)
    
    for project, functions in paired_data.items():
        for func_name, opt_data in functions.items():
            num_opts = sum(1 for opt in opt_levels if opt in opt_data)
            completeness_counts[num_opts] += 1
    
    for num_opts in sorted(completeness_counts.keys(), reverse=True):
        count = completeness_counts[num_opts]
        print(f"  {num_opts} opt levels: {count} functions")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate paired data from address-aware format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate paired data with default opt levels (O0,O1,O2,O3,Os)
  python generate_paired_data.py /data/kun/jtransdata/addr_extract/ -o paired_data.pkl
  
  # Use only specific opt levels
  python generate_paired_data.py /data/kun/jtransdata/addr_extract/ -o paired_data.pkl --opt O0 O1 O2 O3
  
  # Require at least 3 opt levels per project
  python generate_paired_data.py /data/kun/jtransdata/addr_extract/ -o paired_data.pkl --min-opts 3
        """
    )
    
    parser.add_argument('data_path', type=str,
                        help='Path to directory containing _extract.pkl files')
    parser.add_argument('-o', '--output', type=str, default='paired_data.pkl',
                        help='Output pickle file path (default: paired_data.pkl)')
    parser.add_argument('--opt', nargs='+', default=['O0', 'O1', 'O2', 'O3', 'Os'],
                        help='Optimization levels to include (default: O0 O1 O2 O3 Os)')
    parser.add_argument('--min-opts', type=int, default=2,
                        help='Minimum number of opt levels required per project (default: 2)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, do not generate output file')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        return 1
    
    # Load paired data
    paired_data = load_paired_data(args.data_path, args.opt, args.min_opts)
    
    if len(paired_data) == 0:
        print("\nError: No paired data found!")
        return 1
    
    # Print statistics
    print_statistics(paired_data, args.opt)
    
    # Generate output dataset
    if not args.stats_only:
        dataset, metadata = generate_triplet_dataset(paired_data, args.opt, args.output)
        
        # Print sample
        print("\nSample function group:")
        if len(metadata) > 0:
            sample = metadata[0]
            print(f"  Project: {sample['project']}")
            print(f"  Function: {sample['function']}")
            print(f"  Available variants: {list(sample['opts'].keys())}")
    
    return 0


if __name__ == '__main__':
    exit(main())
