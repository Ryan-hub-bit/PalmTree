#!/usr/bin/env python
"""
Compare baseline and address-aware model results from separate evaluation outputs.

Usage:
    python compare_results.py \
        --baseline_result /path/to/baseline_result.txt \
        --addressaware_result /path/to/addressaware_result.txt \
        --output_file comparison.txt
"""

import argparse
import re
from pathlib import Path


def parse_results(result_file):
    """Parse metrics from a result file."""
    with open(result_file, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract metrics
    mrr_match = re.search(r'MRR\s+(\d+\.\d+)', content)
    recall1_match = re.search(r'Recall@1\s+(\d+\.\d+)', content)
    recall5_match = re.search(r'Recall@5\s+(\d+\.\d+)', content)
    recall10_match = re.search(r'Recall@10\s+(\d+\.\d+)', content)
    
    if mrr_match:
        metrics['mrr'] = float(mrr_match.group(1))
    if recall1_match:
        metrics['recall@1'] = float(recall1_match.group(1))
    if recall5_match:
        metrics['recall@5'] = float(recall5_match.group(1))
    if recall10_match:
        metrics['recall@10'] = float(recall10_match.group(1))
    
    # Extract metadata
    pool_match = re.search(r'Pool file: (.+)', content)
    query_match = re.search(r'Query file: (.+)', content)
    queries_match = re.search(r'Queries: (\d+)', content)
    pool_size_match = re.search(r'Pool size: (\d+)', content)
    
    metadata = {
        'pool_file': pool_match.group(1) if pool_match else 'Unknown',
        'query_file': query_match.group(1) if query_match else 'Unknown',
        'num_queries': int(queries_match.group(1)) if queries_match else 0,
        'pool_size': int(pool_size_match.group(1)) if pool_size_match else 0
    }
    
    return metrics, metadata


def main():
    parser = argparse.ArgumentParser(description='Compare baseline and address-aware evaluation results')
    parser.add_argument('--baseline_result', type=str, required=True,
                        help='Path to baseline evaluation result file')
    parser.add_argument('--addressaware_result', type=str, required=True,
                        help='Path to address-aware evaluation result file')
    parser.add_argument('--output_file', type=str, default='comparison.txt',
                        help='Path to output comparison file')
    
    args = parser.parse_args()
    
    # Parse results
    baseline_metrics, baseline_meta = parse_results(args.baseline_result)
    addressaware_metrics, addressaware_meta = parse_results(args.addressaware_result)
    
    # Calculate improvements
    improvements = {}
    for metric in baseline_metrics:
        baseline_val = baseline_metrics[metric]
        addressaware_val = addressaware_metrics.get(metric, 0)
        if baseline_val > 0:
            improvement = ((addressaware_val - baseline_val) / baseline_val) * 100
        else:
            improvement = 0
        improvements[metric] = improvement
    
    # Print comparison
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"Pool: {baseline_meta['pool_file']}")
    print(f"Queries: {baseline_meta['num_queries']}")
    print(f"Pool size: {baseline_meta['pool_size']}")
    print(f"\n{'Metric':<15} {'Baseline':<15} {'Address-Aware':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for metric in ['mrr', 'recall@1', 'recall@5', 'recall@10']:
        if metric in baseline_metrics and metric in addressaware_metrics:
            baseline_val = baseline_metrics[metric]
            addressaware_val = addressaware_metrics[metric]
            improvement = improvements[metric]
            
            metric_name = metric.upper().replace('@', '@')
            print(f"{metric_name:<15} {baseline_val:<15.4f} {addressaware_val:<15.4f} {improvement:>13.2f}%")
    
    print(f"{'='*80}\n")
    
    # Save comparison
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"Model Comparison Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Baseline result: {args.baseline_result}\n")
        f.write(f"Address-aware result: {args.addressaware_result}\n")
        f.write(f"\nPool file: {baseline_meta['pool_file']}\n")
        f.write(f"Query file: {baseline_meta['query_file']}\n")
        f.write(f"Queries: {baseline_meta['num_queries']}\n")
        f.write(f"Pool size: {baseline_meta['pool_size']}\n")
        f.write(f"\n{'='*80}\n\n")
        f.write(f"{'Metric':<15} {'Baseline':<15} {'Address-Aware':<15} {'Improvement':<15}\n")
        f.write(f"{'-'*60}\n")
        
        for metric in ['mrr', 'recall@1', 'recall@5', 'recall@10']:
            if metric in baseline_metrics and metric in addressaware_metrics:
                baseline_val = baseline_metrics[metric]
                addressaware_val = addressaware_metrics[metric]
                improvement = improvements[metric]
                
                metric_name = metric.upper().replace('@', '@')
                f.write(f"{metric_name:<15} {baseline_val:<15.4f} {addressaware_val:<15.4f} {improvement:>13.2f}%\n")
    
    print(f"Comparison saved to {output_path}")


if __name__ == '__main__':
    main()
