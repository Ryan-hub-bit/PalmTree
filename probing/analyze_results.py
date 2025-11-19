"""
Analyze and Compare Probing Results

Load probing results and create comparison tables and plots.
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_results(results_path):
    """Load probing results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "="*80)
    print("POSITION PREDICTION PROBING RESULTS")
    print("="*80)
    
    print("\nAddress-Aware Model:")
    print("-" * 80)
    print(f"{'Level':<12} {'MAE':<12} {'RMSE':<12} {'MSE':<12}")
    print("-" * 80)
    
    for level in ['binary', 'function', 'bb']:
        metrics = results['addressaware'][level]
        print(f"{level.upper():<12} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f} {metrics['mse']:<12.4f}")
    
    print("\n" + "-" * 80)
    print("Baseline Model:")
    print("-" * 80)
    print(f"{'Level':<12} {'MAE':<12} {'RMSE':<12} {'MSE':<12}")
    print("-" * 80)
    
    for level in ['binary', 'function', 'bb']:
        metrics = results['baseline'][level]
        print(f"{level.upper():<12} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f} {metrics['mse']:<12.4f}")
    
    print("\n" + "="*80)
    print("COMPARISON: Address-Aware vs Baseline")
    print("="*80)
    print(f"{'Level':<12} {'AA MAE':<12} {'BL MAE':<12} {'Improvement':<12}")
    print("-" * 80)
    
    for level in ['binary', 'function', 'bb']:
        aa_mae = results['addressaware'][level]['mae']
        bl_mae = results['baseline'][level]['mae']
        improvement = ((bl_mae - aa_mae) / bl_mae) * 100 if bl_mae > 0 else 0
        
        print(f"{level.upper():<12} {aa_mae:<12.4f} {bl_mae:<12.4f} {improvement:+.2f}%")
    
    print("="*80)


def plot_comparison(results, output_path):
    """Create bar plot comparing MAE across models and levels."""
    levels = ['Binary', 'Function', 'BB']
    
    aa_maes = [results['addressaware']['binary']['mae'],
               results['addressaware']['function']['mae'],
               results['addressaware']['bb']['mae']]
    
    bl_maes = [results['baseline']['binary']['mae'],
               results['baseline']['function']['mae'],
               results['baseline']['bb']['mae']]
    
    x = np.arange(len(levels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, aa_maes, width, label='Address-Aware', color='steelblue')
    rects2 = ax.bar(x + width/2, bl_maes, width, label='Baseline', color='coral')
    
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('Position Prediction Error: Address-Aware vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison plot saved to: {output_path}")


def plot_improvement(results, output_path):
    """Create bar plot showing improvement percentage."""
    levels = ['Binary', 'Function', 'BB']
    improvements = []
    
    for level_key in ['binary', 'function', 'bb']:
        aa_mae = results['addressaware'][level_key]['mae']
        bl_mae = results['baseline'][level_key]['mae']
        improvement = ((bl_mae - aa_mae) / bl_mae) * 100 if bl_mae > 0 else 0
        improvements.append(improvement)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(levels, improvements, color=['green' if x > 0 else 'red' for x in improvements])
    
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Position Prediction Improvement: Address-Aware vs Baseline')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.1f}%',
               ha='center', va='bottom' if val > 0 else 'top',
               fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Improvement plot saved to: {output_path}")


def create_summary_report(results, output_path):
    """Create a text summary report."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("POSITION EMBEDDING PROBING - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OBJECTIVE:\n")
        f.write("-" * 80 + "\n")
        f.write("Evaluate how well learned embeddings encode positional information\n")
        f.write("at three levels: binary, function, and basic block.\n\n")
        
        f.write("METHOD:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Extract contextualized embeddings from pretrained models\n")
        f.write("2. Train linear probes to predict position values from embeddings\n")
        f.write("3. Evaluate prediction accuracy using MAE and RMSE\n\n")
        
        f.write("RESULTS:\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Address-Aware Model:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Level':<12} {'MAE':<12} {'RMSE':<12}\n")
        f.write("-" * 80 + "\n")
        for level in ['binary', 'function', 'bb']:
            metrics = results['addressaware'][level]
            f.write(f"{level.upper():<12} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f}\n")
        
        f.write("\n")
        f.write("Baseline Model:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Level':<12} {'MAE':<12} {'RMSE':<12}\n")
        f.write("-" * 80 + "\n")
        for level in ['binary', 'function', 'bb']:
            metrics = results['baseline'][level]
            f.write(f"{level.upper():<12} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f}\n")
        
        f.write("\n")
        f.write("COMPARISON:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Level':<12} {'AA MAE':<12} {'BL MAE':<12} {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")
        
        for level in ['binary', 'function', 'bb']:
            aa_mae = results['addressaware'][level]['mae']
            bl_mae = results['baseline'][level]['mae']
            improvement = ((bl_mae - aa_mae) / bl_mae) * 100 if bl_mae > 0 else 0
            f.write(f"{level.upper():<12} {aa_mae:<12.4f} {bl_mae:<12.4f} {improvement:+.2f}%\n")
        
        f.write("\n")
        f.write("INTERPRETATION:\n")
        f.write("=" * 80 + "\n")
        
        # Calculate average improvement
        avg_improvement = 0
        for level in ['binary', 'function', 'bb']:
            aa_mae = results['addressaware'][level]['mae']
            bl_mae = results['baseline'][level]['mae']
            improvement = ((bl_mae - aa_mae) / bl_mae) * 100 if bl_mae > 0 else 0
            avg_improvement += improvement
        avg_improvement /= 3
        
        if avg_improvement > 10:
            f.write("\n✓ STRONG EVIDENCE: Address-aware embeddings encode position information\n")
            f.write(f"  Average improvement: {avg_improvement:.2f}%\n")
            f.write("  The address-aware model learns meaningful positional representations\n")
            f.write("  that go beyond sequential ordering.\n")
        elif avg_improvement > 0:
            f.write("\n✓ MODERATE EVIDENCE: Some positional information is captured\n")
            f.write(f"  Average improvement: {avg_improvement:.2f}%\n")
            f.write("  Address embeddings provide some benefit, but effect is limited.\n")
        else:
            f.write("\n✗ NO EVIDENCE: Address embeddings do not improve position encoding\n")
            f.write(f"  Average improvement: {avg_improvement:.2f}%\n")
            f.write("  Baseline performs equally well or better.\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Summary report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze probing results')
    parser.add_argument('--results', type=str, default='./results/probing_results.json',
                        help='Path to probing results JSON file')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Output directory for analysis plots and reports')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)
    
    # Print results table
    print_results_table(results)
    
    # Create comparison plot
    plot_comparison(results, os.path.join(args.output_dir, 'mae_comparison.png'))
    
    # Create improvement plot
    plot_improvement(results, os.path.join(args.output_dir, 'improvement.png'))
    
    # Create summary report
    create_summary_report(results, os.path.join(args.output_dir, 'summary_report.txt'))
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
