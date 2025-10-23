import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_model_results(results_dir):
    """Load all metrics from a model's results directory."""
    with open(os.path.join(results_dir, 'all_metrics.json'), 'r') as f:
        return json.load(f)

def create_comparison_plots(baseline_results, address_aware_results, output_dir):
    """Create comparison plots for all metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metrics data
    baseline_metrics = baseline_results['metrics']
    address_metrics = address_aware_results['metrics']
    epochs = list(range(len(baseline_metrics['mlm_loss'])))
    
    # Set up plot style
    plt.style.use('seaborn')
    colors = ['#2ecc71', '#e74c3c']  # green for baseline, red for address-aware
    
    # Define metrics to plot and their groupings
    plot_groups = {
        'mlm_metrics': {
            'title': 'Masked Language Model Metrics',
            'metrics': [
                ('mlm_loss', 'MLM Loss'),
                ('perplexity', 'Perplexity')
            ]
        },
        'dfg_metrics': {
            'title': 'DFG Next Sentence Prediction Metrics',
            'metrics': [
                ('dfg_nsp_loss', 'DFG NSP Loss'),
                ('dfg_nsp_acc', 'DFG NSP Accuracy')
            ]
        },
        'cfg_metrics': {
            'title': 'CFG Next Sentence Prediction Metrics',
            'metrics': [
                ('cfg_nsp_loss', 'CFG NSP Loss'),
                ('cfg_nsp_acc', 'CFG NSP Accuracy')
            ]
        }
    }
    
    # Create plots for each group
    for group_name, group_info in plot_groups.items():
        fig, axes = plt.subplots(len(group_info['metrics']), 1, 
                                figsize=(10, 5*len(group_info['metrics'])))
        if len(group_info['metrics']) == 1:
            axes = [axes]
            
        fig.suptitle(group_info['title'], fontsize=16, y=1.02)
        
        for (metric_name, metric_label), ax in zip(group_info['metrics'], axes):
            # Plot baseline
            ax.plot(epochs, baseline_metrics[metric_name], color=colors[0], 
                   marker='o', label='Baseline BERT')
            # Plot address-aware
            ax.plot(epochs, address_metrics[metric_name], color=colors[1], 
                   marker='o', label='Address-aware BERT')
            
            ax.set_title(metric_label)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_label)
            ax.legend()
            ax.grid(True)
            
            # Add value annotations at start and end points
            for color, data in zip(colors, [baseline_metrics[metric_name], 
                                          address_metrics[metric_name]]):
                ax.annotate(f'{data[0]:.3f}', 
                          (0, data[0]),
                          xytext=(5, 5), 
                          textcoords='offset points',
                          color=color)
                ax.annotate(f'{data[-1]:.3f}', 
                          (len(data)-1, data[-1]),
                          xytext=(5, 5), 
                          textcoords='offset points',
                          color=color)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{group_name}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

    # Save comparison summary
    summary = {
        'final_metrics': {
            'baseline': {k: v[-1] for k, v in baseline_metrics.items()},
            'address_aware': {k: v[-1] for k, v in address_metrics.items()}
        },
        'improvement': {
            k: {
                'absolute': address_metrics[k][-1] - baseline_metrics[k][-1],
                'relative': ((address_metrics[k][-1] - baseline_metrics[k][-1]) / 
                           abs(baseline_metrics[k][-1])) * 100
            }
            for k in baseline_metrics.keys()
        }
    }
    
    with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', required=True,
                       help='Directory containing both model evaluation results')
    parser.add_argument('--output_dir', default='comparison_plots',
                       help='Directory to save comparison plots')
    args = parser.parse_args()
    
    # Load results
    baseline_results = load_model_results(os.path.join(args.eval_dir, 'baseline'))
    address_aware_results = load_model_results(os.path.join(args.eval_dir, 'address_aware'))
    
    # Create plots
    create_comparison_plots(baseline_results, address_aware_results, args.output_dir)
    
    print(f"Comparison plots saved in {args.output_dir}")
    print(f"Summary of improvements saved in {os.path.join(args.output_dir, 'comparison_summary.json')}")