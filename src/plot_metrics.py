import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_model_results(results_dir, model_type):
    """Load all metrics for a model type."""
    with open(os.path.join(results_dir, model_type, 'all_metrics.json'), 'r') as f:
        return json.load(f)

def plot_metrics(baseline_data, address_data, output_dir):
    """Create comparison plots for all metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    baseline_metrics = baseline_data['metrics']
    address_metrics = address_data['metrics']
    epochs = range(len(baseline_metrics['mlm_loss']))
    
    # Plot settings
    plt.style.use('seaborn')
    colors = ['#2ecc71', '#e74c3c']  # green for baseline, red for address-aware
    
    # Plot groups
    plot_groups = [
        {
            'name': 'language_model_metrics',
            'title': 'Language Model Performance',
            'metrics': [
                ('mlm_loss', 'MLM Loss'),
                ('perplexity', 'Perplexity')
            ]
        },
        {
            'name': 'dfg_metrics',
            'title': 'Data Flow Graph (DFG) Metrics',
            'metrics': [
                ('dfg_nsp_loss', 'DFG NSP Loss'),
                ('dfg_nsp_acc', 'DFG NSP Accuracy')
            ]
        },
        {
            'name': 'cfg_metrics',
            'title': 'Control Flow Graph (CFG) Metrics',
            'metrics': [
                ('cfg_nsp_loss', 'CFG NSP Loss'),
                ('cfg_nsp_acc', 'CFG NSP Accuracy')
            ]
        }
    ]
    
    for group in plot_groups:
        # Create subplot for each metric group
        fig, axes = plt.subplots(len(group['metrics']), 1, figsize=(12, 6*len(group['metrics'])))
        if len(group['metrics']) == 1:
            axes = [axes]
            
        fig.suptitle(group['title'], fontsize=16, y=1.02)
        
        for (metric_name, metric_label), ax in zip(group['metrics'], axes):
            # Plot baseline
            ax.plot(epochs, baseline_metrics[metric_name], color=colors[0], 
                   marker='o', label='Baseline BERT', linewidth=2)
            # Plot address-aware
            ax.plot(epochs, address_metrics[metric_name], color=colors[1], 
                   marker='o', label='Address-aware BERT', linewidth=2)
            
            # Styling
            ax.set_title(metric_label, fontsize=14, pad=10)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_label, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(labelsize=10)
            
            # Add annotations for start and end values
            for color, data, model in zip(colors, 
                                        [baseline_metrics[metric_name], 
                                         address_metrics[metric_name]],
                                        ['Baseline', 'Address-aware']):
                # Start value
                ax.annotate(f'{model}: {data[0]:.3f}', 
                          (0, data[0]),
                          xytext=(10, 10), 
                          textcoords='offset points',
                          color=color,
                          fontsize=9,
                          bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))
                # End value
                ax.annotate(f'{model}: {data[-1]:.3f}', 
                          (len(data)-1, data[-1]),
                          xytext=(10, -10), 
                          textcoords='offset points',
                          color=color,
                          fontsize=9,
                          bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{group["name"]}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    # Save improvement summary
    improvement_summary = {
        'metrics_improvement': {
            metric: {
                'start': {
                    'baseline': baseline_metrics[metric][0],
                    'address_aware': address_metrics[metric][0],
                    'difference': address_metrics[metric][0] - baseline_metrics[metric][0],
                    'relative_improvement': ((address_metrics[metric][0] - baseline_metrics[metric][0]) / 
                                          abs(baseline_metrics[metric][0])) * 100
                },
                'end': {
                    'baseline': baseline_metrics[metric][-1],
                    'address_aware': address_metrics[metric][-1],
                    'difference': address_metrics[metric][-1] - baseline_metrics[metric][-1],
                    'relative_improvement': ((address_metrics[metric][-1] - baseline_metrics[metric][-1]) / 
                                          abs(baseline_metrics[metric][-1])) * 100
                }
            }
            for metric in baseline_metrics.keys()
        }
    }
    
    with open(os.path.join(output_dir, 'improvement_summary.json'), 'w') as f:
        json.dump(improvement_summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', required=True,
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', default='comparison_plots',
                       help='Directory to save comparison plots')
    args = parser.parse_args()
    
    # Load results for both models
    baseline_data = load_model_results(args.eval_dir, 'baseline')
    address_data = load_model_results(args.eval_dir, 'address_aware')
    
    # Create comparison plots
    plot_metrics(baseline_data, address_data, args.output_dir)
    
    print(f"Comparison plots saved in {args.output_dir}")
    print(f"Improvement summary saved in {os.path.join(args.output_dir, 'improvement_summary.json')}")