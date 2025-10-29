"""
Compare evaluation metrics between windows8noaddr (no addresses) and windows8 (with addresses).
Generates comparison graphs to visualize the impact of address embeddings.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_metrics(result_dir):
    """Load all epoch metrics from a result directory."""
    metrics = {
        'epochs': [],
        'total_loss': [],
        'mlm_loss': [],
        'perplexity': [],
        'dfg_nsp_loss': [],
        'cfg_nsp_loss': [],
        'dfg_nsp_acc': [],
        'cfg_nsp_acc': []
    }
    
    # Find all epoch_XX.json files
    epoch_files = []
    for fname in os.listdir(result_dir):
        if fname.startswith('epoch_') and fname.endswith('.json'):
            epoch_num = int(fname.replace('epoch_', '').replace('.json', ''))
            epoch_files.append((epoch_num, fname))
    
    epoch_files.sort(key=lambda x: x[0])
    
    for epoch_num, fname in epoch_files:
        fpath = os.path.join(result_dir, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
            metrics['epochs'].append(epoch_num)
            metrics['total_loss'].append(data.get('total_loss', 0))
            metrics['mlm_loss'].append(data.get('mlm_loss', 0))
            metrics['perplexity'].append(data.get('perplexity', 0))
            metrics['dfg_nsp_loss'].append(data.get('dfg_nsp_loss', 0))
            metrics['cfg_nsp_loss'].append(data.get('cfg_nsp_loss', 0))
            metrics['dfg_nsp_acc'].append(data.get('dfg_nsp_acc', 0))
            metrics['cfg_nsp_acc'].append(data.get('cfg_nsp_acc', 0))
    
    return metrics


def plot_comparison(noaddr_metrics, addr_metrics, output_dir):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    epochs_noaddr = noaddr_metrics['epochs']
    epochs_addr = addr_metrics['epochs']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison: windows8noaddr vs windows8 (with addresses)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs_noaddr, noaddr_metrics['total_loss'], 'b-o', label='No Address', linewidth=2)
    ax.plot(epochs_addr, addr_metrics['total_loss'], 'r-s', label='With Address', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MLM Loss
    ax = axes[0, 1]
    ax.plot(epochs_noaddr, noaddr_metrics['mlm_loss'], 'b-o', label='No Address', linewidth=2)
    ax.plot(epochs_addr, addr_metrics['mlm_loss'], 'r-s', label='With Address', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MLM Loss')
    ax.set_title('Masked Language Model Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Perplexity
    ax = axes[1, 0]
    ax.plot(epochs_noaddr, noaddr_metrics['perplexity'], 'b-o', label='No Address', linewidth=2)
    ax.plot(epochs_addr, addr_metrics['perplexity'], 'r-s', label='With Address', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: DFG NSP Accuracy
    ax = axes[1, 1]
    ax.plot(epochs_noaddr, noaddr_metrics['dfg_nsp_acc'], 'b-o', label='No Address', linewidth=2)
    ax.plot(epochs_addr, addr_metrics['dfg_nsp_acc'], 'r-s', label='With Address', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('DFG Next Sentence Prediction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 5: CFG NSP Accuracy
    ax = axes[2, 0]
    ax.plot(epochs_noaddr, noaddr_metrics['cfg_nsp_acc'], 'b-o', label='No Address', linewidth=2)
    ax.plot(epochs_addr, addr_metrics['cfg_nsp_acc'], 'r-s', label='With Address', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('CFG Next Sentence Prediction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 6: NSP Loss Comparison
    ax = axes[2, 1]
    ax.plot(epochs_noaddr, noaddr_metrics['dfg_nsp_loss'], 'b-o', label='DFG NSP (No Addr)', linewidth=2)
    ax.plot(epochs_noaddr, noaddr_metrics['cfg_nsp_loss'], 'b-s', label='CFG NSP (No Addr)', linewidth=2)
    ax.plot(epochs_addr, addr_metrics['dfg_nsp_loss'], 'r-o', label='DFG NSP (With Addr)', linewidth=2, alpha=0.7)
    ax.plot(epochs_addr, addr_metrics['cfg_nsp_loss'], 'r-s', label='CFG NSP (With Addr)', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NSP Loss')
    ax.set_title('Next Sentence Prediction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_file}")
    plt.close()
    
    # Create a summary comparison table
    create_summary_table(noaddr_metrics, addr_metrics, output_dir)


def create_summary_table(noaddr_metrics, addr_metrics, output_dir):
    """Create a summary comparison table for best epochs."""
    # Find best epoch for each model (lowest total loss)
    best_noaddr_idx = np.argmin(noaddr_metrics['total_loss'])
    best_addr_idx = np.argmin(addr_metrics['total_loss'])
    
    summary = {
        'windows8noaddr (no addresses)': {
            'best_epoch': noaddr_metrics['epochs'][best_noaddr_idx],
            'total_loss': noaddr_metrics['total_loss'][best_noaddr_idx],
            'mlm_loss': noaddr_metrics['mlm_loss'][best_noaddr_idx],
            'perplexity': noaddr_metrics['perplexity'][best_noaddr_idx],
            'dfg_nsp_acc': noaddr_metrics['dfg_nsp_acc'][best_noaddr_idx],
            'cfg_nsp_acc': noaddr_metrics['cfg_nsp_acc'][best_noaddr_idx],
        },
        'windows8 (with addresses)': {
            'best_epoch': addr_metrics['epochs'][best_addr_idx],
            'total_loss': addr_metrics['total_loss'][best_addr_idx],
            'mlm_loss': addr_metrics['mlm_loss'][best_addr_idx],
            'perplexity': addr_metrics['perplexity'][best_addr_idx],
            'dfg_nsp_acc': addr_metrics['dfg_nsp_acc'][best_addr_idx],
            'cfg_nsp_acc': addr_metrics['cfg_nsp_acc'][best_addr_idx],
        }
    }
    
    # Calculate improvements
    improvements = {
        'total_loss_diff': summary['windows8 (with addresses)']['total_loss'] - summary['windows8noaddr (no addresses)']['total_loss'],
        'mlm_loss_diff': summary['windows8 (with addresses)']['mlm_loss'] - summary['windows8noaddr (no addresses)']['mlm_loss'],
        'perplexity_diff': summary['windows8 (with addresses)']['perplexity'] - summary['windows8noaddr (no addresses)']['perplexity'],
        'dfg_nsp_acc_diff': summary['windows8 (with addresses)']['dfg_nsp_acc'] - summary['windows8noaddr (no addresses)']['dfg_nsp_acc'],
        'cfg_nsp_acc_diff': summary['windows8 (with addresses)']['cfg_nsp_acc'] - summary['windows8noaddr (no addresses)']['cfg_nsp_acc'],
    }
    
    summary['improvements'] = improvements
    
    # Save as JSON
    summary_file = os.path.join(output_dir, 'comparison_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved comparison summary: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"\nwindows8noaddr (NO ADDRESSES) - Best Epoch: {summary['windows8noaddr (no addresses)']['best_epoch']}")
    print(f"  Total Loss:     {summary['windows8noaddr (no addresses)']['total_loss']:.4f}")
    print(f"  MLM Loss:       {summary['windows8noaddr (no addresses)']['mlm_loss']:.4f}")
    print(f"  Perplexity:     {summary['windows8noaddr (no addresses)']['perplexity']:.2f}")
    print(f"  DFG NSP Acc:    {summary['windows8noaddr (no addresses)']['dfg_nsp_acc']:.4f}")
    print(f"  CFG NSP Acc:    {summary['windows8noaddr (no addresses)']['cfg_nsp_acc']:.4f}")
    
    print(f"\nwindows8 (WITH ADDRESSES) - Best Epoch: {summary['windows8 (with addresses)']['best_epoch']}")
    print(f"  Total Loss:     {summary['windows8 (with addresses)']['total_loss']:.4f}")
    print(f"  MLM Loss:       {summary['windows8 (with addresses)']['mlm_loss']:.4f}")
    print(f"  Perplexity:     {summary['windows8 (with addresses)']['perplexity']:.2f}")
    print(f"  DFG NSP Acc:    {summary['windows8 (with addresses)']['dfg_nsp_acc']:.4f}")
    print(f"  CFG NSP Acc:    {summary['windows8 (with addresses)']['cfg_nsp_acc']:.4f}")
    
    print("\nIMPROVEMENTS (WITH ADDRESSES - NO ADDRESSES):")
    print(f"  Total Loss:     {improvements['total_loss_diff']:+.4f} {'✓ Better' if improvements['total_loss_diff'] < 0 else '✗ Worse'}")
    print(f"  MLM Loss:       {improvements['mlm_loss_diff']:+.4f} {'✓ Better' if improvements['mlm_loss_diff'] < 0 else '✗ Worse'}")
    print(f"  Perplexity:     {improvements['perplexity_diff']:+.2f} {'✓ Better' if improvements['perplexity_diff'] < 0 else '✗ Worse'}")
    print(f"  DFG NSP Acc:    {improvements['dfg_nsp_acc_diff']:+.4f} {'✓ Better' if improvements['dfg_nsp_acc_diff'] > 0 else '✗ Worse'}")
    print(f"  CFG NSP Acc:    {improvements['cfg_nsp_acc_diff']:+.4f} {'✓ Better' if improvements['cfg_nsp_acc_diff'] > 0 else '✗ Worse'}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare windows8noaddr vs windows8 models')
    parser.add_argument('--noaddr_dir', type=str, 
                        default='/home/louie/PalmTree/evaluation_result/windows8noaddr',
                        help='Directory with windows8noaddr evaluation results')
    parser.add_argument('--addr_dir', type=str,
                        default='/home/louie/PalmTree/evaluation_result/windows8',
                        help='Directory with windows8 evaluation results')
    parser.add_argument('--output_dir', type=str,
                        default='/home/louie/PalmTree/evaluation_result/comparison',
                        help='Directory to save comparison results')
    args = parser.parse_args()
    
    print("Loading metrics...")
    print(f"  No-address model: {args.noaddr_dir}")
    print(f"  Address-aware model: {args.addr_dir}")
    
    noaddr_metrics = load_metrics(args.noaddr_dir)
    addr_metrics = load_metrics(args.addr_dir)
    
    print(f"\nLoaded {len(noaddr_metrics['epochs'])} epochs for no-address model")
    print(f"Loaded {len(addr_metrics['epochs'])} epochs for address-aware model")
    
    print("\nGenerating comparison plots...")
    plot_comparison(noaddr_metrics, addr_metrics, args.output_dir)
    
    print(f"\nComparison complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
