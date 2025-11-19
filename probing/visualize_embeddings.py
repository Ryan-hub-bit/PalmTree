"""
Visualize Learned Embeddings

Create visualizations to understand how position information is encoded
in the learned representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import json
import os
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from addressaware.dataloader_paired import PairedAddressAwareDataset
from addressaware.model import AddressAwareBERT
from addressaware.model_baseline import BaselineBERT
from pre_trained_model.vocab import WordVocab
from torch.utils.data import DataLoader


def extract_embeddings_with_positions(model, dataloader, device, is_addressaware=True, max_samples=5000):
    """
    Extract embeddings and position information from model.
    
    Returns:
        embeddings: (num_samples, hidden_size)
        positions: dict with 'binary', 'function', 'bb' arrays
        tokens: (num_samples,) - token indices
    """
    model.eval()
    
    all_embeddings = []
    all_binary_pos = []
    all_function_pos = []
    all_bb_pos = []
    all_tokens = []
    
    total_collected = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if total_collected >= max_samples:
                break
            
            if is_addressaware:
                cfg_input = batch['cfg_bert_input'].to(device)
                cfg_segment = batch['cfg_segment_label'].to(device)
                cfg_binary_pos = batch['cfg_binary_pos'].to(device)
                cfg_function_pos = batch['cfg_function_pos'].to(device)
                cfg_bb_pos = batch['cfg_bb_pos'].to(device)
                
                # Get embeddings
                embeddings = model.bert.embedding(
                    cfg_input, cfg_segment,
                    cfg_binary_pos, cfg_function_pos, cfg_bb_pos
                )
                
                # Pass through transformer
                for layer in model.bert.transformer_blocks:
                    embeddings = layer(embeddings, mask=None)
                
                # Mask for valid tokens
                mask = (cfg_input != model.bert.embedding.token.padding_idx)
                
                # Extract valid embeddings and positions
                for i in range(embeddings.size(0)):
                    valid_idx = mask[i].nonzero(as_tuple=True)[0]
                    
                    for idx in valid_idx:
                        if total_collected >= max_samples:
                            break
                        
                        all_embeddings.append(embeddings[i, idx].cpu().numpy())
                        all_binary_pos.append(cfg_binary_pos[i, idx].cpu().item())
                        all_function_pos.append(cfg_function_pos[i, idx].cpu().item())
                        all_bb_pos.append(cfg_bb_pos[i, idx].cpu().item())
                        all_tokens.append(cfg_input[i, idx].cpu().item())
                        total_collected += 1
            
            else:  # Baseline
                cfg_input = batch['bert_input'].to(device)
                cfg_segment = batch['segment_label'].to(device)
                
                # Get embeddings
                embeddings = model.bert.embedding(cfg_input, cfg_segment)
                
                # Pass through transformer
                for layer in model.bert.transformer_blocks:
                    embeddings = layer(embeddings, mask=None)
                
                # Mask for valid tokens
                mask = (cfg_input != model.bert.embedding.token.padding_idx)
                
                # Extract valid embeddings (no position ground truth)
                for i in range(embeddings.size(0)):
                    valid_idx = mask[i].nonzero(as_tuple=True)[0]
                    
                    for idx in valid_idx:
                        if total_collected >= max_samples:
                            break
                        
                        all_embeddings.append(embeddings[i, idx].cpu().numpy())
                        all_binary_pos.append(0.0)  # No ground truth
                        all_function_pos.append(0.0)
                        all_bb_pos.append(0.0)
                        all_tokens.append(cfg_input[i, idx].cpu().item())
                        total_collected += 1
    
    embeddings = np.array(all_embeddings)
    positions = {
        'binary': np.array(all_binary_pos),
        'function': np.array(all_function_pos),
        'bb': np.array(all_bb_pos)
    }
    tokens = np.array(all_tokens)
    
    return embeddings, positions, tokens


def plot_tsne_by_position(embeddings, positions, level, output_path, title_prefix=""):
    """
    Create t-SNE visualization colored by position values.
    
    Args:
        embeddings: (num_samples, hidden_size)
        positions: position values (num_samples,)
        level: 'binary', 'function', or 'bb'
        output_path: Path to save figure
        title_prefix: Prefix for plot title
    """
    print(f"Computing t-SNE for {level} level...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=positions,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label=f'{level.capitalize()} Position')
    plt.title(f'{title_prefix}t-SNE Visualization Colored by {level.upper()} Position')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved to: {output_path}")


def plot_pca_by_position(embeddings, positions, level, output_path, title_prefix=""):
    """Create PCA visualization colored by position values."""
    print(f"Computing PCA for {level} level...")
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=positions,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label=f'{level.capitalize()} Position')
    plt.title(f'{title_prefix}PCA Visualization Colored by {level.upper()} Position')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved to: {output_path}")


def plot_position_distribution(positions, level, output_path, title_prefix=""):
    """Plot distribution of position values."""
    plt.figure(figsize=(10, 6))
    plt.hist(positions, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel(f'{level.capitalize()} Position')
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix}Distribution of {level.upper()} Positions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize learned embeddings')
    parser.add_argument('--addressaware_model', type=str, required=True)
    parser.add_argument('--baseline_model', type=str, required=True)
    parser.add_argument('--test_cfg', type=str,
                        default='./test_data/all_cfg_probing.txt',
                        help='Path to test CFG data')
    parser.add_argument('--test_dfg', type=str,
                        default='./test_data/all_dfg_probing.txt',
                        help='Path to test DFG data')
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./visualizations')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print(f"\nLoading vocabulary from: {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    
    # Load test data
    print(f"\nLoading test data...")
    test_dataset = PairedAddressAwareDataset(
        cfg_corpus_path=args.test_cfg,
        dfg_corpus_path=args.test_dfg,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        data_percentage=0.05  # Use 5% for faster visualization
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # ========================================================================
    # VISUALIZE ADDRESS-AWARE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("VISUALIZING ADDRESS-AWARE MODEL")
    print("="*80)
    
    # Load model
    addressaware_model = AddressAwareBERT(
        len(vocab), hidden=args.hidden_size, n_layers=12, attn_heads=8, max_len=args.seq_len
    ).to(device)
    checkpoint = torch.load(args.addressaware_model, map_location=device)
    addressaware_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract embeddings
    aa_embeddings, aa_positions, aa_tokens = extract_embeddings_with_positions(
        addressaware_model, test_loader, device, is_addressaware=True, max_samples=args.max_samples
    )
    
    print(f"\nExtracted {len(aa_embeddings)} embeddings from address-aware model")
    
    # Create visualizations
    aa_dir = os.path.join(args.output_dir, 'addressaware')
    os.makedirs(aa_dir, exist_ok=True)
    
    for level in ['binary', 'function', 'bb']:
        # t-SNE
        plot_tsne_by_position(
            aa_embeddings, aa_positions[level], level,
            os.path.join(aa_dir, f'tsne_{level}.png'),
            title_prefix="Address-Aware: "
        )
        
        # PCA
        plot_pca_by_position(
            aa_embeddings, aa_positions[level], level,
            os.path.join(aa_dir, f'pca_{level}.png'),
            title_prefix="Address-Aware: "
        )
        
        # Distribution
        plot_position_distribution(
            aa_positions[level], level,
            os.path.join(aa_dir, f'dist_{level}.png'),
            title_prefix="Address-Aware: "
        )
    
    # ========================================================================
    # VISUALIZE BASELINE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("VISUALIZING BASELINE MODEL")
    print("="*80)
    
    # Load model
    baseline_model = BaselineBERT(
        len(vocab), hidden=args.hidden_size, n_layers=12, attn_heads=8, max_len=args.seq_len
    ).to(device)
    checkpoint = torch.load(args.baseline_model, map_location=device)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract embeddings
    bl_embeddings, bl_positions, bl_tokens = extract_embeddings_with_positions(
        baseline_model, test_loader, device, is_addressaware=False, max_samples=args.max_samples
    )
    
    print(f"\nExtracted {len(bl_embeddings)} embeddings from baseline model")
    
    # Create visualizations
    bl_dir = os.path.join(args.output_dir, 'baseline')
    os.makedirs(bl_dir, exist_ok=True)
    
    # Note: Baseline has no position ground truth, so we use address-aware positions for coloring
    for level in ['binary', 'function', 'bb']:
        # t-SNE
        plot_tsne_by_position(
            bl_embeddings, aa_positions[level], level,
            os.path.join(bl_dir, f'tsne_{level}.png'),
            title_prefix="Baseline: "
        )
        
        # PCA
        plot_pca_by_position(
            bl_embeddings, aa_positions[level], level,
            os.path.join(bl_dir, f'pca_{level}.png'),
            title_prefix="Baseline: "
        )
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
