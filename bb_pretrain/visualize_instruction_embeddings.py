"""
Visualize instruction-level embeddings using t-SNE
Compare Vanilla PalmTree vs Address-Aware PalmTree embeddings
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
import os
from collections import defaultdict
import json

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pre-trained_model'))

import config
from data_loader import BBPairDataset
from models.addr_palmtree import AddressAwarePalmTree, load_pretrained_palmtree
from vocab import WordVocab


# Instruction samples for visualization
SAMPLE_INSTRUCTIONS = [
    # Data movement
    "mov rax rbx",
    "mov rdi rsi",
    "mov ecx edx",
    "lea rax [ rip + 0x1000 ]",
    "push rbp",
    "pop rbp",
    
    # Arithmetic
    "add rax rbx",
    "add rdi 0x10",
    "sub rsp 0x20",
    "sub rax rdi",
    "inc rax",
    "dec rcx",
    
    # Logical
    "and rax 0xff",
    "or rdi rsi",
    "xor rax rax",
    "test rax rax",
    "shl rax 0x2",
    "shr rbx 0x1",
    
    # Control flow
    "call 0x401000",
    "call [ rax ]",
    "jmp 0x402000",
    "je 0x401050",
    "jne 0x401060",
    "ret",
    
    # Comparison
    "cmp rax rbx",
    "cmp rdi 0x0",
    
    # Common patterns
    "mov [ rsp ] rax",
    "mov rax [ rsp + 0x8 ]",
    "lea rax [ rbp - 0x10 ]",
]

# Opcode groups for coloring
OPCODE_GROUPS = {
    'mov': 'data_movement',
    'lea': 'data_movement',
    'push': 'stack_ops',
    'pop': 'stack_ops',
    'add': 'arithmetic',
    'sub': 'arithmetic',
    'inc': 'arithmetic',
    'dec': 'arithmetic',
    'and': 'logical',
    'or': 'logical',
    'xor': 'logical',
    'test': 'logical',
    'shl': 'logical',
    'shr': 'logical',
    'call': 'control_flow',
    'jmp': 'control_flow',
    'je': 'control_flow',
    'jne': 'control_flow',
    'ret': 'control_flow',
    'cmp': 'comparison',
}

GROUP_COLORS = {
    'data_movement': '#1f77b4',
    'stack_ops': '#ff7f0e',
    'arithmetic': '#2ca02c',
    'logical': '#d62728',
    'control_flow': '#9467bd',
    'comparison': '#8c564b',
    'other': '#7f7f7f',
}


def tokenize_instruction(instruction, vocab):
    """Tokenize instruction into IDs"""
    tokens = instruction.split()
    
    if hasattr(vocab, 'stoi'):
        # WordVocab
        ids = [vocab.stoi.get(token, vocab.unk_index) for token in tokens]
    else:
        # Dict vocab
        ids = [vocab.get(token, 1) for token in tokens]
    
    return ids


def get_instruction_embedding(model, instruction, vocab, addr_vocab, device, model_type='vanilla'):
    """Get embedding for a single instruction"""
    # Tokenize
    token_ids = tokenize_instruction(instruction, vocab)
    
    # Convert to tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    if model_type == 'address_aware':
        # Create dummy address encodings for address-aware model
        seq_len = input_ids.shape[1]
        address_encodings = torch.zeros((1, seq_len, config.ADDRESS_ENCODING_DIM)).to(device)
        addr_type_ids = torch.ones((1, seq_len), dtype=torch.long).to(device) * addr_vocab.unk_index
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, address_encodings, addr_type_ids)
    else:
        # Vanilla model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
    
    # Get embeddings - average over sequence
    embeddings = outputs['hidden_states']  # [1, seq_len, hidden_dim]
    instruction_embedding = torch.mean(embeddings, dim=1).squeeze(0)  # [hidden_dim]
    
    return instruction_embedding.cpu().numpy()


def visualize_embeddings(vanilla_embeddings, addr_embeddings, instructions, output_dir):
    """Visualize embeddings using t-SNE"""
    
    # Prepare data
    all_embeddings = np.vstack([vanilla_embeddings, addr_embeddings])
    labels = ['Vanilla'] * len(vanilla_embeddings) + ['Address-Aware'] * len(addr_embeddings)
    
    # Get opcode groups
    groups = []
    for inst in instructions + instructions:  # Duplicate for both models
        opcode = inst.split()[0].lower()
        group = OPCODE_GROUPS.get(opcode, 'other')
        groups.append(group)
    
    print(f"\nReducing dimensionality of {len(all_embeddings)} embeddings...")
    
    # Apply PCA first to reduce dimensions (faster t-SNE)
    print("Applying PCA...")
    pca = PCA(n_components=min(50, all_embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(all_embeddings)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # Split back into vanilla and address-aware
    split_idx = len(vanilla_embeddings)
    vanilla_2d = embeddings_2d[:split_idx]
    addr_2d = embeddings_2d[split_idx:]
    vanilla_groups = groups[:split_idx]
    addr_groups = groups[split_idx:]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Vanilla PalmTree by opcode group
    ax = axes[0, 0]
    for group in set(vanilla_groups):
        mask = [g == group for g in vanilla_groups]
        ax.scatter(vanilla_2d[mask, 0], vanilla_2d[mask, 1], 
                  c=GROUP_COLORS.get(group, '#7f7f7f'),
                  label=group, alpha=0.6, s=100)
    ax.set_title('Vanilla PalmTree - Instruction Embeddings by Opcode Group', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 2. Address-Aware PalmTree by opcode group
    ax = axes[0, 1]
    for group in set(addr_groups):
        mask = [g == group for g in addr_groups]
        ax.scatter(addr_2d[mask, 0], addr_2d[mask, 1],
                  c=GROUP_COLORS.get(group, '#7f7f7f'),
                  label=group, alpha=0.6, s=100)
    ax.set_title('Address-Aware PalmTree - Instruction Embeddings by Opcode Group', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 3. Both models - compare same instructions
    ax = axes[1, 0]
    for i, (v_point, a_point) in enumerate(zip(vanilla_2d, addr_2d)):
        group = vanilla_groups[i]
        color = GROUP_COLORS.get(group, '#7f7f7f')
        
        # Draw line connecting same instruction in both models
        ax.plot([v_point[0], a_point[0]], [v_point[1], a_point[1]], 
               'k-', alpha=0.2, linewidth=0.5)
        
        # Plot points
        ax.scatter(v_point[0], v_point[1], c=color, marker='o', s=100, alpha=0.6, edgecolors='black')
        ax.scatter(a_point[0], a_point[1], c=color, marker='^', s=100, alpha=0.6, edgecolors='black')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Vanilla'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Address-Aware'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title('Comparison: Vanilla vs Address-Aware (lines connect same instruction)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Distance analysis - which instructions moved most?
    ax = axes[1, 1]
    distances = np.sqrt(np.sum((vanilla_2d - addr_2d) ** 2, axis=1))
    sorted_indices = np.argsort(distances)[::-1][:15]  # Top 15 changes
    
    instructions_shortened = [inst[:30] for inst in instructions]
    top_changed = [instructions_shortened[i] for i in sorted_indices]
    top_distances = [distances[i] for i in sorted_indices]
    top_groups = [vanilla_groups[i] for i in sorted_indices]
    
    colors = [GROUP_COLORS.get(g, '#7f7f7f') for g in top_groups]
    y_pos = np.arange(len(top_changed))
    
    ax.barh(y_pos, top_distances, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_changed, fontsize=8)
    ax.set_xlabel('Embedding Distance (t-SNE space)', fontsize=12)
    ax.set_title('Instructions with Largest Embedding Changes\n(Vanilla → Address-Aware)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'instruction_embeddings_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    
    # Also create a focused plot for control flow instructions
    create_control_flow_plot(vanilla_2d, addr_2d, vanilla_groups, instructions, output_dir)
    
    plt.show()
    
    return embeddings_2d


def create_control_flow_plot(vanilla_2d, addr_2d, groups, instructions, output_dir):
    """Create focused visualization for multiple instruction types with multiple graph types"""
    from matplotlib.patches import Ellipse
    from scipy.spatial import ConvexHull
    import matplotlib.patches as mpatches
    
    # Instead of filtering, use ALL instructions and their groups
    # This shows control_flow, arithmetic, logical, data_movement, etc. all together
    all_vanilla = vanilla_2d
    all_addr = addr_2d
    all_groups = groups
    all_instructions = instructions
    
    if len(all_vanilla) == 0:
        return
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ========== 1. Original with connecting lines ==========
    ax1 = fig.add_subplot(gs[0, 0])
    for v_point, a_point in zip(all_vanilla, all_addr):
        ax1.plot([v_point[0], a_point[0]], [v_point[1], a_point[1]], 
               'k-', alpha=0.3, linewidth=1)
    
    # Plot by group with different colors
    for group in set(all_groups):
        mask = [g == group for g in all_groups]
        v_points = all_vanilla[mask]
        a_points = all_addr[mask]
        color = GROUP_COLORS.get(group, '#7f7f7f')
        
        ax1.scatter(v_points[:, 0], v_points[:, 1], 
                  c=color, marker='o', s=150, 
                  alpha=0.7, edgecolors='black', linewidth=1.5, label=f'{group} (V)')
        ax1.scatter(a_points[:, 0], a_points[:, 1],
                  c=color, marker='^', s=150,
                  alpha=0.7, edgecolors='black', linewidth=1.5, label=f'{group} (A)')
    
    ax1.set_title('(A) All Instructions with Connecting Lines', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, ncol=2, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Density Contours (no lines) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    from scipy.stats import gaussian_kde
    
    # KDE for vanilla
    if len(all_vanilla) > 2:
        kde_vanilla = gaussian_kde(all_vanilla.T)
        x_range = np.linspace(all_vanilla[:, 0].min(), all_vanilla[:, 0].max(), 100)
        y_range = np.linspace(all_vanilla[:, 1].min(), all_vanilla[:, 1].max(), 100)
        xx, yy = np.meshgrid(x_range, y_range)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z_vanilla = kde_vanilla(positions).reshape(xx.shape)
        ax2.contour(xx, yy, z_vanilla, colors='blue', alpha=0.5, linewidths=2, levels=5)
    
    # KDE for address-aware
    if len(all_addr) > 2:
        kde_addr = gaussian_kde(all_addr.T)
        x_range = np.linspace(all_addr[:, 0].min(), all_addr[:, 0].max(), 100)
        y_range = np.linspace(all_addr[:, 1].min(), all_addr[:, 1].max(), 100)
        xx, yy = np.meshgrid(x_range, y_range)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z_addr = kde_addr(positions).reshape(xx.shape)
        ax2.contour(xx, yy, z_addr, colors='red', alpha=0.5, linewidths=2, levels=5)
    
    # Plot by group
    for group in set(all_groups):
        mask = [g == group for g in all_groups]
        v_points = all_vanilla[mask]
        a_points = all_addr[mask]
        color = GROUP_COLORS.get(group, '#7f7f7f')
        
        ax2.scatter(v_points[:, 0], v_points[:, 1], 
                  c=color, marker='o', s=100, alpha=0.7, edgecolors='black', linewidth=1)
        ax2.scatter(a_points[:, 0], a_points[:, 1],
                  c=color, marker='^', s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    ax2.set_title('(B) Density Contours by Model', fontsize=12, fontweight='bold')
    # Add simple legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Vanilla'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Address-Aware'),
    ]
    ax2.legend(handles=legend_elements, fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ========== 3. Convex Hulls by Group (no lines) ==========
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Plot convex hull for each group
    for group in set(all_groups):
        mask = [g == group for g in all_groups]
        group_vanilla = all_vanilla[mask]
        group_addr = all_addr[mask]
        color = GROUP_COLORS.get(group, '#7f7f7f')
        
        # Convex hull for vanilla (if enough points)
        if len(group_vanilla) >= 3:
            try:
                hull_v = ConvexHull(group_vanilla)
                hull_points_v = group_vanilla[hull_v.vertices]
                hull_patch_v = mpatches.Polygon(hull_points_v, alpha=0.15, facecolor=color, 
                                               edgecolor=color, linewidth=2, linestyle='--')
                ax3.add_patch(hull_patch_v)
            except:
                pass
        
        # Convex hull for address-aware (if enough points)
        if len(group_addr) >= 3:
            try:
                hull_a = ConvexHull(group_addr)
                hull_points_a = group_addr[hull_a.vertices]
                hull_patch_a = mpatches.Polygon(hull_points_a, alpha=0.15, facecolor=color,
                                               edgecolor=color, linewidth=2, linestyle='-')
                ax3.add_patch(hull_patch_a)
            except:
                pass
        
        # Plot points
        ax3.scatter(group_vanilla[:, 0], group_vanilla[:, 1], 
                  c=color, marker='o', s=100, alpha=0.8, edgecolors='black', linewidth=1)
        ax3.scatter(group_addr[:, 0], group_addr[:, 1],
                  c=color, marker='^', s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    ax3.set_title('(C) Convex Hulls by Instruction Group', fontsize=12, fontweight='bold')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Vanilla'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Address-Aware'),
    ]
    ax3.legend(handles=legend_elements, fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========== 4. Scatter by Opcode Group (no lines) ==========
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Plot each group separately with labels
    for group in set(all_groups):
        mask = [g == group for g in all_groups]
        v_points = all_vanilla[mask]
        a_points = all_addr[mask]
        color = GROUP_COLORS.get(group, '#7f7f7f')
        
        ax4.scatter(v_points[:, 0], v_points[:, 1], 
                  c=color, marker='o', s=120, alpha=0.7, 
                  edgecolors='black', linewidth=1.5, label=f'{group}')
        ax4.scatter(a_points[:, 0], a_points[:, 1],
                  c=color, marker='^', s=120, alpha=0.7,
                  edgecolors='black', linewidth=1.5)
    
    ax4.set_title('(D) All Instruction Types (No Lines)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # ========== 5. Grouped by Opcode with Color Coding (no lines) ==========
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Create a color map for each unique group
    unique_groups = list(set(all_groups))
    
    # Plot vanilla and address-aware side by side for each instruction
    for i, (v_point, a_point, group) in enumerate(zip(all_vanilla, all_addr, all_groups)):
        color = GROUP_COLORS.get(group, '#7f7f7f')
        ax5.scatter(v_point[0], v_point[1], 
                  c=color, marker='o', s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax5.scatter(a_point[0], a_point[1],
                  c=color, marker='^', s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Create legend for groups
    legend_elements = []
    for group in unique_groups:
        color = GROUP_COLORS.get(group, '#7f7f7f')
        legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor=color, markersize=10, label=group))
    
    ax5.set_title('(E) Color-Coded by Instruction Type', fontsize=12, fontweight='bold')
    ax5.legend(handles=legend_elements, fontsize=9, loc='best', ncol=2)
    ax5.grid(True, alpha=0.3)
    
    # ========== 6. Sample Instruction Labels (no lines) ==========
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Plot by group
    for group in set(all_groups):
        mask = [g == group for g in all_groups]
        v_points = all_vanilla[mask]
        a_points = all_addr[mask]
        color = GROUP_COLORS.get(group, '#7f7f7f')
        
        ax6.scatter(v_points[:, 0], v_points[:, 1], 
                  c=color, marker='o', s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax6.scatter(a_points[:, 0], a_points[:, 1],
                  c=color, marker='^', s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add labels for a subset of instructions (to avoid clutter)
    sample_indices = list(range(0, len(all_instructions), max(1, len(all_instructions) // 10)))
    for i in sample_indices:
        inst = all_instructions[i]
        group = all_groups[i]
        color = GROUP_COLORS.get(group, '#7f7f7f')
        
        # Label vanilla point
        ax6.annotate(inst[:20], (all_vanilla[i, 0], all_vanilla[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=6, alpha=0.8, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3, edgecolor='none'))
    
    ax6.set_title('(F) Sample Instruction Labels', fontsize=12, fontweight='bold')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Vanilla'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Address-Aware'),
    ]
    ax6.legend(handles=legend_elements, fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Multiple Instruction Types: Various Visualization Styles\n(Vanilla vs Address-Aware Embeddings)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    output_path = os.path.join(output_dir, 'control_flow_embeddings.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved enhanced multi-instruction visualization to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize instruction embeddings')
    parser.add_argument('--vocab_file', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--addr_model_path', type=str,
                        default=os.path.join(config.OUTPUT_DIR, 'best_model.pt'),
                        help='Path to trained address-aware model')
    parser.add_argument('--instructions_file', type=str, default=None,
                        help='Optional: file with custom instructions (one per line)')
    parser.add_argument('--output_dir', type=str,
                        default=config.OUTPUT_DIR,
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load vocabulary
    print('Loading vocabulary...')
    vocab = WordVocab.load_vocab(args.vocab_file)
    print(f'Vocab size: {len(vocab)}')
    
    # Create addr vocab
    from addr_vocab import AddrVocab
    addr_vocab = AddrVocab()
    
    # Load instructions
    if args.instructions_file:
        with open(args.instructions_file, 'r') as f:
            instructions = [line.strip() for line in f if line.strip()]
    else:
        instructions = SAMPLE_INSTRUCTIONS
    
    print(f'\nProcessing {len(instructions)} instructions...')
    
    # Load models
    print('\nLoading Vanilla PalmTree...')
    pretrained_path = os.path.join(config.PRETRAINED_MODEL_PATH, 'palmtree', 'transformer.ep19')
    vanilla_bert = load_pretrained_palmtree(
        model_path=pretrained_path,
        vocab_size=len(vocab),
        hidden=128,
        n_layers=12,
        attn_heads=12
    )
    
    # Wrap vanilla model
    class VanillaWrapper(torch.nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.palmtree = bert
            
        def forward(self, input_ids, attention_mask):
            segment_info = attention_mask.clone()
            embeddings = self.palmtree(input_ids, segment_info)
            return {'hidden_states': embeddings}
    
    vanilla_model = VanillaWrapper(vanilla_bert).to(device)
    vanilla_model.eval()
    
    print('Loading Address-Aware PalmTree...')
    addr_bert = load_pretrained_palmtree(
        model_path=pretrained_path,
        vocab_size=len(vocab),
        hidden=128,
        n_layers=12,
        attn_heads=12
    )
    addr_model = AddressAwarePalmTree(addr_bert).to(device)
    
    # Load trained weights
    checkpoint = torch.load(args.addr_model_path, map_location=device)
    addr_model.load_state_dict(checkpoint['model_state_dict'])
    addr_model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Extract embeddings
    print('\nExtracting embeddings...')
    vanilla_embeddings = []
    addr_embeddings = []
    
    for inst in instructions:
        v_emb = get_instruction_embedding(vanilla_model, inst, vocab, addr_vocab, device, 'vanilla')
        a_emb = get_instruction_embedding(addr_model, inst, vocab, addr_vocab, device, 'address_aware')
        
        vanilla_embeddings.append(v_emb)
        addr_embeddings.append(a_emb)
    
    vanilla_embeddings = np.array(vanilla_embeddings)
    addr_embeddings = np.array(addr_embeddings)
    
    print(f'Vanilla embeddings shape: {vanilla_embeddings.shape}')
    print(f'Address-aware embeddings shape: {addr_embeddings.shape}')
    
    # Visualize
    print('\nGenerating visualizations...')
    os.makedirs(args.output_dir, exist_ok=True)
    visualize_embeddings(vanilla_embeddings, addr_embeddings, instructions, args.output_dir)
    
    print('\n✅ Visualization complete!')


if __name__ == '__main__':
    main()
