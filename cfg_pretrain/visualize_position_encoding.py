"""
Visualize sin/cos position encoding for address positions
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

class SinCosPositionEncoding:
    """Sin/Cos encoding for continuous position values"""
    def __init__(self, embed_dim, position_type='binary'):
        self.embed_dim = embed_dim
        self.position_type = position_type
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        self.div_term = div_term
        
    def forward(self, positions):
        """positions: Position values (binary: 0-1, function: -N to +N)"""
        if self.position_type == 'binary':
            scaled_pos = positions.unsqueeze(-1) * 2 * torch.pi  # [0, 2π]
        else:  # function positions
            scaled_pos = positions.unsqueeze(-1) * torch.pi  # [-Nπ, +Nπ]
        
        pe = torch.zeros(positions.size(0), positions.size(1), self.embed_dim)
        pe[:, :, 0::2] = torch.sin(scaled_pos * self.div_term)  # Even indices
        pe[:, :, 1::2] = torch.cos(scaled_pos * self.div_term)  # Odd indices
        
        return pe


def visualize_encoding():
    embed_dim = 128
    encoder = SinCosPositionEncoding(embed_dim)
    
    # Create sample positions from 0.0 to 1.0
    positions = torch.linspace(0.0, 1.0, 100).unsqueeze(0)  # [1, 100]
    
    # Get encodings
    encodings = encoder.forward(positions)  # [1, 100, 128]
    encodings = encodings.squeeze(0).numpy()  # [100, 128]
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Heatmap of all dimensions
    ax = axes[0, 0]
    im = ax.imshow(encodings.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xlabel('Position Index (0 to 1.0 normalized)')
    ax.set_ylabel('Embedding Dimension')
    ax.set_title('Sin/Cos Encoding Heatmap (All 128 Dimensions)')
    plt.colorbar(im, ax=ax)
    
    # Plot 2: First few dimensions as line plots
    ax = axes[0, 1]
    x = np.linspace(0, 1, 100)
    for i in range(0, 8):
        ax.plot(x, encodings[:, i], label=f'Dim {i}', alpha=0.7)
    ax.set_xlabel('Normalized Position (0.0 to 1.0)')
    ax.set_ylabel('Encoding Value')
    ax.set_title('First 8 Dimensions (showing different frequencies)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Compare low vs high frequency components
    ax = axes[1, 0]
    ax.plot(x, encodings[:, 0], label='Dim 0 (Low freq)', linewidth=2)
    ax.plot(x, encodings[:, 32], label='Dim 32 (Mid freq)', linewidth=2)
    ax.plot(x, encodings[:, 64], label='Dim 64 (High freq)', linewidth=2)
    ax.plot(x, encodings[:, 96], label='Dim 96 (Very high freq)', linewidth=2)
    ax.set_xlabel('Normalized Position (0.0 to 1.0)')
    ax.set_ylabel('Encoding Value')
    ax.set_title('Different Frequency Bands')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Show how positions are distinguished
    ax = axes[1, 1]
    # Sample a few specific positions
    test_positions = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).unsqueeze(0)
    test_encodings = encoder.forward(test_positions).squeeze(0).numpy()
    
    for i, pos in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        ax.plot(test_encodings[i, :32], marker='o', label=f'Pos {pos:.2f}', alpha=0.7)
    ax.set_xlabel('Embedding Dimension (first 32)')
    ax.set_ylabel('Encoding Value')
    ax.set_title('Embeddings for Different Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('position_encoding_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: position_encoding_visualization.png")
    plt.close()
    
    # Print some statistics
    print("\nSin/Cos Position Encoding Analysis:")
    print("="*60)
    print(f"Embedding dimension: {embed_dim}")
    print(f"Position range: 0.0 to 1.0 (normalized)")
    print(f"Encoding range: {encodings.min():.3f} to {encodings.max():.3f}")
    print(f"\nFrequency bands:")
    print(f"  - Low freq (dims 0-31):   Capture coarse patterns")
    print(f"  - Mid freq (dims 32-63):  Capture medium patterns")
    print(f"  - High freq (dims 64-95): Capture fine patterns")
    print(f"  - Very high (dims 96-127): Capture very fine patterns")
    print("\nKey advantage:")
    print("  • Multiple frequency bands allow model to attend to both")
    print("    nearby positions (high freq) and distant positions (low freq)")
    print("  • Each position gets a unique, continuous encoding")
    print("  • Generalizes well to unseen positions")


if __name__ == "__main__":
    visualize_encoding()
