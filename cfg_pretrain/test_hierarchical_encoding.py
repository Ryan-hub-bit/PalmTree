"""
Test the hierarchical position encoding
"""
import torch
import sys
sys.path.insert(0, '/home/louie/PalmTree/cfg_pretrain')

from train import HierarchicalPositionEncoding

def test_hierarchical_encoding():
    print("="*80)
    print("Testing Hierarchical Position Encoding")
    print("="*80)
    
    embed_dim = 128
    encoder = HierarchicalPositionEncoding(embed_dim)
    
    print(f"\nEmbedding dimension: {embed_dim}")
    print(f"  - Global (binary) dims: 0-{embed_dim//2-1} (lower frequencies)")
    print(f"  - Local (function) dims: {embed_dim//2}-{embed_dim-1} (higher frequencies)")
    
    # Test case 1: Same binary position, different function positions
    print("\n" + "="*80)
    print("Test 1: Same binary position (0.5), different function positions")
    print("="*80)
    binary_pos = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # All same global position
    func_pos = torch.tensor([[-2.0, -1.0, 0.0, 1.0]])  # Different local positions
    
    encoding = encoder(binary_pos, func_pos)
    print(f"Input:")
    print(f"  Binary positions:   {binary_pos[0].tolist()}")
    print(f"  Function positions: {func_pos[0].tolist()}")
    print(f"\nOutput encoding shape: {encoding.shape}")
    
    # Check if global part is similar (should be, since binary pos is same)
    global_part = encoding[0, :, :64]  # First 64 dims
    print(f"\nGlobal part (first 64 dims) similarity:")
    for i in range(3):
        cos_sim = torch.nn.functional.cosine_similarity(
            global_part[i:i+1], global_part[i+1:i+2], dim=1
        )
        print(f"  Position {i} vs {i+1}: {cos_sim.item():.4f} (should be ~1.0)")
    
    # Check if local part is different (should be, since function pos differs)
    local_part = encoding[0, :, 64:]  # Last 64 dims
    print(f"\nLocal part (last 64 dims) similarity:")
    for i in range(3):
        cos_sim = torch.nn.functional.cosine_similarity(
            local_part[i:i+1], local_part[i+1:i+2], dim=1
        )
        print(f"  Position {i} vs {i+1}: {cos_sim.item():.4f} (should be varied)")
    
    # Test case 2: Different binary positions, same function position
    print("\n" + "="*80)
    print("Test 2: Different binary positions, same function position (0.0)")
    print("="*80)
    binary_pos = torch.tensor([[0.2, 0.4, 0.6, 0.8]])  # Different global positions
    func_pos = torch.tensor([[0.0, 0.0, 0.0, 0.0]])   # All same local position
    
    encoding = encoder(binary_pos, func_pos)
    print(f"Input:")
    print(f"  Binary positions:   {binary_pos[0].tolist()}")
    print(f"  Function positions: {func_pos[0].tolist()}")
    
    # Check if global part is different
    global_part = encoding[0, :, :64]
    print(f"\nGlobal part (first 64 dims) similarity:")
    for i in range(3):
        cos_sim = torch.nn.functional.cosine_similarity(
            global_part[i:i+1], global_part[i+1:i+2], dim=1
        )
        print(f"  Position {i} vs {i+1}: {cos_sim.item():.4f} (should be varied)")
    
    # Check if local part is similar
    local_part = encoding[0, :, 64:]
    print(f"\nLocal part (last 64 dims) similarity:")
    for i in range(3):
        cos_sim = torch.nn.functional.cosine_similarity(
            local_part[i:i+1], local_part[i+1:i+2], dim=1
        )
        print(f"  Position {i} vs {i+1}: {cos_sim.item():.4f} (should be ~1.0)")
    
    # Test case 3: Handle negative function positions
    print("\n" + "="*80)
    print("Test 3: Negative function positions work correctly")
    print("="*80)
    binary_pos = torch.tensor([[0.5, 0.5]])
    func_pos = torch.tensor([[-3.0, 3.0]])  # Symmetric around 0
    
    encoding = encoder(binary_pos, func_pos)
    print(f"Input:")
    print(f"  Binary positions:   {binary_pos[0].tolist()}")
    print(f"  Function positions: {func_pos[0].tolist()}")
    
    # Check encoding is valid
    print(f"\nEncoding statistics:")
    print(f"  Min value: {encoding.min().item():.4f}")
    print(f"  Max value: {encoding.max().item():.4f}")
    print(f"  Mean: {encoding.mean().item():.4f}")
    print(f"  Std: {encoding.std().item():.4f}")
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80)
    print("\nKey insights:")
    print("  • Global (binary) position encoded in first 64 dims with lower frequencies")
    print("  • Local (function) position encoded in last 64 dims with higher frequencies")
    print("  • Single unified vector captures both hierarchical levels")
    print("  • Model can attend to different frequency bands for different granularity")

if __name__ == "__main__":
    test_hierarchical_encoding()
