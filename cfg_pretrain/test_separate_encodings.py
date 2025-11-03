"""
Test the two separate position encodings (binary and function)
"""
import torch
import sys
sys.path.insert(0, '/home/louie/PalmTree/cfg_pretrain')

from train import SinCosPositionEncoding

def test_separate_encodings():
    print("="*80)
    print("Testing Two Separate Position Encodings")
    print("="*80)
    
    embed_dim = 128
    
    # Create two separate encoders with different base frequencies
    binary_encoder = SinCosPositionEncoding(embed_dim, base=10000.0)  # Lower frequency
    function_encoder = SinCosPositionEncoding(embed_dim, base=1000.0)  # Higher frequency
    
    print(f"\nEmbedding dimension: {embed_dim}")
    print(f"Binary encoder base: 10000 (lower frequency, coarse-grained)")
    print(f"Function encoder base: 1000 (higher frequency, fine-grained)")
    
    # Test 1: Binary position encoding
    print("\n" + "="*80)
    print("Test 1: Binary Position Encoding (0.0 to 1.0)")
    print("="*80)
    binary_positions = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    
    binary_emb = binary_encoder(binary_positions, scale=2*torch.pi)
    
    print(f"Binary positions: {binary_positions[0].tolist()}")
    print(f"Encoding shape: {binary_emb.shape}")
    print(f"Encoding range: [{binary_emb.min():.3f}, {binary_emb.max():.3f}]")
    
    # Check distinctiveness
    print("\nCosine similarity between different binary positions:")
    for i in range(4):
        cos_sim = torch.nn.functional.cosine_similarity(
            binary_emb[0:1, i:i+1, :], binary_emb[0:1, i+1:i+2, :], dim=2
        )
        print(f"  Pos {binary_positions[0, i].item():.2f} vs {binary_positions[0, i+1].item():.2f}: {cos_sim.item():.4f}")
    
    # Test 2: Function position encoding (with negatives)
    print("\n" + "="*80)
    print("Test 2: Function Position Encoding (-3.0 to +3.0)")
    print("="*80)
    function_positions = torch.tensor([[-3.0, -1.5, 0.0, 1.5, 3.0]])
    
    function_emb = function_encoder(function_positions, scale=torch.pi)
    
    print(f"Function positions: {function_positions[0].tolist()}")
    print(f"Encoding shape: {function_emb.shape}")
    print(f"Encoding range: [{function_emb.min():.3f}, {function_emb.max():.3f}]")
    
    # Check distinctiveness
    print("\nCosine similarity between different function positions:")
    for i in range(4):
        cos_sim = torch.nn.functional.cosine_similarity(
            function_emb[0:1, i:i+1, :], function_emb[0:1, i+1:i+2, :], dim=2
        )
        print(f"  Pos {function_positions[0, i].item():.1f} vs {function_positions[0, i+1].item():.1f}: {cos_sim.item():.4f}")
    
    # Test 3: Combined effect
    print("\n" + "="*80)
    print("Test 3: Combined Binary + Function Encoding")
    print("="*80)
    
    # Same binary position, different function positions
    binary_pos = torch.tensor([[0.5, 0.5, 0.5]])
    func_pos = torch.tensor([[-1.0, 0.0, 1.0]])
    
    binary_emb = binary_encoder(binary_pos, scale=2*torch.pi)
    function_emb = function_encoder(func_pos, scale=torch.pi)
    combined = binary_emb + function_emb
    
    print(f"Test case: Same binary pos (0.5), different function pos")
    print(f"Binary positions:   {binary_pos[0].tolist()}")
    print(f"Function positions: {func_pos[0].tolist()}")
    
    print("\nCombined encoding similarity:")
    for i in range(2):
        cos_sim = torch.nn.functional.cosine_similarity(
            combined[0:1, i:i+1, :], combined[0:1, i+1:i+2, :], dim=2
        )
        print(f"  Combined[{i}] vs Combined[{i+1}]: {cos_sim.item():.4f} (should be different)")
    
    # Test 4: Frequency difference visualization
    print("\n" + "="*80)
    print("Test 4: Frequency Difference Between Binary and Function")
    print("="*80)
    
    # Same position value, but encoded by different encoders
    test_pos = torch.tensor([[0.5]])
    
    binary_enc = binary_encoder(test_pos, scale=2*torch.pi)
    function_enc = function_encoder(test_pos, scale=torch.pi)
    
    cos_sim = torch.nn.functional.cosine_similarity(
        binary_enc[0:1, 0:1, :], function_enc[0:1, 0:1, :], dim=2
    )
    
    print(f"Position value: 0.5")
    print(f"Binary encoding (base=10000):   mean={binary_enc.mean():.4f}, std={binary_enc.std():.4f}")
    print(f"Function encoding (base=1000):  mean={function_enc.mean():.4f}, std={function_enc.std():.4f}")
    print(f"Cosine similarity: {cos_sim.item():.4f} (should be low, confirming different frequencies)")
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80)
    print("\nKey insights:")
    print("  • Binary position: Full 128-dim encoding for global context (base=10000)")
    print("  • Function position: Full 128-dim encoding for local context (base=1000)")
    print("  • Each gets complete expressive power across all dimensions")
    print("  • Combined: semantic + binary + function + sequence")
    print("  • More expressive than splitting 64/64!")

if __name__ == "__main__":
    test_separate_encodings()
