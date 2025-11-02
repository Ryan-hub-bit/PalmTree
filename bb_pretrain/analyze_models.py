"""
Comprehensive Analysis: Address-Aware PalmTree vs Vanilla PalmTree
Analyzes architecture, embeddings, performance, and insights
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from collections import defaultdict
import json

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pre-trained_model'))

import config
from models.addr_palmtree import AddressAwarePalmTree, load_pretrained_palmtree
from vocab import WordVocab


def analyze_model_architecture(model, name="Model"):
    """Analyze model architecture and parameters"""
    print(f"\n{'='*70}")
    print(f"MODEL ARCHITECTURE: {name}")
    print(f"{'='*70}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    # Analyze layer structure
    print(f"\n🏗️  Layer Structure:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<30} {num_params:>12,} params")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
    }


def analyze_embedding_dimensions(model, model_type="vanilla"):
    """Analyze embedding dimensions and structure"""
    print(f"\n{'='*70}")
    print(f"EMBEDDING ANALYSIS: {model_type.upper()}")
    print(f"{'='*70}")
    
    if model_type == "vanilla":
        print(f"\n📐 Embedding Structure:")
        print(f"  Level 1 (Semantic): Token embeddings from pre-trained BERT")
        print(f"  Hidden dimension: {model.palmtree.hidden}")
        print(f"  Number of layers: {model.palmtree.n_layers}")
        print(f"  Attention heads: {model.palmtree.attn_heads}")
        
        embedding_info = {
            'levels': 1,
            'semantic_dim': model.palmtree.hidden,
            'address_dim': 0,
            'positional_dim': 0,
            'total_dim': model.palmtree.hidden,
        }
        
    else:  # address-aware
        print(f"\n📐 Embedding Structure (3-Level Architecture):")
        print(f"  Level 1 (Semantic): Token embeddings from pre-trained BERT")
        print(f"    - Hidden dimension: {model.hidden_size}")
        print(f"  Level 2 (Address): Type + Value embeddings")
        print(f"    - Address type embedding: {model.addr_type_embedding.embedding_dim}")
        print(f"    - Address value projection: {config.ADDRESS_ENCODING_DIM} → {model.hidden_size}")
        print(f"  Level 3 (Positional): Position embeddings")
        print(f"    - Position embedding: {model.position_embedding.embedding_dim}")
        print(f"\n  Fusion Network:")
        print(f"    - Input: 3 × {model.hidden_size} = {3 * model.hidden_size}")
        print(f"    - Output: {model.hidden_size}")
        
        embedding_info = {
            'levels': 3,
            'semantic_dim': model.hidden_size,
            'address_type_dim': model.addr_type_embedding.embedding_dim,
            'address_value_dim': config.ADDRESS_ENCODING_DIM,
            'positional_dim': model.position_embedding.embedding_dim,
            'fusion_input_dim': 3 * model.hidden_size,
            'total_dim': model.hidden_size,
        }
    
    return embedding_info


def analyze_task_capabilities(model, model_type="vanilla"):
    """Analyze what tasks the model can perform"""
    print(f"\n{'='*70}")
    print(f"TASK CAPABILITIES: {model_type.upper()}")
    print(f"{'='*70}")
    
    capabilities = []
    
    if model_type == "vanilla":
        print(f"\n✅ Supported Tasks:")
        print(f"  1. Instruction embedding generation")
        print(f"  2. Semantic similarity computation")
        print(f"  3. Token prediction (with additional head)")
        
        capabilities = [
            'instruction_embedding',
            'semantic_similarity',
            'token_prediction',
        ]
        
    else:  # address-aware
        print(f"\n✅ Supported Tasks:")
        print(f"  1. Instruction embedding generation (3-level)")
        print(f"  2. Semantic similarity computation")
        print(f"  3. Next basic block prediction")
        print(f"  4. Address type classification (code/data/target)")
        print(f"  5. Edge type classification (call/branch/return)")
        print(f"  6. Address-aware code analysis")
        
        capabilities = [
            'instruction_embedding_3level',
            'semantic_similarity',
            'next_bb_prediction',
            'address_type_classification',
            'edge_type_classification',
            'address_aware_analysis',
        ]
        
        print(f"\n📊 Task Heads:")
        print(f"  - Next BB head: {model.next_bb_head.out_features} classes")
        print(f"  - Address type head: {model.addr_type_head.out_features} classes")
        print(f"  - Edge type head: {model.edge_type_head.out_features} classes")
    
    return capabilities


def analyze_computational_complexity(model, model_type="vanilla", seq_len=128, batch_size=8):
    """Analyze computational complexity"""
    print(f"\n{'='*70}")
    print(f"COMPUTATIONAL COMPLEXITY: {model_type.upper()}")
    print(f"{'='*70}")
    
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randint(0, 6000, (batch_size, seq_len)).to(device)
    dummy_mask = torch.ones((batch_size, seq_len)).to(device)
    
    if model_type == "address_aware":
        dummy_addr = torch.randn((batch_size, seq_len, config.ADDRESS_ENCODING_DIM)).to(device)
        dummy_addr_type = torch.randint(0, 6, (batch_size, seq_len)).to(device)
    
    # Measure inference time
    import time
    
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            if model_type == "vanilla":
                _ = model(dummy_input, dummy_mask)
            else:
                _ = model(dummy_input, dummy_mask, dummy_addr, dummy_addr_type)
        
        # Actual timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        for _ in range(100):
            if model_type == "vanilla":
                _ = model(dummy_input, dummy_mask)
            else:
                _ = model(dummy_input, dummy_mask, dummy_addr, dummy_addr_type)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
    
    avg_time = (end - start) / 100
    throughput = batch_size / avg_time
    
    print(f"\n⚡ Performance Metrics:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Average inference time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"  GPU memory allocated: {memory_allocated:.1f} MB")
        print(f"  GPU memory reserved: {memory_reserved:.1f} MB")
    
    return {
        'inference_time_ms': avg_time * 1000,
        'throughput': throughput,
    }


def compare_models_side_by_side(vanilla_info, addr_info):
    """Create side-by-side comparison table"""
    print(f"\n{'='*70}")
    print(f"SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Aspect':<30} {'Vanilla PalmTree':<20} {'Address-Aware':<20}")
    print(f"{'-'*70}")
    
    # Architecture
    print(f"\n🏗️  ARCHITECTURE")
    print(f"{'Embedding Levels':<30} {vanilla_info['embedding']['levels']:<20} {addr_info['embedding']['levels']:<20}")
    print(f"{'Hidden Dimension':<30} {vanilla_info['embedding']['total_dim']:<20} {addr_info['embedding']['total_dim']:<20}")
    print(f"{'Total Parameters':<30} {vanilla_info['architecture']['total_params']:>19,} {addr_info['architecture']['total_params']:>19,}")
    
    # Capabilities
    print(f"\n✨ CAPABILITIES")
    print(f"{'Task Types':<30} {len(vanilla_info['capabilities']):<20} {len(addr_info['capabilities']):<20}")
    print(f"{'Next BB Prediction':<30} {'❌':<20} {'✅':<20}")
    print(f"{'Address Type Classification':<30} {'❌':<20} {'✅':<20}")
    print(f"{'Edge Type Classification':<30} {'❌':<20} {'✅':<20}")
    
    # Performance
    print(f"\n⚡ PERFORMANCE")
    print(f"{'Inference Time (ms)':<30} {vanilla_info['performance']['inference_time_ms']:<20.2f} {addr_info['performance']['inference_time_ms']:<20.2f}")
    print(f"{'Throughput (samples/s)':<30} {vanilla_info['performance']['throughput']:<20.1f} {addr_info['performance']['throughput']:<20.1f}")
    
    overhead_time = ((addr_info['performance']['inference_time_ms'] - vanilla_info['performance']['inference_time_ms']) 
                     / vanilla_info['performance']['inference_time_ms'] * 100)
    overhead_params = ((addr_info['architecture']['total_params'] - vanilla_info['architecture']['total_params']) 
                       / vanilla_info['architecture']['total_params'] * 100)
    
    print(f"\n📈 OVERHEAD")
    print(f"{'Parameter Increase':<30} {overhead_params:>19.1f}%")
    print(f"{'Inference Time Increase':<30} {overhead_time:>19.1f}%")


def generate_insights(vanilla_info, addr_info):
    """Generate insights and recommendations"""
    print(f"\n{'='*70}")
    print(f"💡 KEY INSIGHTS")
    print(f"{'='*70}")
    
    param_increase = ((addr_info['architecture']['total_params'] - vanilla_info['architecture']['total_params']) 
                     / vanilla_info['architecture']['total_params'] * 100)
    
    print(f"\n1. Architecture Design:")
    print(f"   - Address-aware model adds {param_increase:.1f}% more parameters")
    print(f"   - Uses 3-level embedding (semantic + address + positional)")
    print(f"   - Preserves pre-trained knowledge from vanilla PalmTree")
    
    print(f"\n2. Capabilities:")
    print(f"   - Vanilla: General-purpose instruction embeddings")
    print(f"   - Address-aware: Specialized for control flow and address-sensitive tasks")
    
    print(f"\n3. Use Cases:")
    print(f"   Vanilla PalmTree:")
    print(f"   - General instruction similarity")
    print(f"   - Code search and retrieval")
    print(f"   - Function matching")
    
    print(f"\n   Address-Aware PalmTree:")
    print(f"   - Control flow graph recovery")
    print(f"   - Next basic block prediction")
    print(f"   - Address-sensitive code analysis")
    print(f"   - Jump target prediction")
    print(f"   - Indirect call resolution")
    
    print(f"\n4. Performance Trade-offs:")
    time_increase = ((addr_info['performance']['inference_time_ms'] - vanilla_info['performance']['inference_time_ms']) 
                    / vanilla_info['performance']['inference_time_ms'] * 100)
    print(f"   - Inference time increase: {time_increase:.1f}%")
    print(f"   - Benefit: Multi-task learning with address awareness")
    print(f"   - Recommendation: Use address-aware for CFG tasks, vanilla for general embeddings")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive model analysis')
    parser.add_argument('--vocab_file', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--addr_model_path', type=str,
                        default=os.path.join(config.OUTPUT_DIR, 'best_model.pt'),
                        help='Path to trained address-aware model')
    parser.add_argument('--output_file', type=str,
                        default=os.path.join(config.OUTPUT_DIR, 'model_analysis.json'),
                        help='Output file for analysis results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load vocabulary
    print('\nLoading vocabulary...')
    vocab = WordVocab.load_vocab(args.vocab_file)
    vocab_size = len(vocab)
    
    # Load Vanilla PalmTree
    print('\nLoading Vanilla PalmTree...')
    pretrained_path = os.path.join(config.PRETRAINED_MODEL_PATH, 'palmtree', 'transformer.ep19')
    vanilla_bert = load_pretrained_palmtree(
        model_path=pretrained_path,
        vocab_size=vocab_size,
        hidden=128,
        n_layers=12,
        attn_heads=12
    )
    
    # Wrap vanilla model
    class VanillaWrapper(nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.palmtree = bert
            
        def forward(self, input_ids, attention_mask):
            segment_info = attention_mask.clone()
            embeddings = self.palmtree(input_ids, segment_info)
            return {'hidden_states': embeddings}
    
    vanilla_model = VanillaWrapper(vanilla_bert).to(device)
    
    # Load Address-Aware PalmTree
    print('Loading Address-Aware PalmTree...')
    addr_bert = load_pretrained_palmtree(
        model_path=pretrained_path,
        vocab_size=vocab_size,
        hidden=128,
        n_layers=12,
        attn_heads=12
    )
    addr_model = AddressAwarePalmTree(addr_bert).to(device)
    
    checkpoint = torch.load(args.addr_model_path, map_location=device)
    addr_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f}")
    
    # Analyze both models
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Vanilla analysis
    vanilla_arch = analyze_model_architecture(vanilla_model, "Vanilla PalmTree")
    vanilla_emb = analyze_embedding_dimensions(vanilla_model, "vanilla")
    vanilla_cap = analyze_task_capabilities(vanilla_model, "vanilla")
    vanilla_perf = analyze_computational_complexity(vanilla_model, "vanilla")
    
    vanilla_info = {
        'architecture': vanilla_arch,
        'embedding': vanilla_emb,
        'capabilities': vanilla_cap,
        'performance': vanilla_perf,
    }
    
    # Address-aware analysis
    addr_arch = analyze_model_architecture(addr_model, "Address-Aware PalmTree")
    addr_emb = analyze_embedding_dimensions(addr_model, "address-aware")
    addr_cap = analyze_task_capabilities(addr_model, "address-aware")
    addr_perf = analyze_computational_complexity(addr_model, "address-aware")
    
    addr_info = {
        'architecture': addr_arch,
        'embedding': addr_emb,
        'capabilities': addr_cap,
        'performance': addr_perf,
    }
    
    # Comparison and insights
    compare_models_side_by_side(vanilla_info, addr_info)
    generate_insights(vanilla_info, addr_info)
    
    # Save results
    results = {
        'vanilla_palmtree': vanilla_info,
        'address_aware_palmtree': addr_info,
        'checkpoint_info': {
            'epoch': int(checkpoint['epoch']),
            'val_loss': float(checkpoint['val_loss']),
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"✅ Analysis complete!")
    print(f"Results saved to: {args.output_file}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
