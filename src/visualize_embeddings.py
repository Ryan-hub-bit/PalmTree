"""
Visualize instruction embeddings grouped by opcode to test semantic understanding.
This script loads a trained BERT model, extracts embeddings for instructions from test data,
and creates visualizations (t-SNE, PCA) to show if similar opcodes cluster together.
"""
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from palmtree.dataset import WordVocab
import bert_pytorch
import json
import argparse
import random

def load_instructions_from_file(file_path, max_instructions=1000):
    """Load instructions from test data file."""
    instructions = []
    print(f"Loading instructions from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_instructions:
                break
            # Each line contains tab-separated instructions
            line = line.strip()
            if line:
                # Split by tabs to get individual instructions
                inst_list = line.split('\t')
                instructions.extend(inst_list)
    
    print(f"Loaded {len(instructions)} instructions")
    return instructions

def split_by_opcode(instructions):
    """Split instructions by their opcode (first token)."""
    opcode_groups = defaultdict(list)
    
    for inst in instructions:
        tokens = inst.strip().split()
        if tokens:
            opcode = tokens[0]
            opcode_groups[opcode].append(inst)
    
    return opcode_groups

def get_instruction_embedding(model, vocab, instruction, device):
    """Get embedding for a single instruction from the BERT model."""
    # Tokenize instruction
    tokens = instruction.strip().split()
    token_ids = [vocab.stoi.get(token, vocab.unk_index) for token in tokens]
    
    # Add padding if needed (up to some reasonable length)
    max_len = 20
    if len(token_ids) < max_len:
        token_ids = token_ids + [vocab.pad_index] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    
    # Convert to tensor
    input_tensor = torch.tensor([token_ids]).to(device)
    segment_tensor = torch.zeros_like(input_tensor).to(device)
    
    # Get embedding from BERT
    with torch.no_grad():
        # Pass through BERT
        output = model(input_tensor, segment_tensor)
        # Use [CLS] token embedding (first token)
        embedding = output[0, 0, :].cpu().numpy()
    
    return embedding

def extract_embeddings_by_opcode(model, vocab, opcode_groups, device, samples_per_opcode=50):
    """Extract embeddings for instructions grouped by opcode."""
    embeddings_by_opcode = {}
    
    print("\nExtracting embeddings...")
    for opcode, instructions in opcode_groups.items():
        if len(instructions) < 5:  # Skip opcodes with too few examples
            continue
        
        # Sample instructions if too many
        sampled = random.sample(instructions, min(samples_per_opcode, len(instructions)))
        
        embeddings = []
        for inst in sampled:
            emb = get_instruction_embedding(model, vocab, inst, device)
            embeddings.append(emb)
        
        embeddings_by_opcode[opcode] = np.array(embeddings)
        print(f"  {opcode}: {len(embeddings)} embeddings extracted")
    
    return embeddings_by_opcode

def visualize_embeddings_tsne(embeddings_by_opcode, output_path, perplexity=30):
    """Visualize embeddings using t-SNE."""
    print("\nGenerating t-SNE visualization...")
    
    # Prepare data
    all_embeddings = []
    all_labels = []
    opcode_to_idx = {}
    
    for idx, (opcode, embeddings) in enumerate(embeddings_by_opcode.items()):
        all_embeddings.append(embeddings)
        all_labels.extend([opcode] * len(embeddings))
        opcode_to_idx[opcode] = idx
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Apply t-SNE
    print(f"Running t-SNE on {len(all_embeddings)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=(16, 12))
    unique_opcodes = list(embeddings_by_opcode.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_opcodes)))
    
    for opcode, color in zip(unique_opcodes, colors):
        mask = np.array(all_labels) == opcode
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   label=opcode, alpha=0.6, s=50, c=[color])
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Instruction Embeddings Grouped by Opcode (t-SNE)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved to {output_path}")
    plt.close()

def visualize_embeddings_pca(embeddings_by_opcode, output_path):
    """Visualize embeddings using PCA."""
    print("\nGenerating PCA visualization...")
    
    # Prepare data
    all_embeddings = []
    all_labels = []
    
    for opcode, embeddings in embeddings_by_opcode.items():
        all_embeddings.append(embeddings)
        all_labels.extend([opcode] * len(embeddings))
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Apply PCA
    print(f"Running PCA on {len(all_embeddings)} embeddings...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=(16, 12))
    unique_opcodes = list(embeddings_by_opcode.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_opcodes)))
    
    for opcode, color in zip(unique_opcodes, colors):
        mask = np.array(all_labels) == opcode
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   label=opcode, alpha=0.6, s=50, c=[color])
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('Instruction Embeddings Grouped by Opcode (PCA)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PCA plot saved to {output_path}")
    plt.close()

def compute_semantic_groups_analysis(embeddings_by_opcode):
    """
    Compute intra-group vs inter-group distances to quantify semantic clustering.
    Good embeddings should have small intra-group distance and large inter-group distance.
    """
    print("\nComputing semantic grouping analysis...")
    
    # Define semantic groups (opcodes with similar semantics)
    semantic_groups = {
        'arithmetic': ['add', 'sub', 'inc', 'dec', 'neg', 'adc', 'sbb'],
        'data_movement': ['mov', 'movsx', 'movzx', 'lea', 'push', 'pop'],
        'logical': ['and', 'or', 'xor', 'not', 'test'],
        'comparison': ['cmp', 'test'],
        'control_flow': ['jmp', 'je', 'jne', 'jz', 'jnz', 'call', 'ret'],
        'shift': ['shl', 'shr', 'sal', 'sar', 'rol', 'ror'],
    }
    
    results = {}
    
    for group_name, opcodes in semantic_groups.items():
        # Filter opcodes that exist in our data
        valid_opcodes = [op for op in opcodes if op in embeddings_by_opcode]
        if len(valid_opcodes) < 2:
            continue
        
        # Compute intra-group distance (average distance within same semantic group)
        intra_distances = []
        for i, op1 in enumerate(valid_opcodes):
            for op2 in valid_opcodes[i+1:]:
                emb1 = embeddings_by_opcode[op1]
                emb2 = embeddings_by_opcode[op2]
                # Average distance between all pairs
                for e1 in emb1:
                    for e2 in emb2:
                        dist = np.linalg.norm(e1 - e2)
                        intra_distances.append(dist)
        
        if intra_distances:
            results[group_name] = {
                'opcodes': valid_opcodes,
                'intra_distance': np.mean(intra_distances),
                'intra_std': np.std(intra_distances)
            }
    
    # Compute inter-group distance (average distance between different semantic groups)
    inter_distances = []
    group_names = list(results.keys())
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i+1:]:
            for op1 in results[g1]['opcodes']:
                for op2 in results[g2]['opcodes']:
                    emb1 = embeddings_by_opcode[op1]
                    emb2 = embeddings_by_opcode[op2]
                    for e1 in emb1[:10]:  # Sample to avoid too many comparisons
                        for e2 in emb2[:10]:
                            dist = np.linalg.norm(e1 - e2)
                            inter_distances.append(dist)
    
    # Print results
    print("\n" + "="*60)
    print("SEMANTIC GROUPING ANALYSIS")
    print("="*60)
    
    for group_name, data in results.items():
        print(f"\n{group_name.upper()}:")
        print(f"  Opcodes: {', '.join(data['opcodes'])}")
        print(f"  Intra-group distance: {data['intra_distance']:.4f} ± {data['intra_std']:.4f}")
    
    if inter_distances:
        print(f"\nINTER-GROUP DISTANCE: {np.mean(inter_distances):.4f} ± {np.std(inter_distances):.4f}")
        
        # Compute ratio: good embeddings should have ratio > 1
        avg_intra = np.mean([r['intra_distance'] for r in results.values()])
        ratio = np.mean(inter_distances) / avg_intra
        print(f"\nINTER/INTRA RATIO: {ratio:.4f}")
        print(f"  (> 1.0 = good semantic clustering, < 1.0 = poor clustering)")
    
    # Save to JSON
    analysis_results = {
        'semantic_groups': {k: {
            'opcodes': v['opcodes'],
            'intra_distance_mean': float(v['intra_distance']),
            'intra_distance_std': float(v['intra_std'])
        } for k, v in results.items()},
        'inter_group_distance_mean': float(np.mean(inter_distances)) if inter_distances else None,
        'inter_group_distance_std': float(np.std(inter_distances)) if inter_distances else None,
        'inter_intra_ratio': float(ratio) if inter_distances else None
    }
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description='Visualize instruction embeddings by opcode')
    parser.add_argument('--model_path', required=True, help='Path to trained BERT model checkpoint')
    parser.add_argument('--vocab_path', required=True, help='Path to vocabulary file')
    parser.add_argument('--test_data', default='/home/louie/PalmTree/datalong/test/cfg_8.txt',
                       help='Path to test data file')
    parser.add_argument('--max_instructions', type=int, default=2000,
                       help='Maximum instructions to load')
    parser.add_argument('--samples_per_opcode', type=int, default=100,
                       help='Number of samples per opcode')
    parser.add_argument('--output_dir', default='/home/louie/PalmTree/evaluation_results',
                       help='Output directory for plots')
    parser.add_argument('--top_opcodes', type=int, default=15,
                       help='Number of most common opcodes to visualize')
    args = parser.parse_args()
    
    # Load model and vocabulary
    print(f"Loading model from {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.eval()
    
    print(f"Loading vocabulary from {args.vocab_path}")
    vocab = WordVocab.load_vocab(args.vocab_path)
    
    # Load instructions
    instructions = load_instructions_from_file(args.test_data, args.max_instructions)
    
    # Split by opcode
    opcode_groups = split_by_opcode(instructions)
    print(f"\nFound {len(opcode_groups)} unique opcodes")
    
    # Get top N most common opcodes
    sorted_opcodes = sorted(opcode_groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_opcodes = dict(sorted_opcodes[:args.top_opcodes])
    
    print(f"\nTop {args.top_opcodes} opcodes:")
    for opcode, insts in top_opcodes.items():
        print(f"  {opcode}: {len(insts)} instructions")
    
    # Extract embeddings
    embeddings_by_opcode = extract_embeddings_by_opcode(
        model, vocab, top_opcodes, device, args.samples_per_opcode
    )
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    tsne_path = os.path.join(args.output_dir, 'embeddings_tsne.png')
    pca_path = os.path.join(args.output_dir, 'embeddings_pca.png')
    
    visualize_embeddings_tsne(embeddings_by_opcode, tsne_path)
    visualize_embeddings_pca(embeddings_by_opcode, pca_path)
    
    # Semantic analysis
    analysis = compute_semantic_groups_analysis(embeddings_by_opcode)
    
    # Save analysis results
    analysis_path = os.path.join(args.output_dir, 'semantic_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis results saved to {analysis_path}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"Check the following files:")
    print(f"  - t-SNE plot: {tsne_path}")
    print(f"  - PCA plot: {pca_path}")
    print(f"  - Analysis: {analysis_path}")

if __name__ == '__main__':
    main()
