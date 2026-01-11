"""
Jump Understanding Probe for Binary Code Models

Tests whether models truly understand control flow by analyzing:
1. Jump Target Prediction: Can the model predict where a jump goes?
2. Attention Patterns: Does attention focus on jump targets?

Usage:
    python jump_probe.py --baseline_model ./output/baseline_pretrain/best_model \
                         --addressaware_model ./output/addressaware_pretrain/best_model \
                         --data_file ./probe/data/probe_functions.json \
                         --output_dir ./probe/results/jump_probe
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import pickle
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extern', 'jTrans'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extern', 'jTrans', 'pretrain', 'address_aware'))

from transformers import BertTokenizer, BertModel, BertConfig
from finetune import BinBertModel, AddressAwareBertWrapper
from address_embedding import AddressAwareBERTEmbedding


class JumpProbe:
    """Probe for testing jump understanding in binary code models."""
    
    def __init__(self, model, tokenizer, model_type='baseline'):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.model.eval()
        
        # Jump opcodes to detect
        self.jump_opcodes = {
            'jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle',
            'ja', 'jae', 'jb', 'jbe', 'jo', 'jno', 'js', 'jns', 'jp', 'jnp',
            'jcxz', 'jecxz', 'call', 'ret'
        }
    
    def load_probe_data(self, data_file):
        """
        Load probe data from JSON file.
        
        Expected format:
        {
            "func_id": {
                "tokens": "token1 token2 ...",
                "cfg": {"5": 10, "15": 20},  # jump_pos -> target_pos
                "binary_pos": [...],  # For address-aware
                "function_pos": [...],
                "bb_pos": [...],
                "var_offsets": [...]
            }
        }
        """
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} functions from {data_file}")
        return data
    
    def prepare_input(self, func_data):
        """Prepare model input from function data."""
        
        tokens_str = func_data['tokens']
        tokens = tokens_str.split()
        
        # Tokenize
        if self.model_type == 'baseline':
            # Add special tokens
            tokens_with_special = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in tokens_with_special]
            
            # Truncate or pad
            max_len = 512
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            
            attention_mask = [1] * len(input_ids)
            
            # Pad
            padding_len = max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            attention_mask += [0] * padding_len
            
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask]),
                'token_type_ids': torch.zeros(1, max_len, dtype=torch.long),
                'tokens': tokens_with_special[:max_len]
            }
        
        else:  # address-aware
            tokens_with_special = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in tokens_with_special]
            
            # Get position data from func_data (already extracted by prepare script)
            binary_pos = func_data['binary_pos']
            function_pos = func_data['function_pos']
            bb_pos = func_data['bb_pos']
            var_offsets = func_data['var_offsets']
            
            # Add CLS/SEP positions
            binary_pos = [-1.0] + binary_pos + [-1.0]
            function_pos = [-1.0] + function_pos + [-1.0]
            bb_pos = [-1.0] + bb_pos + [-1.0]
            var_offsets = [-1] + var_offsets + [-1]
            
            max_len = 512
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                binary_pos = binary_pos[:max_len]
                function_pos = function_pos[:max_len]
                bb_pos = bb_pos[:max_len]
                var_offsets = var_offsets[:max_len]
            
            attention_mask = [1] * len(input_ids)
            
            # Pad
            padding_len = max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            attention_mask += [0] * padding_len
            binary_pos += [-1.0] * padding_len
            function_pos += [-1.0] * padding_len
            bb_pos += [-1.0] * padding_len
            var_offsets += [-1] * padding_len
            
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask]),
                'token_type_ids': torch.zeros(1, max_len, dtype=torch.long),
                'binary_pos': torch.tensor([binary_pos], dtype=torch.float32),
                'function_pos': torch.tensor([function_pos], dtype=torch.float32),
                'bb_pos': torch.tensor([bb_pos], dtype=torch.float32),
                'var_offsets': torch.tensor([var_offsets], dtype=torch.long),
                'tokens': tokens_with_special[:max_len]
            }
    
    def get_attention_weights(self, inputs):
        """Extract attention weights from model."""
        
        device = next(self.model.parameters()).device
        
        if self.model_type == 'baseline':
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True,
                    output_hidden_states=True
                )
            
            attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq]
            hidden_states = outputs.hidden_states  # Tuple of [batch, seq, hidden]
            
        else:  # address-aware
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            binary_pos = inputs['binary_pos'].to(device)
            function_pos = inputs['function_pos'].to(device)
            bb_pos = inputs['bb_pos'].to(device)
            var_offsets = inputs['var_offsets'].to(device)
            
            # Get BERT from wrapper
            if hasattr(self.model, 'bert'):
                bert = self.model.bert
            else:
                bert = self.model
            
            with torch.no_grad():
                # Get embeddings
                embeddings = bert.embeddings(
                    input_ids.squeeze(0),
                    token_type_ids.squeeze(0),
                    binary_pos.squeeze(0),
                    function_pos.squeeze(0),
                    bb_pos.squeeze(0),
                    var_offsets.squeeze(0)
                )
                embeddings = embeddings.unsqueeze(0)  # Add batch dim
                
                # Forward through encoder with attention output
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                
                # Collect attention and hidden states
                attentions = []
                hidden_states = [embeddings]
                
                hidden = embeddings
                for layer in bert.encoder.layer:
                    layer_outputs = layer(hidden, extended_attention_mask, output_attentions=True)
                    hidden = layer_outputs[0]
                    attentions.append(layer_outputs[1])
                    hidden_states.append(hidden)
        
        # Average across heads for the last layer
        last_layer_attn = attentions[-1]  # [batch, heads, seq, seq]
        avg_attention = last_layer_attn.mean(dim=1).squeeze(0)  # [seq, seq]
        
        # Last hidden state
        last_hidden = hidden_states[-1].squeeze(0)  # [seq, hidden]
        
        return avg_attention.cpu().numpy(), last_hidden.cpu()
    
    def find_jump_positions(self, tokens, cfg):
        """Find jump instructions and their targets."""
        
        jumps = []
        
        # Parse CFG (jump_pos -> target_pos)
        cfg_dict = {int(k): int(v) for k, v in cfg.items()}
        
        for jump_pos, target_pos in cfg_dict.items():
            if jump_pos >= len(tokens) or target_pos >= len(tokens):
                continue
            
            # Verify it's actually a jump opcode
            token = tokens[jump_pos].lower()
            is_jump = any(op in token for op in self.jump_opcodes)
            
            if is_jump:
                jumps.append({
                    'jump_pos': jump_pos,
                    'target_pos': target_pos,
                    'opcode': tokens[jump_pos]
                })
        
        return jumps
    
    def evaluate_jump_prediction(self, data, top_k=5):
        """
        Evaluate jump target prediction accuracy.
        
        For each jump, check if the true target is in top-k attended positions.
        """
        
        results = {
            'total_jumps': 0,
            'correct_top1': 0,
            'correct_top3': 0,
            'correct_top5': 0,
            'all_ranks': [],
            'attention_scores': []
        }
        
        for func_id, func_data in tqdm(data.items(), desc=f"Evaluating {self.model_type}"):
            inputs = self.prepare_input(func_data)
            attention, hidden = self.get_attention_weights(inputs)
            
            tokens = inputs['tokens']
            cfg = func_data.get('cfg', {})
            
            jumps = self.find_jump_positions(tokens, cfg)
            
            for jump in jumps:
                jump_pos = jump['jump_pos']
                target_pos = jump['target_pos']
                
                # Get attention from jump position
                jump_attention = attention[jump_pos]
                
                # Find top-k positions
                top_positions = np.argsort(-jump_attention)[:top_k]
                
                # Check if target is in top-k
                if target_pos in top_positions[:1]:
                    results['correct_top1'] += 1
                if target_pos in top_positions[:3]:
                    results['correct_top3'] += 1
                if target_pos in top_positions[:5]:
                    results['correct_top5'] += 1
                
                # Find rank of true target
                rank = np.where(np.argsort(-jump_attention) == target_pos)[0]
                if len(rank) > 0:
                    results['all_ranks'].append(rank[0] + 1)
                
                # Store attention score to target
                results['attention_scores'].append(jump_attention[target_pos])
                
                results['total_jumps'] += 1
        
        # Calculate accuracies
        if results['total_jumps'] > 0:
            results['accuracy_top1'] = results['correct_top1'] / results['total_jumps'] * 100
            results['accuracy_top3'] = results['correct_top3'] / results['total_jumps'] * 100
            results['accuracy_top5'] = results['correct_top5'] / results['total_jumps'] * 100
            results['mean_rank'] = np.mean(results['all_ranks'])
            results['median_rank'] = np.median(results['all_ranks'])
        
        return results
    
    def visualize_attention_single_function(self, func_data, output_path):
        """Visualize attention pattern for a single function with jumps."""
        
        inputs = self.prepare_input(func_data)
        attention, hidden = self.get_attention_weights(inputs)
        
        tokens = inputs['tokens']
        cfg = func_data.get('cfg', {})
        jumps = self.find_jump_positions(tokens, cfg)
        
        if len(jumps) == 0:
            print("No jumps found in this function")
            return
        
        # Create subplots for each jump
        n_jumps = min(len(jumps), 4)  # Max 4 jumps to visualize
        fig, axes = plt.subplots(n_jumps, 1, figsize=(14, 4 * n_jumps))
        
        if n_jumps == 1:
            axes = [axes]
        
        for idx, jump in enumerate(jumps[:n_jumps]):
            ax = axes[idx]
            
            jump_pos = jump['jump_pos']
            target_pos = jump['target_pos']
            opcode = jump['opcode']
            
            # Get attention from jump position
            jump_attention = attention[jump_pos, :len(tokens)]
            
            # Plot attention distribution
            x = np.arange(len(jump_attention))
            bars = ax.bar(x, jump_attention, color='lightblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
            
            # Highlight jump position
            bars[jump_pos].set_color('orange')
            bars[jump_pos].set_alpha(1.0)
            
            # Highlight true target
            bars[target_pos].set_color('red')
            bars[target_pos].set_alpha(1.0)
            
            # Mark top attended position
            max_attn_pos = np.argmax(jump_attention)
            if max_attn_pos != target_pos and max_attn_pos != jump_pos:
                bars[max_attn_pos].set_color('green')
                bars[max_attn_pos].set_alpha(0.8)
            
            # Add vertical lines
            ax.axvline(x=jump_pos, color='orange', linestyle='--', linewidth=2, 
                      label=f'Jump Position ({jump_pos})', alpha=0.7)
            ax.axvline(x=target_pos, color='red', linestyle='--', linewidth=2,
                      label=f'True Target ({target_pos})', alpha=0.7)
            
            if max_attn_pos != target_pos and max_attn_pos != jump_pos:
                ax.axvline(x=max_attn_pos, color='green', linestyle='--', linewidth=2,
                          label=f'Max Attention ({max_attn_pos})', alpha=0.7)
            
            # Labels and title
            ax.set_xlabel('Token Position', fontsize=11)
            ax.set_ylabel('Attention Weight', fontsize=11)
            
            # Check if prediction is correct
            is_correct = "✓ CORRECT" if max_attn_pos == target_pos else "✗ WRONG"
            ax.set_title(f'{self.model_type.upper()}: Jump "{opcode}" at pos {jump_pos} → target {target_pos} {is_correct}',
                        fontsize=12, fontweight='bold')
            
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            # Set x-axis limits
            ax.set_xlim(-1, min(len(tokens), 100))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved attention visualization to {output_path}")


def visualize_comparison(baseline_results, addressaware_results, output_dir):
    """Create comparison visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    models = ['Baseline', 'Address-Aware']
    
    # Top-1 accuracy
    top1_accs = [baseline_results['accuracy_top1'], addressaware_results['accuracy_top1']]
    bars1 = axes[0].bar(models, top1_accs, color=['#3498db', '#e74c3c'], width=0.6)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Top-1 Jump Target Prediction', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='~Random (5%)')
    axes[0].legend()
    for bar, acc in zip(bars1, top1_accs):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Top-3 and Top-5
    top3_accs = [baseline_results['accuracy_top3'], addressaware_results['accuracy_top3']]
    top5_accs = [baseline_results['accuracy_top5'], addressaware_results['accuracy_top5']]
    
    x = np.arange(len(models))
    width = 0.25
    axes[1].bar(x - width, top1_accs, width, label='Top-1', color='#3498db')
    axes[1].bar(x, top3_accs, width, label='Top-3', color='#2ecc71')
    axes[1].bar(x + width, top5_accs, width, label='Top-5', color='#f39c12')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Top-K Jump Target Prediction', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].legend()
    axes[1].set_ylim([0, 100])
    
    # Mean rank comparison
    mean_ranks = [baseline_results['mean_rank'], addressaware_results['mean_rank']]
    bars3 = axes[2].bar(models, mean_ranks, color=['#3498db', '#e74c3c'], width=0.6)
    axes[2].set_ylabel('Mean Rank', fontsize=12)
    axes[2].set_title('Average Rank of True Target', fontsize=13, fontweight='bold')
    axes[2].invert_yaxis()  # Lower is better
    for bar, rank in zip(bars3, mean_ranks):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{rank:.1f}', ha='center', va='top', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jump_prediction_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Saved comparison to {output_dir}/jump_prediction_comparison.png")
    
    # 2. Rank distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(baseline_results['all_ranks'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(baseline_results['mean_rank'], color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {baseline_results["mean_rank"]:.1f}')
    ax1.set_xlabel('Rank of True Target', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Baseline: Distribution of True Target Ranks', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xlim([0, 100])
    
    ax2.hist(addressaware_results['all_ranks'], bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax2.axvline(addressaware_results['mean_rank'], color='red', linestyle='--', linewidth=2,
               label=f'Mean: {addressaware_results["mean_rank"]:.1f}')
    ax2.set_xlabel('Rank of True Target', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Address-Aware: Distribution of True Target Ranks', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_xlim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_distribution.png'), dpi=300)
    plt.close()
    
    print(f"Saved rank distribution to {output_dir}/rank_distribution.png")


def main():
    parser = argparse.ArgumentParser(description="Jump Understanding Probe")
    parser.add_argument("--baseline_model", type=str, required=True,
                       help="Path to baseline pretrained model")
    parser.add_argument("--addressaware_model", type=str, required=True,
                       help="Path to address-aware pretrained model")
    parser.add_argument("--baseline_tokenizer", type=str, required=True,
                       help="Path to baseline tokenizer")
    parser.add_argument("--addressaware_tokenizer", type=str, required=True,
                       help="Path to address-aware tokenizer")
    parser.add_argument("--baseline_data_file", type=str, required=True,
                       help="Path to baseline probe data JSON file")
    parser.add_argument("--addressaware_data_file", type=str, required=True,
                       help="Path to address-aware probe data JSON file")
    parser.add_argument("--output_dir", type=str, default="./probe/results/jump_probe",
                       help="Output directory for results")
    parser.add_argument("--visualize_samples", type=int, default=5,
                       help="Number of sample functions to visualize")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizers
    print("Loading tokenizers...")
    baseline_tokenizer = BertTokenizer.from_pretrained(args.baseline_tokenizer)
    addressaware_tokenizer = BertTokenizer.from_pretrained(args.addressaware_tokenizer)
    
    # Load baseline model
    print("Loading baseline model...")
    baseline_model = BinBertModel.from_pretrained(args.baseline_model)
    baseline_model.eval()
    if torch.cuda.is_available():
        baseline_model = baseline_model.cuda()
    
    print(f"Baseline vocab size: {len(baseline_tokenizer.vocab)}")
    
    # Load address-aware model
    print("Loading address-aware model...")
    config_path = os.path.join(args.addressaware_model, 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = BertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict['hidden_size'] * 4,
        max_position_embeddings=config_dict['max_position_embeddings'],
        type_vocab_size=config_dict.get('type_vocab_size', 2),
    )
    
    bert_model = BertModel(config, add_pooling_layer=False)
    bert_model.embeddings = AddressAwareBERTEmbedding(
        vocab_size=config_dict['vocab_size'],
        embed_size=config_dict['hidden_size'],
        dropout=0.1,
        max_len=config_dict['max_position_embeddings'],
        use_address_embedding=True,
        use_var_embedding=True,
        segment_types=256,
        vocab_stoi=None
    )
    
    weights_path = os.path.join(args.addressaware_model, 'pytorch_model.bin')
    state_dict = torch.load(weights_path, map_location='cpu')
    bert_model.load_state_dict(state_dict)
    
    addressaware_model = AddressAwareBertWrapper(bert_model)
    addressaware_model.eval()
    if torch.cuda.is_available():
        addressaware_model = addressaware_model.cuda()
    
    print(f"Address-aware vocab size: {len(addressaware_tokenizer.vocab)}")
    
    # Create probes with their respective tokenizers
    baseline_probe = JumpProbe(baseline_model, baseline_tokenizer, 'baseline')
    addressaware_probe = JumpProbe(addressaware_model, addressaware_tokenizer, 'addressaware')
    
    # Load data (each model uses its own formatted data)
    print("Loading probe data...")
    print(f"  Baseline data: {args.baseline_data_file}")
    print(f"  Address-aware data: {args.addressaware_data_file}")
    
    baseline_data = baseline_probe.load_probe_data(args.baseline_data_file)
    addressaware_data = addressaware_probe.load_probe_data(args.addressaware_data_file)
    
    # Evaluate jump prediction
    print("\n" + "="*80)
    print("EVALUATING JUMP TARGET PREDICTION")
    print("="*80)
    
    print("\nEvaluating baseline model...")
    baseline_results = baseline_probe.evaluate_jump_prediction(baseline_data)
    
    print("\nEvaluating address-aware model...")
    addressaware_results = addressaware_probe.evaluate_jump_prediction(addressaware_data)
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\nBaseline Model:")
    print(f"  Total jumps: {baseline_results['total_jumps']}")
    print(f"  Top-1 Accuracy: {baseline_results['accuracy_top1']:.2f}%")
    print(f"  Top-3 Accuracy: {baseline_results['accuracy_top3']:.2f}%")
    print(f"  Top-5 Accuracy: {baseline_results['accuracy_top5']:.2f}%")
    print(f"  Mean Rank: {baseline_results['mean_rank']:.2f}")
    print(f"  Median Rank: {baseline_results['median_rank']:.2f}")
    
    print(f"\nAddress-Aware Model:")
    print(f"  Total jumps: {addressaware_results['total_jumps']}")
    print(f"  Top-1 Accuracy: {addressaware_results['accuracy_top1']:.2f}%")
    print(f"  Top-3 Accuracy: {addressaware_results['accuracy_top3']:.2f}%")
    print(f"  Top-5 Accuracy: {addressaware_results['accuracy_top5']:.2f}%")
    print(f"  Mean Rank: {addressaware_results['mean_rank']:.2f}")
    print(f"  Median Rank: {addressaware_results['median_rank']:.2f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'baseline': {k: v for k, v in baseline_results.items() 
                        if k not in ['all_ranks', 'attention_scores']},
            'addressaware': {k: v for k, v in addressaware_results.items()
                           if k not in ['all_ranks', 'attention_scores']}
        }, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    # Visualize comparison
    print("\nGenerating comparis (use same function IDs from both datasets)
    print(f"\nVisualizing {args.visualize_samples} sample functions...")
    
    # Get common function IDs
    baseline_func_ids = list(baseline_data.keys())
    addressaware_func_ids = set(addressaware_data.keys())
    common_func_ids = [fid for fid in baseline_func_ids if fid in addressaware_func_ids][:args.visualize_samples]
    
    print(f"  Found {len(common_func_ids)} common functions to visualize")
    
    for idx, func_id in enumerate(common_func_ids):
        print(f"  Visualizing function {idx+1}/{len(common_func_ids)} (ID: {func_id})")
        
        baseline_path = os.path.join(args.output_dir, f'baseline_func_{idx+1}.png')
        baseline_probe.visualize_attention_single_function(baseline_data[func_id], baseline_path)
        
        addressaware_path = os.path.join(args.output_dir, f'addressaware_func_{idx+1}.png')
        addressaware_probe.visualize_attention_single_function(addressaware_data[func_id]seline_path)
        
        addressaware_path = os.path.join(args.output_dir, f'addressaware_func_{idx+1}.png')
        addressaware_probe.visualize_attention_single_function(func_data, addressaware_path)
    
    print(f"\n{'='*80}")
    print(f"DONE! Results saved to {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
