#!/usr/bin/env python3
"""
Function Bucket Probe

Evaluates how well BERT embeddings encode position within a function
by predicting which bucket (0-9) an instruction belongs to based on func_norm.

Bucket assignment: bucket = int(func_norm * 10)
where func_norm is the normalized position within the function [0.0, 1.0)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../pre-trained_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../addressaware'))
from vocab import WordVocab


class FunctionInstructionDataset(Dataset):
    """
    Dataset for instruction-level function bucket prediction.
    Each instruction is labeled with its function bucket (0-9 based on func_norm).
    """
    
    def __init__(self, instructions, labels, vocab, seq_len=20):
        """
        Args:
            instructions: List of instruction strings with address metadata
            labels: List of function bucket labels (0-9)
            vocab: WordVocab instance
            seq_len: Maximum sequence length
        """
        self.instructions = instructions
        self.labels = labels
        self.vocab = vocab
        self.seq_len = seq_len
    
    def _parse_instruction_with_positions(self, instruction_str):
        """
        Parse instruction to extract tokens and position information.
        
        Format: opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 address(0xADDR:bnorm:fnorm:bbnorm) ...
        
        Returns:
            cleaned_instruction: String with just tokens (no address metadata)
            token_positions: List of (binary_pos, function_pos, bb_pos) for each token
        """
        tokens = instruction_str.split()
        cleaned_tokens = []
        token_positions = []
        
        for i, token in enumerate(tokens):
            # Check if token has address metadata
            if '(' in token and ')' in token:
                # Extract the actual token (before parenthesis)
                actual_token = token.split('(')[0]
                
                # Extract position metadata
                metadata = token.split('(')[1].rstrip(')')
                parts = metadata.split(':')
                
                if len(parts) == 4:
                    # Format: 0xADDR:bnorm:fnorm:bbnorm
                    _, bnorm, fnorm, bbnorm = parts
                    binary_pos = float(bnorm)
                    function_pos = float(fnorm)
                    bb_pos = float(bbnorm)
                else:
                    binary_pos = function_pos = bb_pos = 0.0
                
                cleaned_tokens.append(actual_token)
                token_positions.append((binary_pos, function_pos, bb_pos))
            else:
                # Regular token without address (e.g., registers)
                cleaned_tokens.append(token)
                token_positions.append((0.0, 0.0, 0.0))
        
        cleaned_instruction = ' '.join(cleaned_tokens)
        return cleaned_instruction, token_positions
    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        """
        Get a single instruction with its label.
        Following UsableTransformer.encode() pattern but with address positions.
        """
        instruction_str = self.instructions[idx]
        label = self.labels[idx]
        
        # Parse instruction to get cleaned text and positions
        cleaned_inst, token_positions = self._parse_instruction_with_positions(instruction_str)
        
        # Convert to sequence using vocab (like UsableTransformer)
        s = self.vocab.to_seq(cleaned_inst)
        
        # Add special tokens: [CLS]=3 at start, [SEP]=2 at end
        s = [3] + s + [2]
        
        # Validate token IDs are in valid range
        vocab_size = len(self.vocab)
        for i, tid in enumerate(s):
            if tid < 0 or tid >= vocab_size:
                print(f"WARNING: Invalid token ID {tid} at position {i} in instruction: {cleaned_inst}")
                # Clamp to valid range
                s[i] = min(max(tid, 0), vocab_size - 1)
        
        # Add positions for special tokens ([CLS] and [SEP] get 0.0, 0.0, 0.0)
        token_positions = [(0.0, 0.0, 0.0)] + token_positions + [(0.0, 0.0, 0.0)]
        
        # Create segment labels (all 1s for the actual sequence length)
        l = len(s) * [1]
        
        # Truncate or pad to seq_len
        if len(s) > self.seq_len:
            sequence = s[:self.seq_len]
            segment_label = l[:self.seq_len]
            token_positions = token_positions[:self.seq_len]
        else:
            sequence = s + [0] * (self.seq_len - len(s))
            segment_label = l + [0] * (self.seq_len - len(l))
            token_positions = token_positions + [(0.0, 0.0, 0.0)] * (self.seq_len - len(token_positions))
        
        # Extract position arrays
        binary_pos = [pos[0] for pos in token_positions]
        function_pos = [pos[1] for pos in token_positions]
        bb_pos = [pos[2] for pos in token_positions]
        
        return {
            'sequence': torch.LongTensor(sequence),
            'segment_label': torch.LongTensor(segment_label),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
            'label': torch.tensor(label, dtype=torch.long)
        }


def extract_embeddings(model, data_loader, device, model_type='addressaware'):
    """
    Extract embeddings from a BERT model.
    
    Args:
        model: BERT model
        data_loader: DataLoader for instructions
        device: Device to run on
        model_type: 'addressaware' or 'baseline'
    
    Returns:
        embeddings: numpy array [num_samples, hidden_dim]
        labels: numpy array [num_samples]
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Extracting {model_type} embeddings"):
            sequence = batch['sequence'].to(device)
            segment_label = batch['segment_label'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Forward pass
            if model_type == 'addressaware':
                binary_pos = batch['binary_pos'].to(device)
                function_pos = batch['function_pos'].to(device)
                bb_pos = batch['bb_pos'].to(device)
                
                encoded = model.forward(sequence, segment_label, binary_pos, function_pos, bb_pos)
            else:  # baseline
                encoded = model.forward(sequence, segment_label)
            
            # encoded shape: [batch_size, seq_len, hidden_dim]
            # Use [CLS] token embedding (first token)
            pooled = encoded[:, 0, :]  # [batch_size, hidden_dim]
            
            all_embeddings.append(pooled.cpu().numpy())
            all_labels.append(labels)
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels


def train_probe(embeddings, labels, test_size=0.2, random_state=42, probe_type='linear'):
    """
    Train a probe on embeddings.
    
    Args:
        embeddings: numpy array of embeddings [num_samples, hidden_dim]
        labels: numpy array of labels [num_samples]
        test_size: Fraction of data for testing
        random_state: Random seed
        probe_type: 'linear' for logistic regression, 'mlp' for neural network
    
    Returns:
        probe: Trained classifier
        results: Dictionary with metrics
    """
    # Split data
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"Training probe ({probe_type}): {len(X_train)} train, {len(X_test)} test samples")
    
    # Train probe
    if probe_type == 'linear':
        probe = LogisticRegression(max_iter=1000, random_state=random_state, multi_class='multinomial')
    elif probe_type == 'mlp':
        from sklearn.neural_network import MLPClassifier
        # 2-layer MLP with hidden layer size = 256
        probe = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=1000,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")
    
    probe.fit(X_train, y_train)
    
    # Evaluate
    train_pred = probe.predict(X_train)
    test_pred = probe.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Detailed metrics
    test_report = classification_report(y_test, test_pred, output_dict=True)
    test_cm = confusion_matrix(y_test, test_pred)
    
    results = {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'probe_type': probe_type,
        'classification_report': test_report,
        'confusion_matrix': test_cm.tolist(),
        'num_train': len(X_train),
        'num_test': len(X_test)
    }
    
    return probe, results


def main():
    parser = argparse.ArgumentParser(description="Function Bucket Probe")
    
    # Data
    parser.add_argument('--data_dir', required=True, help='Directory containing instruction files and labels')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file (.pkl)')
    parser.add_argument('--binary_filter', default=None, help='Only use data from this specific binary (e.g., "a52dec__liba52.so.0.0.0")')
    
    # Models
    parser.add_argument('--addressaware_model', required=True, help='Path to address-aware BERT model')
    parser.add_argument('--baseline_model', required=True, help='Path to baseline BERT model')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=20, help='Maximum sequence length (must match model training)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split fraction')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--probe_type', default='linear', choices=['linear', 'mlp'], 
                        help='Probe type: linear (logistic regression) or mlp (neural network)')
    
    # Output
    parser.add_argument('--output', default='probe_results_function', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print(f"\nLoading vocabulary from: {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load data from directory - combine all instruction files
    print(f"\nLoading data from directory: {args.data_dir}")
    all_instructions = []
    all_labels = []
    
    # Find all *_instructions.txt and *_instructions_labels.json files
    import glob
    instruction_files = glob.glob(os.path.join(args.data_dir, '*_instructions.txt'))
    
    # Filter by binary if specified
    if args.binary_filter:
        instruction_files = [f for f in instruction_files if args.binary_filter in os.path.basename(f)]
        if not instruction_files:
            print(f"Error: No instruction files found for binary: {args.binary_filter}")
            return
        print(f"Filtering to binary: {args.binary_filter}")
    
    for inst_file in instruction_files:
        # Get corresponding label file
        base_name = inst_file.replace('_instructions.txt', '')
        label_file = base_name + '_instructions_labels.json'
        
        if not os.path.exists(label_file):
            print(f"Warning: Label file not found for {inst_file}, skipping")
            continue
        
        # Load instructions
        with open(inst_file, 'r') as f:
            instructions = [line.strip() for line in f if line.strip()]
        
        # Load labels
        with open(label_file, 'r') as f:
            labels_data = json.load(f)
        
        # Extract function bucket labels
        labels = [item['function_bucket'] for item in labels_data['instructions']]
        
        # Sanity check
        if len(instructions) != len(labels):
            print(f"Warning: Mismatch in {inst_file}: {len(instructions)} instructions vs {len(labels)} labels")
            continue
        
        all_instructions.extend(instructions)
        all_labels.extend(labels)
        print(f"  Loaded {len(instructions)} instructions from {os.path.basename(inst_file)}")
    
    print(f"\nTotal: {len(all_instructions)} instructions")
    print(f"Label distribution (before balancing):")
    unique, counts = np.unique(all_labels, return_counts=True)
    for bucket, count in zip(unique, counts):
        print(f"  Bucket {bucket}: {count} ({count/len(all_labels)*100:.1f}%)")
    
    # Balance buckets - make each bucket have the same number of samples
    print(f"\nBalancing buckets...")
    min_count = min(counts)
    print(f"Min bucket size: {min_count}, using this for all buckets")
    
    balanced_instructions = []
    balanced_labels = []
    
    # Group by label
    label_to_indices = {label: [] for label in range(10)}
    for idx, label in enumerate(all_labels):
        label_to_indices[label].append(idx)
    
    # Sample min_count from each bucket
    np.random.seed(42)  # For reproducibility
    for label in range(10):
        indices = label_to_indices[label]
        if len(indices) >= min_count:
            sampled_indices = np.random.choice(indices, min_count, replace=False)
        else:
            sampled_indices = indices
        
        for idx in sampled_indices:
            balanced_instructions.append(all_instructions[idx])
            balanced_labels.append(all_labels[idx])
    
    # Update to use balanced data
    all_instructions = balanced_instructions
    all_labels = balanced_labels
    
    print(f"\nBalanced dataset: {len(all_instructions)} instructions")
    print(f"Label distribution (after balancing):")
    unique, counts = np.unique(all_labels, return_counts=True)
    for bucket, count in zip(unique, counts):
        print(f"  Bucket {bucket}: {count} ({count/len(all_labels)*100:.1f}%)")
    
    # Create dataset
    dataset = FunctionInstructionDataset(all_instructions, all_labels, vocab, args.seq_len)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # ========== Address-Aware Model ==========
    print("\n" + "="*70)
    print("Processing Address-Aware Model")
    print("="*70)
    
    print(f"Loading model from: {args.addressaware_model}")
    addressaware_bert = torch.load(args.addressaware_model, map_location=device, weights_only=False)
    addressaware_bert = addressaware_bert.to(device)
    addressaware_bert.eval()
    
    # Extract embeddings
    aa_embeddings, aa_labels = extract_embeddings(
        addressaware_bert, data_loader, device, model_type='addressaware'
    )
    
    print(f"Extracted embeddings: {aa_embeddings.shape}")
    
    # Train probe
    aa_probe, aa_results = train_probe(aa_embeddings, aa_labels, args.test_size, probe_type=args.probe_type)
    
    # Save results
    aa_results_file = os.path.join(args.output, 'addressaware_results.json')
    with open(aa_results_file, 'w') as f:
        json.dump(aa_results, f, indent=2)
    print(f"Results saved to: {aa_results_file}")
    
    # ========== Baseline Model ==========
    print("\n" + "="*70)
    print("Processing Baseline Model")
    print("="*70)
    
    print(f"Loading model from: {args.baseline_model}")
    baseline_bert = torch.load(args.baseline_model, map_location=device, weights_only=False)
    baseline_bert = baseline_bert.to(device)
    baseline_bert.eval()
    
    # Extract embeddings
    bl_embeddings, bl_labels = extract_embeddings(
        baseline_bert, data_loader, device, model_type='baseline'
    )
    
    print(f"Extracted embeddings: {bl_embeddings.shape}")
    
    # Train probe
    bl_probe, bl_results = train_probe(bl_embeddings, bl_labels, args.test_size, probe_type=args.probe_type)
    
    # Save results
    bl_results_file = os.path.join(args.output, 'baseline_results.json')
    with open(bl_results_file, 'w') as f:
        json.dump(bl_results, f, indent=2)
    print(f"Results saved to: {bl_results_file}")
    
    # ========== Comparison ==========
    print("\n" + "="*70)
    print("Comparison: Address-Aware vs Baseline")
    print("="*70)
    
    print(f"\nAddress-Aware Model:")
    print(f"  Train Accuracy: {aa_results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {aa_results['test_accuracy']:.4f}")
    
    print(f"\nBaseline Model:")
    print(f"  Train Accuracy: {bl_results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {bl_results['test_accuracy']:.4f}")
    
    print(f"\nImprovement (Address-Aware - Baseline):")
    print(f"  Train Accuracy: {aa_results['train_accuracy'] - bl_results['train_accuracy']:+.4f}")
    print(f"  Test Accuracy:  {aa_results['test_accuracy'] - bl_results['test_accuracy']:+.4f}")
    
    # Save comparison
    comparison = {
        'addressaware': aa_results,
        'baseline': bl_results,
        'improvement': {
            'train_accuracy': aa_results['train_accuracy'] - bl_results['train_accuracy'],
            'test_accuracy': aa_results['test_accuracy'] - bl_results['test_accuracy']
        }
    }
    
    comparison_file = os.path.join(args.output, 'comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to: {comparison_file}")
    
    print("\n" + "="*70)
    print("✓ Function Bucket Probe Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
