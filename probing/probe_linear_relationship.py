"""
Linear Relationship Probing for Position Embeddings

This script tests if there is a linear relationship between position values
and learned embeddings at three levels:
1. Binary level: instruction-by-instruction across entire binary
2. Function level: within longest function
3. BB level: within longest basic block

Format: One instruction per line with inline addresses
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import os
import sys
import re
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from addressaware.model import AddressAwareBERT
from addressaware.model_baseline import BaselineBERTForPretraining
from palmtree.dataset.vocab import WordVocab
from palmtree.model.bert import BERT
from probing.position_predictor import PositionProbe


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class SingleInstructionDataset:
    """
    Dataset for single instructions with inline addresses.
    Format: opcode(0xADDR:bnorm:fnorm:bbnorm) operands...
    
    For address-aware: Preserves position information
    For baseline: Strips position information (like baseline dataloader)
    """
    
    def __init__(self, file_path, vocab, is_addressaware=True):
        self.vocab = vocab
        self.is_addressaware = is_addressaware
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        
        # Load instructions
        self.instructions = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens, positions = self._parse_instruction(line)
                    if tokens:
                        self.instructions.append((tokens, positions))
        
        mode = "address-aware" if is_addressaware else "baseline (stripped addresses)"
        print(f"Loaded {len(self.instructions)} instructions from {file_path} ({mode})")
    
    def _parse_instruction(self, instruction_str):
        """
        Parse single instruction.
        
        Address-aware: Extract tokens AND positions
        Baseline: Extract tokens ONLY (strip position info like baseline dataloader)
        """
        tokens = []
        positions = []
        
        parts = instruction_str.split()
        for part in parts:
            # Check for opcode with inline address: opcode(0xADDR:bnorm:fnorm:bbnorm)
            addr_match = self.addr_pattern.match(part)
            if addr_match:
                opcode = addr_match.group(1)
                tokens.append(opcode)
                
                if self.is_addressaware:
                    # Keep position information
                    bnorm = float(addr_match.group(3))
                    fnorm = float(addr_match.group(4))
                    bbnorm = float(addr_match.group(5))
                    positions.append((bnorm, fnorm, bbnorm))
                else:
                    # Baseline: strip position info
                    positions.append((0.0, 0.0, 0.0))
                continue
            
            # Check for nested address: address(0xADDR:bnorm:fnorm:bbnorm)
            nested_match = self.nested_addr_pattern.match(part)
            if nested_match:
                # Keep 'address' token, discard address values (both models)
                tokens.append('address')
                positions.append((0.0, 0.0, 0.0))
                continue
            
            # Regular token (no address)
            tokens.append(part)
            positions.append((0.0, 0.0, 0.0))
        
        return tokens, positions
    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        return self.instructions[idx]


def extract_embeddings_for_instructions(model, dataset, device, vocab, is_addressaware=True, max_samples=None):
    """
    Extract embeddings for individual instructions.
    
    Returns:
        embeddings: (num_instructions, hidden_size)
        positions: dict with 'binary', 'function', 'bb' arrays
    """
    model.eval()
    
    all_embeddings = []
    all_binary_pos = []
    all_function_pos = []
    all_bb_pos = []
    
    max_samples = max_samples or len(dataset)
    
    with torch.no_grad():
        for i in tqdm(range(min(max_samples, len(dataset))), desc="Extracting embeddings"):
            tokens, positions = dataset[i]
            
            # Convert tokens to indices
            token_ids = [vocab.sos_index] + \
                       [vocab.stoi.get(token, vocab.unk_index) for token in tokens] + \
                       [vocab.eos_index]
            
            # Prepare positions (add padding for <sos> and <eos>)
            pos_data = [(0.0, 0.0, 0.0)] + positions + [(0.0, 0.0, 0.0)]
            
            # Convert to tensors
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
            segment = torch.zeros_like(input_ids).to(device)
            
            if is_addressaware:
                # Extract position components
                binary_pos = torch.tensor([[p[0] for p in pos_data]], dtype=torch.float).to(device)
                function_pos = torch.tensor([[p[1] for p in pos_data]], dtype=torch.float).to(device)
                bb_pos = torch.tensor([[p[2] for p in pos_data]], dtype=torch.float).to(device)
                
                # Get encoder output using model's forward method (just like how2use.py)
                # AddressAwareBERT.forward() returns [batch_size, seq_len, hidden]
                encoder_output = model(input_ids, segment, binary_pos, function_pos, bb_pos)
                
                # Use mean pooling over sequence (just like how2use.py: torch.mean(encoded, dim=1))
                embedding = encoder_output[0].mean(dim=0).cpu().numpy()
                
                # Store ground truth position (use first real token's position, after <sos>)
                all_binary_pos.append(pos_data[1][0])  # pos_data[0] is <sos>, pos_data[1] is first real token
                all_function_pos.append(pos_data[1][1])
                all_bb_pos.append(pos_data[1][2])
            else:
                # Baseline - use BERT encoder (self.bert), not the full model with task heads
                # BaselineBERTForPretraining wraps BERT, we call self.bert.forward()
                encoder_output = model.bert(input_ids, segment)
                
                # Mean pooling over sequence (just like how2use.py)
                embedding = encoder_output[0].mean(dim=0).cpu().numpy()
                
                # No ground truth for baseline
                all_binary_pos.append(0.0)
                all_function_pos.append(0.0)
                all_bb_pos.append(0.0)
            
            all_embeddings.append(embedding)
    
    embeddings = np.array(all_embeddings)
    positions = {
        'binary': np.array(all_binary_pos),
        'function': np.array(all_function_pos),
        'bb': np.array(all_bb_pos)
    }
    
    return embeddings, positions


def test_nonlinear_relationship(embeddings, positions, level_name):
    """
    Test for NON-LINEAR relationship between position and embeddings using MLP.
    
    Args:
        embeddings: (N, hidden_size) array of embeddings
        positions: (N,) array of position values
        level_name: name of the test level
    
    Uses:
    1. Non-linear probe (MLP with 2 hidden layers)
    2. Compare with linear baseline
    """
    print(f"\n{'='*80}")
    print(f"Testing NON-LINEAR Relationship: {level_name.upper()} Level")
    print(f"{'='*80}")
    print(f"Position range: {positions.min():.6f} to {positions.max():.6f}")
    print(f"Position mean: {positions.mean():.6f}, std: {positions.std():.6f}")
    
    # Convert to torch tensors
    X = torch.tensor(embeddings, dtype=torch.float)
    y = torch.tensor(positions, dtype=torch.float).unsqueeze(1)
    
    # Normalize positions to [0, 1] for better training
    y_min = y.min()
    y_max = y.max()
    y_normalized = (y - y_min) / (y_max - y_min + 1e-8)
    
    # Split into train/test
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_normalized[train_idx], y_normalized[test_idx]
    
    # Build NON-LINEAR probe (MLP with 2 hidden layers + ReLU)
    hidden_size = X.shape[1]
    nonlinear_probe = nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    # Build LINEAR probe for comparison
    linear_probe = nn.Sequential(
        nn.Linear(hidden_size, 1),
        nn.Sigmoid()
    )
    
    # Train both probes
    for probe_name, probe in [("Non-Linear MLP", nonlinear_probe), ("Linear", linear_probe)]:
        optimizer = optim.Adam(probe.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training
        probe.train()
        for epoch in range(100):  # More epochs for non-linear
            optimizer.zero_grad()
            pred = probe(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        probe.eval()
        with torch.no_grad():
            pred_train = probe(X_train).numpy().flatten()
            pred_test = probe(X_test).numpy().flatten()
        
        y_train_np = y_train.numpy().flatten()
        y_test_np = y_test.numpy().flatten()
        
        # Calculate metrics
        train_mae = np.mean(np.abs(pred_train - y_train_np))
        test_mae = np.mean(np.abs(pred_test - y_test_np))
        train_r2 = 1 - np.sum((y_train_np - pred_train)**2) / np.sum((y_train_np - y_train_np.mean())**2)
        test_r2 = 1 - np.sum((y_test_np - pred_test)**2) / np.sum((y_test_np - y_test_np.mean())**2)
        
        # Correlation analysis
        pearson_r_train, pearson_p_train = pearsonr(y_train_np, pred_train)
        pearson_r_test, pearson_p_test = pearsonr(y_test_np, pred_test)
        spearman_rho_train, spearman_p_train = spearmanr(y_train_np, pred_train)
        spearman_rho_test, spearman_p_test = spearmanr(y_test_np, pred_test)
        
        print(f"\n{probe_name} Probe Performance:")
        print(f"  Train MAE: {train_mae:.4f}  |  Test MAE: {test_mae:.4f}")
        print(f"  Train R²:  {train_r2:.4f}  |  Test R²:  {test_r2:.4f}")
        print(f"  Pearson (test):  r={pearson_r_test:.4f}, p={pearson_p_test:.6f}")
        print(f"  Spearman (test): ρ={spearman_rho_test:.4f}, p={spearman_p_test:.6f}")
        
        if probe_name == "Non-Linear MLP":
            nonlinear_r2 = test_r2
        else:
            linear_r2 = test_r2
    
    # Interpretation
    improvement = nonlinear_r2 - linear_r2
    print(f"\n{'='*80}")
    print(f"Improvement (Non-Linear vs Linear): {improvement:.4f}")
    if nonlinear_r2 > 0.7:
        print(f"  ✓ STRONG non-linear relationship (R²={nonlinear_r2:.4f})")
    elif nonlinear_r2 > 0.4:
        print(f"  ~ MODERATE non-linear relationship (R²={nonlinear_r2:.4f})")
    elif nonlinear_r2 > 0.1:
        print(f"  ~ WEAK non-linear relationship (R²={nonlinear_r2:.4f})")
    else:
        print(f"  ✗ NO meaningful relationship (R²={nonlinear_r2:.4f})")
    print(f"{'='*80}")
    
    return {
        'nonlinear_train_mae': float(train_mae),
        'nonlinear_test_mae': float(test_mae),
        'nonlinear_train_r2': float(train_r2),
        'nonlinear_test_r2': float(nonlinear_r2),
        'linear_test_r2': float(linear_r2),
        'improvement': float(improvement),
        'pearson_r': float(pearson_r_test),
        'pearson_p': float(pearson_p_test),
        'spearman_rho': float(spearman_rho_test),
        'spearman_p': float(spearman_p_test)
    }


def test_linear_relationship(embeddings, positions, level_name, use_line_number=True):
    """
    Test for linear relationship between position and embeddings.
    
    Args:
        embeddings: (N, hidden_size) array of embeddings
        positions: (N,) array of position values (or will use line numbers if use_line_number=True)
        level_name: name of the test level
        use_line_number: if True, use line numbers (0, 1, 2, ...) instead of positions
    
    Uses:
    1. Linear probe (train simple linear model)
    2. Correlation analysis (Pearson and Spearman)
    """
    print(f"\n{'='*80}")
    print(f"Testing Linear Relationship: {level_name.upper()} Level")
    print(f"{'='*80}")
    
    # Use line numbers (sequential position in file) instead of address positions
    if use_line_number:
        positions = np.arange(len(embeddings), dtype=np.float32)
        print(f"Using LINE NUMBERS (0 to {len(embeddings)-1}) as position values")
    else:
        print(f"Using ACTUAL ADDRESS POSITIONS (from opcode addresses)")
        print(f"Position range: {positions.min():.6f} to {positions.max():.6f}")
        print(f"Position mean: {positions.mean():.6f}, std: {positions.std():.6f}")
    
    # Convert to torch tensors
    X = torch.tensor(embeddings, dtype=torch.float)
    y = torch.tensor(positions, dtype=torch.float).unsqueeze(1)
    
    # Split into train/test
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train linear probe
    hidden_size = X.shape[1]
    probe = nn.Sequential(
        nn.Linear(hidden_size, 1),
        nn.Sigmoid()
    )
    
    optimizer = optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training
    probe.train()
    for epoch in range(50):
        optimizer.zero_grad()
        pred = probe(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    probe.eval()
    with torch.no_grad():
        pred_train = probe(X_train).numpy().flatten()
        pred_test = probe(X_test).numpy().flatten()
    
    y_train_np = y_train.numpy().flatten()
    y_test_np = y_test.numpy().flatten()
    
    # Calculate metrics
    train_mae = np.mean(np.abs(pred_train - y_train_np))
    test_mae = np.mean(np.abs(pred_test - y_test_np))
    train_r2 = 1 - np.sum((y_train_np - pred_train)**2) / np.sum((y_train_np - y_train_np.mean())**2)
    test_r2 = 1 - np.sum((y_test_np - pred_test)**2) / np.sum((y_test_np - y_test_np.mean())**2)
    
    # Correlation analysis
    pearson_train, pearson_p_train = pearsonr(pred_train, y_train_np)
    pearson_test, pearson_p_test = pearsonr(pred_test, y_test_np)
    spearman_train, spearman_p_train = spearmanr(pred_train, y_train_np)
    spearman_test, spearman_p_test = spearmanr(pred_test, y_test_np)
    
    # Print results
    print(f"\nLinear Probe Performance:")
    print(f"  Train MAE: {train_mae:.4f}  |  Test MAE: {test_mae:.4f}")
    print(f"  Train R²:  {train_r2:.4f}  |  Test R²:  {test_r2:.4f}")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson (train):  r={pearson_train:.4f}, p={pearson_p_train:.6f}")
    print(f"  Pearson (test):   r={pearson_test:.4f}, p={pearson_p_test:.6f}")
    print(f"  Spearman (train): ρ={spearman_train:.4f}, p={spearman_p_train:.6f}")
    print(f"  Spearman (test):  ρ={spearman_test:.4f}, p={spearman_p_test:.6f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if test_r2 > 0.7:
        print(f"  ✓ STRONG linear relationship (R²={test_r2:.4f})")
    elif test_r2 > 0.4:
        print(f"  ✓ MODERATE linear relationship (R²={test_r2:.4f})")
    elif test_r2 > 0.1:
        print(f"  ~ WEAK linear relationship (R²={test_r2:.4f})")
    else:
        print(f"  ✗ NO linear relationship (R²={test_r2:.4f})")
    
    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'pearson_r': pearson_test,
        'pearson_p': pearson_p_test,
        'spearman_rho': spearman_test,
        'spearman_p': spearman_p_test
    }


def main():
    parser = argparse.ArgumentParser(description='Test linear relationship in position embeddings')
    parser.add_argument('--addressaware_model', type=str, required=True)
    parser.add_argument('--baseline_model', type=str, required=True)
    parser.add_argument('--binary_test', type=str, default='./test_data/binary_level_test.txt')
    parser.add_argument('--function_test', type=str, default='./test_data/function_level_test.txt')
    parser.add_argument('--bb_test', type=str, default='./test_data/bb_level_test.txt')
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum samples to use per level')
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
    print(f"Vocabulary size: {len(vocab)}")
    
    # ========================================================================
    # TEST ADDRESS-AWARE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("TESTING ADDRESS-AWARE MODEL")
    print("="*80)
    
    # Load model
    print(f"\nLoading address-aware model from: {args.addressaware_model}")
    aa_model = AddressAwareBERT(
        len(vocab), hidden=args.hidden_size, n_layers=12, attn_heads=8, max_len=512
    ).to(device)
    checkpoint = torch.load(args.addressaware_model, map_location=device)
    
    # Handle DataParallel wrapper (remove 'module.' prefix)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    aa_model.load_state_dict(state_dict, strict=False)
    aa_model.eval()
    print("✓ Model loaded")
    
    aa_results = {}
    
    # Test each level
    for level_name, test_file, position_key in [
        ('binary', args.binary_test, 'binary'),
        ('function', args.function_test, 'function'),
        ('bb', args.bb_test, 'bb')
    ]:
        if not os.path.exists(test_file):
            print(f"\n⚠ Skipping {level_name} level - file not found: {test_file}")
            continue
        
        print(f"\n{'-'*80}")
        print(f"Level: {level_name.upper()}")
        print(f"File: {test_file}")
        print(f"{'-'*80}")
        
        # Load dataset (address-aware mode - keeps position info)
        dataset = SingleInstructionDataset(test_file, vocab, is_addressaware=True)
        
        # Extract embeddings
        embeddings, positions = extract_embeddings_for_instructions(
            aa_model, dataset, device, vocab, is_addressaware=True, max_samples=args.max_samples
        )
        
        # Test NON-LINEAR relationship (MLP probe)
        nonlinear_results = test_nonlinear_relationship(embeddings, positions[position_key], level_name)
        
        # Test linear relationship using ACTUAL ADDRESS POSITIONS (bnorm/fnorm/bbnorm)
        linear_results = test_linear_relationship(embeddings, positions[position_key], level_name, use_line_number=False)
        
        # Combine results
        results = {**linear_results, **nonlinear_results}
        aa_results[level_name] = results
    
    # ========================================================================
    # TEST BASELINE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("TESTING BASELINE MODEL")
    print("="*80)
    
    # Load model
    print(f"\nLoading baseline model from: {args.baseline_model}")
    # Create BERT first, then wrap in BaselineBERTForPretraining
    bert = BERT(
        vocab_size=len(vocab),
        hidden=args.hidden_size,
        n_layers=12,
        attn_heads=8,
        dropout=0.1
    )
    bl_model = BaselineBERTForPretraining(bert, len(vocab)).to(device)
    
    checkpoint = torch.load(args.baseline_model, map_location=device)
    
    # Handle DataParallel wrapper (remove 'module.' prefix)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    bl_model.load_state_dict(state_dict, strict=False)
    bl_model.eval()
    print("✓ Model loaded")
    
    bl_results = {}
    
    # Test each level
    for level_name, test_file, position_key in [
        ('binary', args.binary_test, 'binary'),
        ('function', args.function_test, 'function'),
        ('bb', args.bb_test, 'bb')
    ]:
        if not os.path.exists(test_file):
            continue
        
        print(f"\n{'-'*80}")
        print(f"Level: {level_name.upper()}")
        print(f"File: {test_file}")
        print(f"{'-'*80}")
        
        # Load dataset (baseline mode - strips position info from tokens)
        dataset_baseline = SingleInstructionDataset(test_file, vocab, is_addressaware=False)
        
        # Extract embeddings (baseline model doesn't use position info in model)
        embeddings, _ = extract_embeddings_for_instructions(
            bl_model, dataset_baseline, device, vocab, is_addressaware=False, max_samples=args.max_samples
        )
        
        # Load address-aware dataset to extract ground truth positions
        # (We only need positions, not embeddings)
        dataset_for_positions = SingleInstructionDataset(test_file, vocab, is_addressaware=True)
        ground_truth_positions = []
        max_samples = args.max_samples or len(dataset_for_positions)
        for i in range(min(max_samples, len(dataset_for_positions))):
            tokens, positions = dataset_for_positions[i]
            # Get the position value for this level (from first token's position)
            if positions:
                if position_key == 'binary':
                    ground_truth_positions.append(positions[0][0])  # bnorm
                elif position_key == 'function':
                    ground_truth_positions.append(positions[0][1])  # fnorm
                elif position_key == 'bb':
                    ground_truth_positions.append(positions[0][2])  # bbnorm
            else:
                ground_truth_positions.append(0.0)
        
        ground_truth_positions = np.array(ground_truth_positions, dtype=np.float32)
        
        # Test NON-LINEAR relationship (MLP probe)
        nonlinear_results = test_nonlinear_relationship(embeddings, ground_truth_positions, level_name)
        
        # Test linear relationship using ACTUAL ADDRESS POSITIONS
        linear_results = test_linear_relationship(embeddings, ground_truth_positions, level_name, use_line_number=False)
        
        # Combine results
        results = {**linear_results, **nonlinear_results}
        bl_results[level_name] = results
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    output_file = os.path.join(args.output_dir, 'linear_relationship_results.json')
    results_dict = {
        'addressaware': aa_results,
        'baseline': bl_results,
        'config': vars(args)
    }
    
    # Convert numpy types to Python native types for JSON serialization
    results_dict = convert_to_serializable(results_dict)
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON: Address-Aware vs Baseline")
    print("="*80)
    
    for level in ['binary', 'function', 'bb']:
        if level in aa_results and level in bl_results:
            aa_r2 = aa_results[level]['test_r2']
            bl_r2 = bl_results[level]['test_r2']
            
            print(f"\n{level.upper()} Level:")
            print(f"  Address-Aware R²: {aa_r2:.4f}")
            print(f"  Baseline R²:      {bl_r2:.4f}")
            print(f"  Difference:       {aa_r2 - bl_r2:+.4f}")
            
            if aa_r2 > bl_r2 + 0.1:
                print(f"  → Address-aware shows BETTER linear relationship")
            elif aa_r2 < bl_r2 - 0.1:
                print(f"  → Baseline shows better linear relationship (unexpected!)")
            else:
                print(f"  → Similar linear relationships")


if __name__ == "__main__":
    main()
