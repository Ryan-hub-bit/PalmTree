"""
Test and compare MLM and NSP performance between PalmTree and AddressAware models.

This script evaluates:
1. PalmTree (baseline) - MLM and NSP losses/accuracies
2. AddressAware model - MLM and NSP losses/accuracies with address embeddings
3. Side-by-side comparison of both models on the same test data

IMPORTANT - Input Consistency:
    Both models receive IDENTICAL token sequences from the paired dataloader.
    
    The dataloader parses inline address format:
        mov(0x400000:0.12:0.12:0.12) rax addr_code(0x401000:0.13:0.13:0.13)
        
    And extracts:
        - Tokens: [mov, rax, addr_code]  <- Used by BOTH models
        - Address positions: [(0.12,0.12,0.12), (0,0,0), (0.13,0.13,0.13)]  <- Used ONLY by AddressAware
    
    All tokens are in PalmTree's vocabulary (no unknown tokens introduced).
    
    Key difference:
        - PalmTree: embedding = token_emb + position_emb + segment_emb
        - AddressAware: embedding = token_emb + position_emb + segment_emb + address_emb
    
    This ensures fair comparison:
        ✓ Same token inputs
        ✓ Same vocabulary
        ✓ Same masking/NSP pairs
        ✓ Only difference: AddressAware has additional hierarchical positional information
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
from tqdm import tqdm
from collections import defaultdict

# Import PalmTree components
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from palmtree.model.bert import BERT
from palmtree.model.language_model import BERTLM
from palmtree.dataset.vocab import WordVocab

# Import AddressAware components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import AddressAwareBERT, AddressAwareBERTForPretraining
from dataloader_paired import PairedAddressAwareDataset


class PalmTreeForPretraining(nn.Module):
    """Wrapper for PalmTree model to match testing interface."""
    
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.mask_lm = BERTLM(bert.hidden, vocab_size)
        self.nsp = nn.Linear(bert.hidden, 2)
        
    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.mask_lm(x), self.nsp(x[:, 0])


def load_palmtree_model(checkpoint_path, vocab_size, device):
    """Load PalmTree baseline model."""
    print(f"\n{'='*70}")
    print("Loading PalmTree Model (Baseline)")
    print(f"{'='*70}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model
    if hasattr(checkpoint, 'bert'):
        bert_model = checkpoint.bert
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Load architecture manually
        bert_model = BERT(vocab_size=vocab_size, hidden=128, n_layers=12, attn_heads=8, dropout=0.1)
        bert_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        raise ValueError("Cannot extract BERT model from checkpoint")
    
    # Create pretraining model
    model = PalmTreeForPretraining(bert_model, vocab_size)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded PalmTree model")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_addressaware_model(checkpoint_path, palmtree_checkpoint, vocab_size, device):
    """Load AddressAware model."""
    print(f"\n{'='*70}")
    print("Loading AddressAware Model")
    print(f"{'='*70}")
    
    # Create model
    bert = AddressAwareBERT(
        vocab_size=vocab_size,
        hidden=128,
        n_layers=12,
        attn_heads=8,
        dropout=0.1,
        palmtree_checkpoint=palmtree_checkpoint
    )
    
    model = AddressAwareBERTForPretraining(bert, vocab_size)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Loaded trained checkpoint")
    else:
        print("⚠ No checkpoint provided - using freshly initialized model")
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded AddressAware model")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def test_palmtree_cfg(model, data_loader, device, corpus_name="CFG"):
    """Test PalmTree model on CFG data (MLM + NSP).
    
    NOTE: PalmTree receives the SAME token sequences as AddressAware.
    The only difference is that PalmTree ignores address positions (not in forward pass).
    This ensures fair comparison - both models see identical tokens.
    """
    model.eval()
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    mlm_loss_total = 0
    nsp_loss_total = 0
    
    mlm_correct = 0
    mlm_total = 0
    nsp_correct = 0
    nsp_total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Testing PalmTree ({corpus_name})"):
            # Extract CFG data
            # IMPORTANT: Same token sequences as AddressAware (address positions ignored)
            token_ids = batch['cfg_bert_input'].to(device)
            segment_labels = batch['cfg_segment_label'].to(device)
            mlm_labels = batch['cfg_bert_label'].to(device)
            nsp_labels = batch['cfg_is_next'].to(device)
            # Note: cfg_binary_pos, cfg_function_pos, cfg_bb_pos are NOT used by PalmTree
            
            # Forward pass
            mlm_output, nsp_output = model(token_ids, segment_labels)
            
            # Calculate losses
            mlm_loss = mlm_criterion(mlm_output.transpose(1, 2), mlm_labels)
            nsp_loss = nsp_criterion(nsp_output, nsp_labels)
            loss = mlm_loss + nsp_loss
            
            # Accumulate
            total_loss += loss.item()
            mlm_loss_total += mlm_loss.item()
            nsp_loss_total += nsp_loss.item()
            
            # Accuracy
            mask = mlm_labels != -1
            if mask.any():
                mlm_pred = torch.argmax(mlm_output[mask], dim=-1)
                mlm_correct += (mlm_pred == mlm_labels[mask]).sum().item()
                mlm_total += mask.sum().item()
            
            nsp_pred = torch.argmax(nsp_output, dim=-1)
            nsp_correct += (nsp_pred == nsp_labels).sum().item()
            nsp_total += len(nsp_labels)
    
    n_batches = len(data_loader)
    return {
        'total_loss': total_loss / n_batches,
        'mlm_loss': mlm_loss_total / n_batches,
        'nsp_loss': nsp_loss_total / n_batches,
        'mlm_acc': mlm_correct / mlm_total if mlm_total > 0 else 0,
        'nsp_acc': nsp_correct / nsp_total if nsp_total > 0 else 0,
    }


def test_palmtree_dfg(model, data_loader, device, corpus_name="DFG"):
    """Test PalmTree model on DFG data (NSP only, no MLM).
    
    NOTE: PalmTree receives the SAME token sequences as AddressAware.
    Address positions are ignored by PalmTree (not in forward pass).
    """
    model.eval()
    
    nsp_criterion = nn.CrossEntropyLoss()
    
    nsp_loss_total = 0
    nsp_correct = 0
    nsp_total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Testing PalmTree ({corpus_name})"):
            # Extract DFG data
            # IMPORTANT: Same token sequences as AddressAware (address positions ignored)
            token_ids = batch['dfg_bert_input'].to(device)
            segment_labels = batch['dfg_segment_label'].to(device)
            nsp_labels = batch['dfg_is_next'].to(device)
            # Note: dfg_binary_pos, dfg_function_pos, dfg_bb_pos are NOT used by PalmTree
            
            # Forward pass (ignore MLM output)
            _, nsp_output = model(token_ids, segment_labels)
            
            # Calculate loss
            nsp_loss = nsp_criterion(nsp_output, nsp_labels)
            nsp_loss_total += nsp_loss.item()
            
            # Accuracy
            nsp_pred = torch.argmax(nsp_output, dim=-1)
            nsp_correct += (nsp_pred == nsp_labels).sum().item()
            nsp_total += len(nsp_labels)
    
    n_batches = len(data_loader)
    return {
        'nsp_loss': nsp_loss_total / n_batches,
        'nsp_acc': nsp_correct / nsp_total if nsp_total > 0 else 0,
    }


def test_addressaware(model, data_loader, device):
    """Test AddressAware model on paired CFG+DFG data."""
    model.eval()
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    
    # Metrics
    cfg_total_loss = 0
    cfg_mlm_loss = 0
    cfg_nsp_loss = 0
    cfg_mlm_correct = 0
    cfg_mlm_total = 0
    cfg_nsp_correct = 0
    cfg_nsp_total = 0
    
    dfg_nsp_loss = 0
    dfg_nsp_correct = 0
    dfg_nsp_total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing AddressAware"):
            # === Test CFG (MLM + NSP) ===
            cfg_token_ids = batch['cfg_bert_input'].to(device)
            cfg_segment_labels = batch['cfg_segment_label'].to(device)
            cfg_binary_pos = batch['cfg_binary_pos'].to(device)
            cfg_function_pos = batch['cfg_function_pos'].to(device)
            cfg_bb_pos = batch['cfg_bb_pos'].to(device)
            cfg_mlm_labels = batch['cfg_bert_label'].to(device)
            cfg_nsp_labels = batch['cfg_is_next'].to(device)
            
            # CFG forward
            cfg_mlm_output, cfg_nsp_output = model(
                cfg_token_ids, cfg_segment_labels,
                cfg_binary_pos, cfg_function_pos, cfg_bb_pos,
                corpus_type='cfg'
            )
            
            # CFG losses
            mlm_loss = mlm_criterion(cfg_mlm_output.transpose(1, 2), cfg_mlm_labels)
            nsp_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels)
            cfg_total_loss += (mlm_loss + nsp_loss).item()
            cfg_mlm_loss += mlm_loss.item()
            cfg_nsp_loss += nsp_loss.item()
            
            # CFG accuracy
            mask = cfg_mlm_labels != -1
            if mask.any():
                mlm_pred = torch.argmax(cfg_mlm_output[mask], dim=-1)
                cfg_mlm_correct += (mlm_pred == cfg_mlm_labels[mask]).sum().item()
                cfg_mlm_total += mask.sum().item()
            
            nsp_pred = torch.argmax(cfg_nsp_output, dim=-1)
            cfg_nsp_correct += (nsp_pred == cfg_nsp_labels).sum().item()
            cfg_nsp_total += len(cfg_nsp_labels)
            
            # === Test DFG (NSP only) ===
            dfg_token_ids = batch['dfg_bert_input'].to(device)
            dfg_segment_labels = batch['dfg_segment_label'].to(device)
            dfg_binary_pos = batch['dfg_binary_pos'].to(device)
            dfg_function_pos = batch['dfg_function_pos'].to(device)
            dfg_bb_pos = batch['dfg_bb_pos'].to(device)
            dfg_nsp_labels = batch['dfg_is_next'].to(device)
            
            # DFG forward
            _, dfg_nsp_output = model(
                dfg_token_ids, dfg_segment_labels,
                dfg_binary_pos, dfg_function_pos, dfg_bb_pos,
                corpus_type='dfg'
            )
            
            # DFG loss
            nsp_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
            dfg_nsp_loss += nsp_loss.item()
            
            # DFG accuracy
            nsp_pred = torch.argmax(dfg_nsp_output, dim=-1)
            dfg_nsp_correct += (nsp_pred == dfg_nsp_labels).sum().item()
            dfg_nsp_total += len(dfg_nsp_labels)
    
    n_batches = len(data_loader)
    return {
        'cfg': {
            'total_loss': cfg_total_loss / n_batches,
            'mlm_loss': cfg_mlm_loss / n_batches,
            'nsp_loss': cfg_nsp_loss / n_batches,
            'mlm_acc': cfg_mlm_correct / cfg_mlm_total if cfg_mlm_total > 0 else 0,
            'nsp_acc': cfg_nsp_correct / cfg_nsp_total if cfg_nsp_total > 0 else 0,
        },
        'dfg': {
            'nsp_loss': dfg_nsp_loss / n_batches,
            'nsp_acc': dfg_nsp_correct / dfg_nsp_total if dfg_nsp_total > 0 else 0,
        }
    }


def print_results(results, model_name):
    """Pretty print test results."""
    print(f"\n{'='*70}")
    print(f"{model_name} Test Results")
    print(f"{'='*70}")
    
    if 'cfg' in results:
        # AddressAware model (paired)
        print("\n📊 CFG Performance:")
        print(f"  Total Loss:  {results['cfg']['total_loss']:.4f}")
        print(f"  MLM Loss:    {results['cfg']['mlm_loss']:.4f}")
        print(f"  MLM Acc:     {results['cfg']['mlm_acc']*100:.2f}%")
        print(f"  NSP Loss:    {results['cfg']['nsp_loss']:.4f}")
        print(f"  NSP Acc:     {results['cfg']['nsp_acc']*100:.2f}%")
        
        print("\n📊 DFG Performance:")
        print(f"  NSP Loss:    {results['dfg']['nsp_loss']:.4f}")
        print(f"  NSP Acc:     {results['dfg']['nsp_acc']*100:.2f}%")
    else:
        # PalmTree model (separate tests)
        if 'mlm_loss' in results:
            # CFG test
            print("\n📊 CFG Performance:")
            print(f"  Total Loss:  {results['total_loss']:.4f}")
            print(f"  MLM Loss:    {results['mlm_loss']:.4f}")
            print(f"  MLM Acc:     {results['mlm_acc']*100:.2f}%")
            print(f"  NSP Loss:    {results['nsp_loss']:.4f}")
            print(f"  NSP Acc:     {results['nsp_acc']*100:.2f}%")
        else:
            # DFG test
            print("\n📊 DFG Performance:")
            print(f"  NSP Loss:    {results['nsp_loss']:.4f}")
            print(f"  NSP Acc:     {results['nsp_acc']*100:.2f}%")


def compare_results(palmtree_cfg, palmtree_dfg, addressaware):
    """Compare results between models."""
    print(f"\n{'='*70}")
    print("📈 Comparison: PalmTree vs AddressAware")
    print(f"{'='*70}")
    
    # CFG comparison
    print("\n🔍 CFG (MLM + NSP):")
    print(f"{'Metric':<20} {'PalmTree':<15} {'AddressAware':<15} {'Difference':<15}")
    print("-" * 70)
    
    pt_cfg = palmtree_cfg
    aa_cfg = addressaware['cfg']
    
    print(f"{'MLM Loss':<20} {pt_cfg['mlm_loss']:<15.4f} {aa_cfg['mlm_loss']:<15.4f} {aa_cfg['mlm_loss']-pt_cfg['mlm_loss']:+.4f}")
    print(f"{'MLM Accuracy':<20} {pt_cfg['mlm_acc']*100:<15.2f} {aa_cfg['mlm_acc']*100:<15.2f} {(aa_cfg['mlm_acc']-pt_cfg['mlm_acc'])*100:+.2f}%")
    print(f"{'NSP Loss':<20} {pt_cfg['nsp_loss']:<15.4f} {aa_cfg['nsp_loss']:<15.4f} {aa_cfg['nsp_loss']-pt_cfg['nsp_loss']:+.4f}")
    print(f"{'NSP Accuracy':<20} {pt_cfg['nsp_acc']*100:<15.2f} {aa_cfg['nsp_acc']*100:<15.2f} {(aa_cfg['nsp_acc']-pt_cfg['nsp_acc'])*100:+.2f}%")
    
    # DFG comparison
    print("\n🔍 DFG (NSP only):")
    print(f"{'Metric':<20} {'PalmTree':<15} {'AddressAware':<15} {'Difference':<15}")
    print("-" * 70)
    
    pt_dfg = palmtree_dfg
    aa_dfg = addressaware['dfg']
    
    print(f"{'NSP Loss':<20} {pt_dfg['nsp_loss']:<15.4f} {aa_dfg['nsp_loss']:<15.4f} {aa_dfg['nsp_loss']-pt_dfg['nsp_loss']:+.4f}")
    print(f"{'NSP Accuracy':<20} {pt_dfg['nsp_acc']*100:<15.2f} {aa_dfg['nsp_acc']*100:<15.2f} {(aa_dfg['nsp_acc']-pt_dfg['nsp_acc'])*100:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Test and compare PalmTree vs AddressAware models')
    
    # Data arguments
    parser.add_argument('--cfg_data', type=str, required=True, help='CFG corpus path')
    parser.add_argument('--dfg_data', type=str, required=True, help='DFG corpus path')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary path')
    
    # Model checkpoints
    parser.add_argument('--palmtree_checkpoint', type=str, required=True, help='PalmTree checkpoint')
    parser.add_argument('--addressaware_checkpoint', type=str, default=None, help='AddressAware checkpoint (optional)')
    
    # Test parameters
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for testing')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--test_samples', type=int, default=10000, help='Number of test samples')
    
    # Output
    parser.add_argument('--output', type=str, default='comparison_results.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create test dataset
    print(f"\nCreating test dataset...")
    test_dataset = PairedAddressAwareDataset(
        cfg_corpus_path=args.cfg_data,
        dfg_corpus_path=args.dfg_data,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        nsp_prob=0.5,
        mask_prob=0.15,
        data_percentage=min(1.0, args.test_samples / 3698396),  # Approximate
        train_split=1.0,  # Use all as test
        is_train=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Test batches: {len(test_loader)}")
    
    # Load models
    palmtree_model = load_palmtree_model(args.palmtree_checkpoint, len(vocab), device)
    addressaware_model = load_addressaware_model(
        args.addressaware_checkpoint,
        args.palmtree_checkpoint,
        len(vocab),
        device
    )
    
    # Test PalmTree
    print(f"\n{'='*70}")
    print("Testing PalmTree Model")
    print(f"{'='*70}")
    palmtree_cfg_results = test_palmtree_cfg(palmtree_model, test_loader, device, "CFG")
    palmtree_dfg_results = test_palmtree_dfg(palmtree_model, test_loader, device, "DFG")
    
    # Test AddressAware
    print(f"\n{'='*70}")
    print("Testing AddressAware Model")
    print(f"{'='*70}")
    addressaware_results = test_addressaware(addressaware_model, test_loader, device)
    
    # Print results
    print_results(palmtree_cfg_results, "PalmTree (CFG)")
    print_results(palmtree_dfg_results, "PalmTree (DFG)")
    print_results(addressaware_results, "AddressAware")
    
    # Comparison
    compare_results(palmtree_cfg_results, palmtree_dfg_results, addressaware_results)
    
    # Save results
    results = {
        'palmtree': {
            'cfg': palmtree_cfg_results,
            'dfg': palmtree_dfg_results,
        },
        'addressaware': addressaware_results,
        'test_config': {
            'test_samples': len(test_dataset),
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
