"""
Test the original PalmTree model on CFG and DFG data.

This script:
1. Loads the PalmTree checkpoint
2. Evaluates on CFG data (MLM + NSP)
3. Evaluates on DFG data (NSP only)
4. Reports detailed metrics for both
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import argparse
from tqdm import tqdm
import json
import re
import random

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.dataset.vocab import WordVocab
from palmtree.model.language_model import BERTLM


class SimplePalmTreeDataset(Dataset):
    """
    Simple dataset for PalmTree that removes address info from inline format.
    Converts: opcode(0xADDR:bnorm:fnorm:bbnorm) -> opcode
    """
    
    def __init__(self, cfg_corpus_path, dfg_corpus_path, vocab, seq_len=512,
                 nsp_prob=0.5, mask_prob=0.15):
        self.vocab = vocab
        self.seq_len = seq_len
        self.nsp_prob = nsp_prob
        self.mask_prob = mask_prob
        
        # Regex to strip address info
        self.addr_pattern = re.compile(r'(\w+)\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)')
        self.nested_pattern = re.compile(r'address\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)')
        
        # Load data
        print(f"Loading CFG corpus from {cfg_corpus_path}")
        self.cfg_lines = self._load_corpus(cfg_corpus_path)
        print(f"Loaded {len(self.cfg_lines)} CFG lines")
        
        print(f"Loading DFG corpus from {dfg_corpus_path}")
        self.dfg_lines = self._load_corpus(dfg_corpus_path)
        print(f"Loaded {len(self.dfg_lines)} DFG lines")
        
        self.n_cfg = len(self.cfg_lines)
        self.n_dfg = len(self.dfg_lines)
    
    def _load_corpus(self, path):
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines
    
    def _strip_addresses(self, instruction_str):
        """Remove address information and extract clean tokens."""
        # Replace opcode(address info) with just opcode
        cleaned = self.addr_pattern.sub(r'\1', instruction_str)
        # Replace address(address info) with just 'address'
        cleaned = self.nested_pattern.sub('address', cleaned)
        # Split into tokens
        tokens = cleaned.split()
        return tokens
    
    def _get_nsp_pair(self, index, corpus):
        """Get NSP pair from corpus."""
        line1 = corpus[index % len(corpus)]
        t1_tokens = self._strip_addresses(line1)
        
        if random.random() < self.nsp_prob:
            line2 = corpus[random.randint(0, len(corpus) - 1)]
            is_next = 0
        else:
            if (index + 1) % len(corpus) < len(corpus):
                line2 = corpus[(index + 1) % len(corpus)]
            else:
                line2 = corpus[0]
            is_next = 1
        
        t2_tokens = self._strip_addresses(line2)
        return t1_tokens, t2_tokens, is_next
    
    def _mask_tokens(self, tokens):
        """Apply MLM masking."""
        output_tokens = []
        output_labels = []
        
        for token in tokens:
            prob = random.random()
            
            if prob < self.mask_prob:
                prob = random.random()
                if prob < 0.8:
                    output_tokens.append(self.vocab.mask_index)
                elif prob < 0.9:
                    output_tokens.append(random.randint(0, len(self.vocab) - 1))
                else:
                    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_labels.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            else:
                output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_labels.append(-1)
        
        return output_tokens, output_labels
    
    def _pad_sequence(self, sequence, max_len, pad_value):
        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [pad_value] * (max_len - len(sequence))
    
    def __len__(self):
        return self.n_cfg
    
    def __getitem__(self, index):
        # Get CFG pair
        cfg_t1, cfg_t2, cfg_is_next = self._get_nsp_pair(index, self.cfg_lines)
        
        # Get DFG pair
        dfg_index = index % self.n_dfg if self.n_dfg > 0 else 0
        if self.n_dfg > 0:
            dfg_t1, dfg_t2, dfg_is_next = self._get_nsp_pair(dfg_index, self.dfg_lines)
        else:
            dfg_t1, dfg_t2, dfg_is_next = [], [], 0
        
        # === Process CFG (with MLM) ===
        cfg_combined = ['<sos>'] + cfg_t1 + ['<eos>'] + cfg_t2 + ['<eos>']
        cfg_segment = [0] * (len(cfg_t1) + 2) + [1] * (len(cfg_t2) + 1)
        
        if len(cfg_combined) > self.seq_len:
            cfg_combined = cfg_combined[:self.seq_len]
            cfg_segment = cfg_segment[:self.seq_len]
        
        cfg_input, cfg_label = self._mask_tokens(cfg_combined)
        cfg_input = self._pad_sequence(cfg_input, self.seq_len, self.vocab.pad_index)
        cfg_label = self._pad_sequence(cfg_label, self.seq_len, -1)
        cfg_segment = self._pad_sequence(cfg_segment, self.seq_len, 0)
        
        # === Process DFG (NO MLM) ===
        if self.n_dfg > 0:
            dfg_combined = ['<sos>'] + dfg_t1 + ['<eos>'] + dfg_t2 + ['<eos>']
            dfg_segment = [0] * (len(dfg_t1) + 2) + [1] * (len(dfg_t2) + 1)
            
            if len(dfg_combined) > self.seq_len:
                dfg_combined = dfg_combined[:self.seq_len]
                dfg_segment = dfg_segment[:self.seq_len]
            
            dfg_input = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in dfg_combined]
            dfg_input = self._pad_sequence(dfg_input, self.seq_len, self.vocab.pad_index)
            dfg_segment = self._pad_sequence(dfg_segment, self.seq_len, 0)
        else:
            dfg_input = [self.vocab.pad_index] * self.seq_len
            dfg_segment = [0] * self.seq_len
        
        return {
            'cfg_bert_input': torch.tensor(cfg_input, dtype=torch.long),
            'cfg_bert_label': torch.tensor(cfg_label, dtype=torch.long),
            'cfg_segment_label': torch.tensor(cfg_segment, dtype=torch.long),
            'cfg_is_next': torch.tensor(cfg_is_next, dtype=torch.long),
            'dfg_bert_input': torch.tensor(dfg_input, dtype=torch.long),
            'dfg_segment_label': torch.tensor(dfg_segment, dtype=torch.long),
            'dfg_is_next': torch.tensor(dfg_is_next, dtype=torch.long),
        }


def evaluate_palmtree(model, data_loader, device, verbose=True):
    """Evaluate PalmTree model."""
    model.eval()
    
    # Check if model has bert attribute or is BERT directly
    has_wrapper = hasattr(model, 'bert')
    
    total_loss = 0
    mlm_loss_total = 0
    nsp_cfg_loss_total = 0
    nsp_dfg_loss_total = 0
    
    mlm_correct = 0
    mlm_total = 0
    nsp_cfg_correct = 0
    nsp_cfg_total = 0
    nsp_dfg_correct = 0
    nsp_dfg_total = 0
    
    # PalmTree uses LogSoftmax, so we need NLLLoss (not CrossEntropyLoss)
    mlm_criterion = nn.NLLLoss(ignore_index=-1)
    nsp_criterion = nn.NLLLoss()
    
    with torch.no_grad():
        iterator = tqdm(data_loader, desc="Evaluating") if verbose else data_loader
        
        for batch in iterator:
            # === CFG ===
            cfg_input = batch['cfg_bert_input'].to(device)
            cfg_segment = batch['cfg_segment_label'].to(device)
            cfg_mlm_labels = batch['cfg_bert_label'].to(device)
            cfg_nsp_labels = batch['cfg_is_next'].to(device)
            
            # Forward (PalmTree uses BERTLM with MLM, CWP, DUP)
            cfg_output = model.bert(cfg_input, cfg_segment)
            cfg_mlm_output = model.MLM(cfg_output)
            cfg_nsp_output = model.CWP(cfg_output)
            
            # Losses
            mlm_loss = mlm_criterion(cfg_mlm_output.transpose(1, 2), cfg_mlm_labels)
            nsp_cfg_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels)
            
            # MLM accuracy
            mask = cfg_mlm_labels != -1
            if mask.any():
                mlm_pred = torch.argmax(cfg_mlm_output[mask], dim=-1)
                mlm_correct += (mlm_pred == cfg_mlm_labels[mask]).sum().item()
                mlm_total += mask.sum().item()
            
            # NSP CFG accuracy
            nsp_cfg_pred = torch.argmax(cfg_nsp_output, dim=-1)
            nsp_cfg_correct += (nsp_cfg_pred == cfg_nsp_labels).sum().item()
            nsp_cfg_total += len(cfg_nsp_labels)
            
            # === DFG ===
            dfg_input = batch['dfg_bert_input'].to(device)
            dfg_segment = batch['dfg_segment_label'].to(device)
            dfg_nsp_labels = batch['dfg_is_next'].to(device)
            
            dfg_output = model.bert(dfg_input, dfg_segment)
            dfg_nsp_output = model.DUP(dfg_output)
            
            nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
            
            # NSP DFG accuracy
            nsp_dfg_pred = torch.argmax(dfg_nsp_output, dim=-1)
            nsp_dfg_correct += (nsp_dfg_pred == dfg_nsp_labels).sum().item()
            nsp_dfg_total += len(dfg_nsp_labels)
            
            # Accumulate
            combined_loss = mlm_loss + nsp_cfg_loss + nsp_dfg_loss
            total_loss += combined_loss.item()
            mlm_loss_total += mlm_loss.item()
            nsp_cfg_loss_total += nsp_cfg_loss.item()
            nsp_dfg_loss_total += nsp_dfg_loss.item()
    
    n_batches = len(data_loader)
    
    return {
        'total_loss': total_loss / n_batches,
        'mlm_loss': mlm_loss_total / n_batches,
        'mlm_accuracy': mlm_correct / mlm_total if mlm_total > 0 else 0.0,
        'mlm_correct': mlm_correct,
        'mlm_total': mlm_total,
        'nsp_cfg_loss': nsp_cfg_loss_total / n_batches,
        'nsp_cfg_accuracy': nsp_cfg_correct / nsp_cfg_total if nsp_cfg_total > 0 else 0.0,
        'nsp_cfg_correct': nsp_cfg_correct,
        'nsp_cfg_total': nsp_cfg_total,
        'nsp_dfg_loss': nsp_dfg_loss_total / n_batches,
        'nsp_dfg_accuracy': nsp_dfg_correct / nsp_dfg_total if nsp_dfg_total > 0 else 0.0,
        'nsp_dfg_correct': nsp_dfg_correct,
        'nsp_dfg_total': nsp_dfg_total,
    }


def print_results(results, title="Evaluation Results"):
    """Pretty print results."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)
    
    print(f"\n{'Combined Loss:':<25} {results['total_loss']:.4f}")
    
    print(f"\n{'CFG - Masked Language Modeling (MLM)':}")
    print(f"  {'Loss:':<23} {results['mlm_loss']:.4f}")
    print(f"  {'Accuracy:':<23} {results['mlm_accuracy']:.2%} ({results['mlm_correct']:,} / {results['mlm_total']:,})")
    
    print(f"\n{'CFG - Next Sentence Prediction (NSP)':}")
    print(f"  {'Loss:':<23} {results['nsp_cfg_loss']:.4f}")
    print(f"  {'Accuracy:':<23} {results['nsp_cfg_accuracy']:.2%} ({results['nsp_cfg_correct']:,} / {results['nsp_cfg_total']:,})")
    
    print(f"\n{'DFG - Next Sentence Prediction (NSP)':}")
    print(f"  {'Loss:':<23} {results['nsp_dfg_loss']:.4f}")
    print(f"  {'Accuracy:':<23} {results['nsp_dfg_accuracy']:.2%} ({results['nsp_dfg_correct']:,} / {results['nsp_dfg_total']:,})")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test original PalmTree model")
    
    parser.add_argument("--checkpoint", type=str,
                       default="../pre-trained_model/palmtree/transformer.ep19",
                       help="Path to PalmTree checkpoint")
    parser.add_argument("--cfg_data", type=str,
                       default="../data/test/cfg/all_cfg_combined.txt",
                       help="CFG data file")
    parser.add_argument("--dfg_data", type=str,
                       default="../data/test/dfg/all_dfg_combined.txt",
                       help="DFG data file")
    parser.add_argument("--vocab", type=str,
                       default="../pre-trained_model/palmtree/vocab",
                       help="Vocabulary file")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Data loader workers")
    parser.add_argument("--seq_len", type=int, default=512,
                       help="Sequence length")
    parser.add_argument("--cuda", action="store_true", default=True,
                       help="Use CUDA")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load vocabulary
    print(f"\nLoading vocabulary: {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load PalmTree model
    print(f"\nLoading PalmTree checkpoint: {args.checkpoint}")
    bert_model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Wrap BERT with language model heads
    print("Creating BERTLM wrapper with task heads...")
    model = BERTLM(bert_model, len(vocab))
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dataset
    print(f"\nLoading test data...")
    test_dataset = SimplePalmTreeDataset(
        cfg_corpus_path=args.cfg_data,
        dfg_corpus_path=args.dfg_data,
        vocab=vocab,
        seq_len=args.seq_len,
        nsp_prob=0.5,
        mask_prob=0.15
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate
    print("\n" + "="*70)
    print("Starting evaluation...")
    print("="*70)
    
    results = evaluate_palmtree(model, test_loader, device, verbose=True)
    
    # Print results
    print_results(results, "Original PalmTree Model - Test Results")
    
    # Save results
    output_file = "palmtree_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
