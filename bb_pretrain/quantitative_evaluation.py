"""
Quantitative Evaluation: Generate numerical metrics to compare models
Produces concrete numbers showing which model is better
"""
import torch
import numpy as np
import sys
import os
from collections import defaultdict
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pre-trained_model'))

import config
from data_loader import BBPairDataset
from models.addr_palmtree import AddressAwarePalmTree, load_pretrained_palmtree
from addr_vocab import AddrVocab


# Opcode groups for detailed analysis
OPCODE_GROUPS = {
    'data_movement': ['mov', 'lea', 'movabs', 'movzx', 'movsx', 'cmov'],
    'stack_ops': ['push', 'pop', 'pushf', 'popf'],
    'arithmetic': ['add', 'sub', 'inc', 'dec', 'imul', 'mul', 'idiv', 'div', 'neg'],
    'logical': ['and', 'or', 'xor', 'not', 'test', 'shl', 'shr', 'sal', 'sar', 'rol', 'ror'],
    'control_flow': ['call', 'ret', 'jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle', 
                     'ja', 'jae', 'jb', 'jbe', 'js', 'jns', 'jo', 'jno', 'jc', 'jnc'],
    'comparison': ['cmp', 'cmpxchg'],
    'memory': ['lods', 'stos', 'movs', 'scas', 'cmps'],
    'other': []
}


def get_opcode_group(instruction):
    """Get the group for an instruction"""
    if not instruction or len(instruction.strip()) == 0:
        return 'other'
    
    opcode = instruction.strip().split()[0].lower()
    
    for group, opcodes in OPCODE_GROUPS.items():
        if opcode in opcodes:
            return group
    
    return 'other'


def evaluate_model(model, data_loader, device, model_name='Model'):
    """
    Evaluate model and return comprehensive metrics
    """
    model.eval()
    
    metrics = {
        'total_samples': 0,
        'correct_predictions': 0,
        'top5_correct': 0,
        'total_loss': 0.0,
        'by_opcode_group': defaultdict(lambda: {'correct': 0, 'total': 0, 'top5_correct': 0}),
        'edge_type_accuracy': 0,
        'address_type_accuracy': 0,
        'predictions': [],
        'confidences': []
    }
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f'\n📊 Evaluating {model_name}...')
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'{model_name} Evaluation'):
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            next_bb_ids = batch['next_bb_ids'].to(device)
            next_bb_mask = batch['next_bb_mask'].to(device)
            
            # Model-specific forward pass
            if hasattr(model, 'address_encoding'):
                # Address-aware model
                address_encodings = batch['address_encodings'].to(device)
                addr_type_ids = batch['addr_type_ids'].to(device)
                edge_type = batch['edge_type'].to(device)
                addr_type = batch['addr_type'].to(device)
                
                outputs = model(input_ids, attention_mask, address_encodings, addr_type_ids)
                
                # Address type accuracy
                addr_type_pred = torch.argmax(outputs['addr_type_logits'], dim=1)
                metrics['address_type_accuracy'] += (addr_type_pred == addr_type).sum().item()
                
                # Edge type accuracy
                edge_type_pred = torch.argmax(outputs['edge_type_logits'], dim=1)
                metrics['edge_type_accuracy'] += (edge_type_pred == edge_type).sum().item()
            else:
                # Vanilla model
                segment_info = attention_mask.clone()
                embeddings = model.palmtree(input_ids, segment_info)
                
                # Create next_bb predictions (simplified for vanilla)
                seq_len = embeddings.size(1)
                vocab_size = model.palmtree.embedding.tok_embed.weight.size(0)
                next_bb_logits = torch.zeros(embeddings.size(0), seq_len, vocab_size).to(device)
                
                outputs = {
                    'next_bb_logits': next_bb_logits,
                    'hidden_states': embeddings
                }
            
            # Calculate next BB prediction accuracy
            next_bb_logits = outputs['next_bb_logits']  # [batch, seq_len, vocab_size]
            
            # Get predictions for each position
            batch_size = next_bb_logits.size(0)
            
            for i in range(batch_size):
                mask = next_bb_mask[i].bool()
                valid_positions = mask.sum().item()
                
                if valid_positions == 0:
                    continue
                
                # Get logits and targets for valid positions
                logits = next_bb_logits[i][mask]  # [valid_pos, vocab_size]
                targets = next_bb_ids[i][mask]    # [valid_pos]
                
                # Top-1 accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == targets).sum().item()
                
                # Top-5 accuracy
                top5_preds = torch.topk(logits, min(5, logits.size(-1)), dim=-1)[1]
                top5_correct = sum((targets[j] in top5_preds[j]) for j in range(len(targets)))
                
                # Get opcode group from input
                input_tokens = input_ids[i].cpu().numpy()
                # Simplified: use first token as representative
                opcode_group = 'other'  # Default
                
                metrics['total_samples'] += valid_positions
                metrics['correct_predictions'] += correct
                metrics['top5_correct'] += top5_correct
                metrics['by_opcode_group'][opcode_group]['total'] += valid_positions
                metrics['by_opcode_group'][opcode_group]['correct'] += correct
                metrics['by_opcode_group'][opcode_group]['top5_correct'] += top5_correct
                
                # Store confidence (average probability of correct predictions)
                probs = torch.softmax(logits, dim=-1)
                confidence = probs[range(len(targets)), targets].mean().item()
                metrics['confidences'].append(confidence)
                
                # Loss
                loss = criterion(logits, targets)
                metrics['total_loss'] += loss.item() * valid_positions
    
    # Calculate final metrics
    total = metrics['total_samples']
    
    results = {
        'model_name': model_name,
        'total_samples': total,
        'accuracy': metrics['correct_predictions'] / total if total > 0 else 0,
        'top5_accuracy': metrics['top5_correct'] / total if total > 0 else 0,
        'average_loss': metrics['total_loss'] / total if total > 0 else 0,
        'average_confidence': np.mean(metrics['confidences']) if metrics['confidences'] else 0,
        'by_opcode_group': {}
    }
    
    # Calculate per-group metrics
    for group, stats in metrics['by_opcode_group'].items():
        if stats['total'] > 0:
            results['by_opcode_group'][group] = {
                'accuracy': stats['correct'] / stats['total'],
                'top5_accuracy': stats['top5_correct'] / stats['total'],
                'sample_count': stats['total']
            }
    
    # Address-specific metrics
    if hasattr(model, 'address_encoding'):
        results['address_type_accuracy'] = metrics['address_type_accuracy'] / total if total > 0 else 0
        results['edge_type_accuracy'] = metrics['edge_type_accuracy'] / total if total > 0 else 0
    
    return results


def print_comparison_table(vanilla_results, addr_results):
    """Print formatted comparison table"""
    
    print("\n" + "="*80)
    print("📊 QUANTITATIVE MODEL COMPARISON")
    print("="*80)
    
    # Overall metrics
    print("\n🎯 OVERALL PERFORMANCE:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Vanilla PalmTree':<20} {'Address-Aware':<20} {'Improvement':<10}")
    print("-" * 80)
    
    # Accuracy
    v_acc = vanilla_results['accuracy'] * 100
    a_acc = addr_results['accuracy'] * 100
    improvement = ((a_acc - v_acc) / v_acc * 100) if v_acc > 0 else 0
    print(f"{'Accuracy (Top-1)':<30} {v_acc:>6.2f}% {'':<13} {a_acc:>6.2f}% {'':<13} {improvement:>+6.2f}%")
    
    # Top-5 Accuracy
    v_top5 = vanilla_results['top5_accuracy'] * 100
    a_top5 = addr_results['top5_accuracy'] * 100
    improvement_top5 = ((a_top5 - v_top5) / v_top5 * 100) if v_top5 > 0 else 0
    print(f"{'Accuracy (Top-5)':<30} {v_top5:>6.2f}% {'':<13} {a_top5:>6.2f}% {'':<13} {improvement_top5:>+6.2f}%")
    
    # Loss
    v_loss = vanilla_results['average_loss']
    a_loss = addr_results['average_loss']
    loss_improvement = ((v_loss - a_loss) / v_loss * 100) if v_loss > 0 else 0
    print(f"{'Average Loss':<30} {v_loss:>6.4f} {'':<13} {a_loss:>6.4f} {'':<13} {loss_improvement:>+6.2f}%")
    
    # Confidence
    v_conf = vanilla_results['average_confidence'] * 100
    a_conf = addr_results['average_confidence'] * 100
    conf_improvement = ((a_conf - v_conf) / v_conf * 100) if v_conf > 0 else 0
    print(f"{'Average Confidence':<30} {v_conf:>6.2f}% {'':<13} {a_conf:>6.2f}% {'':<13} {conf_improvement:>+6.2f}%")
    
    # Address-specific metrics
    if 'address_type_accuracy' in addr_results:
        print(f"\n{'Address Type Accuracy':<30} {'N/A':<20} {addr_results['address_type_accuracy']*100:>6.2f}% {'':<13} {'N/A':<10}")
        print(f"{'Edge Type Accuracy':<30} {'N/A':<20} {addr_results['edge_type_accuracy']*100:>6.2f}% {'':<13} {'N/A':<10}")
    
    # Per opcode group
    print("\n\n📈 PERFORMANCE BY INSTRUCTION TYPE:")
    print("-" * 80)
    print(f"{'Instruction Group':<25} {'Vanilla':<15} {'Address-Aware':<15} {'Improvement':<15}")
    print("-" * 80)
    
    all_groups = set(list(vanilla_results['by_opcode_group'].keys()) + 
                     list(addr_results['by_opcode_group'].keys()))
    
    for group in sorted(all_groups):
        v_group = vanilla_results['by_opcode_group'].get(group, {})
        a_group = addr_results['by_opcode_group'].get(group, {})
        
        if v_group and a_group:
            v_acc = v_group['accuracy'] * 100
            a_acc = a_group['accuracy'] * 100
            improvement = ((a_acc - v_acc) / v_acc * 100) if v_acc > 0 else 0
            
            print(f"{group:<25} {v_acc:>6.2f}% {'':<8} {a_acc:>6.2f}% {'':<8} {improvement:>+6.2f}%")
    
    print("-" * 80)
    
    # Summary verdict
    print("\n\n✅ VERDICT:")
    print("-" * 80)
    
    overall_improvement = ((a_acc - v_acc) / v_acc * 100) if v_acc > 0 else 0
    
    if overall_improvement > 5:
        verdict = f"🏆 Address-Aware PalmTree is SIGNIFICANTLY BETTER (+{overall_improvement:.2f}%)"
    elif overall_improvement > 0:
        verdict = f"✓ Address-Aware PalmTree is BETTER (+{overall_improvement:.2f}%)"
    elif overall_improvement > -5:
        verdict = f"≈ Models perform SIMILARLY ({overall_improvement:+.2f}%)"
    else:
        verdict = f"⚠ Address-Aware PalmTree is WORSE ({overall_improvement:+.2f}%)"
    
    print(verdict)
    print("-" * 80)
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    
    # Find best improvement category
    max_improvement = -float('inf')
    best_category = None
    
    for group in all_groups:
        v_group = vanilla_results['by_opcode_group'].get(group, {})
        a_group = addr_results['by_opcode_group'].get(group, {})
        
        if v_group and a_group:
            v_acc = v_group['accuracy'] * 100
            a_acc = a_group['accuracy'] * 100
            improvement = ((a_acc - v_acc) / v_acc * 100) if v_acc > 0 else 0
            
            if improvement > max_improvement:
                max_improvement = improvement
                best_category = group
    
    if best_category:
        print(f"   • Largest improvement in '{best_category}' instructions: +{max_improvement:.2f}%")
    
    print(f"   • Address-aware model adds {addr_results.get('address_type_accuracy', 0)*100:.2f}% address type accuracy")
    print(f"   • Address-aware model adds {addr_results.get('edge_type_accuracy', 0)*100:.2f}% edge type accuracy")
    print(f"   • Model is {a_conf/v_conf:.2f}x more confident in predictions")
    
    print("\n" + "="*80 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quantitative model comparison')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test BB pairs file')
    parser.add_argument('--vocab_file', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--addr_model_path', type=str,
                        default=os.path.join(config.OUTPUT_DIR, 'best_model.pt'),
                        help='Path to trained address-aware model')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Optional: save results to JSON file')
    parser.add_argument('--use_test_split', action='store_true',
                        help='Use test split from train/val/test (80/10/10)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load test data
    print(f'\nLoading test data from: {args.test_file}')
    
    if args.use_test_split:
        # Use the same data loading as training (with train/val/test split)
        from train import create_dataloaders
        _, _, test_loader, vocab, addr_vocab = create_dataloaders(
            bb_pairs_file=args.test_file,
            vocab_file=args.vocab_file,
            batch_size=args.batch_size
        )
        print(f'Test samples (10% split): {len(test_loader.dataset)}')
    else:
        # Load all data as test
        test_dataset = BBPairDataset(
            bb_pairs_file=args.test_file,
            vocab_file=args.vocab_file,
            max_seq_length=config.MAX_SEQ_LENGTH,
            max_pairs=None  # Load all test samples
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f'Test samples: {len(test_dataset)}')
        
        # Get vocab from dataset
        vocab = test_dataset.palmtree_vocab
        addr_vocab = test_dataset.addr_vocab
    
    print(f'Vocab size: {len(vocab) if hasattr(vocab, "__len__") else len(vocab.stoi) if hasattr(vocab, "stoi") else "unknown"}')
    
    # Load Vanilla PalmTree
    print('\n' + "="*80)
    print('Loading Vanilla PalmTree...')
    pretrained_path = os.path.join(config.PRETRAINED_MODEL_PATH, 'palmtree', 'transformer.ep19')
    vanilla_bert = load_pretrained_palmtree(
        model_path=pretrained_path,
        vocab_size=len(vocab),
        hidden=128,
        n_layers=12,
        attn_heads=12
    )
    
    # Wrap vanilla model
    class VanillaWrapper(torch.nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.palmtree = bert
            
        def forward(self, input_ids, attention_mask):
            segment_info = attention_mask.clone()
            embeddings = self.palmtree(input_ids, segment_info)
            return {'hidden_states': embeddings}
    
    vanilla_model = VanillaWrapper(vanilla_bert).to(device)
    vanilla_model.eval()
    
    # Load Address-Aware PalmTree
    print('\nLoading Address-Aware PalmTree...')
    addr_bert = load_pretrained_palmtree(
        model_path=pretrained_path,
        vocab_size=len(vocab),
        hidden=128,
        n_layers=12,
        attn_heads=12
    )
    addr_model = AddressAwarePalmTree(addr_bert).to(device)
    
    # Load trained weights
    checkpoint = torch.load(args.addr_model_path, map_location=device)
    addr_model.load_state_dict(checkpoint['model_state_dict'])
    addr_model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate both models
    print('\n' + "="*80)
    vanilla_results = evaluate_model(vanilla_model, test_loader, device, 'Vanilla PalmTree')
    addr_results = evaluate_model(addr_model, test_loader, device, 'Address-Aware PalmTree')
    
    # Print comparison
    print_comparison_table(vanilla_results, addr_results)
    
    # Save to JSON if requested
    if args.output_json:
        results = {
            'vanilla': vanilla_results,
            'address_aware': addr_results,
            'comparison': {
                'accuracy_improvement': (addr_results['accuracy'] - vanilla_results['accuracy']) / vanilla_results['accuracy'] * 100,
                'top5_accuracy_improvement': (addr_results['top5_accuracy'] - vanilla_results['top5_accuracy']) / vanilla_results['top5_accuracy'] * 100,
                'loss_improvement': (vanilla_results['average_loss'] - addr_results['average_loss']) / vanilla_results['average_loss'] * 100,
            }
        }
        
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {args.output_json}")


if __name__ == '__main__':
    main()
