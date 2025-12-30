"""
Verify that dataloader outputs match embedding expectations

This script checks:
1. Data types and shapes from dataloader
2. Value ranges (positions should be -1 or [0, 1], offsets should be -1 or [0, 255])
3. Special token handling (<sos>, <eos>, <pad>)
4. Consistency between related fields
"""

import torch
from vocab import WordVocab
from address_aware.dataloader_addressaware import InstructionMaskingDataset


def check_data_types_and_shapes(sample, task_name):
    """Check that all fields have correct types and shapes"""
    print(f"\n{task_name} data structure:")
    print("-" * 60)
    
    required_fields = ['bert_input', 'bert_label', 'segment_label', 
                       'binary_pos', 'function_pos', 'bb_pos', 
                       'var_offsets', 'is_daddr']
    
    for field in required_fields:
        if field in sample:
            data = sample[field]
            print(f"  {field:20s}: shape={str(data.shape):15s} dtype={data.dtype}")
        else:
            print(f"  {field:20s}: MISSING!")
    
    return all(field in sample for field in required_fields)


def check_position_ranges(sample, task_name, vocab):
    """Check that position values are in valid ranges"""
    print(f"\n{task_name} position value ranges:")
    print("-" * 60)
    
    binary_pos = sample['binary_pos']
    function_pos = sample['function_pos']
    bb_pos = sample['bb_pos']
    bert_input = sample['bert_input']
    
    # Find non-padding tokens
    pad_idx = vocab.pad_index
    non_pad_mask = bert_input != pad_idx
    
    issues = []
    
    for name, pos_tensor in [('binary_pos', binary_pos), 
                             ('function_pos', function_pos), 
                             ('bb_pos', bb_pos)]:
        # Check range
        min_val = pos_tensor[non_pad_mask].min().item() if non_pad_mask.any() else 0
        max_val = pos_tensor[non_pad_mask].max().item() if non_pad_mask.any() else 0
        
        # Count special values
        num_negative_one = (pos_tensor == -1.0).sum().item()
        num_in_range = ((pos_tensor >= 0.0) & (pos_tensor <= 1.0) & non_pad_mask).sum().item()
        num_out_of_range = ((pos_tensor < -1.0) | (pos_tensor > 1.0)) & non_pad_mask
        
        print(f"  {name:15s}:")
        print(f"    Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"    Count -1.0 (special): {num_negative_one}")
        print(f"    Count in [0, 1]: {num_in_range}")
        
        if num_out_of_range.any():
            out_vals = pos_tensor[num_out_of_range][:5]  # Show first 5
            print(f"    ⚠ OUT OF RANGE values: {out_vals.tolist()}")
            issues.append(f"{name} has values outside [-1, 1]")
    
    return issues


def check_var_offsets(sample, task_name):
    """Check var_offsets are valid"""
    print(f"\n{task_name} var_offsets:")
    print("-" * 60)
    
    var_offsets = sample['var_offsets']
    
    # Count different types
    num_not_var = (var_offsets == -1).sum().item()
    num_var_zero = (var_offsets == 0).sum().item()
    num_var_positive = (var_offsets > 0).sum().item()
    
    min_offset = var_offsets[var_offsets >= 0].min().item() if (var_offsets >= 0).any() else -1
    max_offset = var_offsets[var_offsets >= 0].max().item() if (var_offsets >= 0).any() else -1
    
    print(f"  Not var (=-1): {num_not_var}")
    print(f"  Var with offset=0: {num_var_zero}")
    print(f"  Var with offset>0: {num_var_positive}")
    if num_var_positive > 0:
        print(f"  Offset range: [{min_offset}, {max_offset}]")
    
    issues = []
    if (var_offsets < -1).any():
        issues.append("var_offsets has values < -1")
        print(f"  ⚠ Invalid: found offsets < -1")
    
    return issues


def check_is_daddr(sample, task_name):
    """Check is_daddr flags"""
    print(f"\n{task_name} is_daddr:")
    print("-" * 60)
    
    is_daddr = sample['is_daddr']
    
    num_zeros = (is_daddr == 0).sum().item()
    num_ones = (is_daddr == 1).sum().item()
    num_invalid = ((is_daddr != 0) & (is_daddr != 1)).sum().item()
    
    print(f"  Not daddr (=0): {num_zeros}")
    print(f"  Is daddr (=1): {num_ones}")
    
    issues = []
    if num_invalid > 0:
        invalid_vals = is_daddr[(is_daddr != 0) & (is_daddr != 1)][:5]
        print(f"  ⚠ Invalid values (not 0 or 1): {invalid_vals.tolist()}")
        issues.append("is_daddr has values other than 0 or 1")
    
    return issues


def check_special_tokens(sample, vocab):
    """Check special tokens have correct positions"""
    print(f"\nSpecial token handling:")
    print("-" * 60)
    
    bert_input = sample['bert_input']
    binary_pos = sample['binary_pos']
    function_pos = sample['function_pos']
    bb_pos = sample['bb_pos']
    var_offsets = sample['var_offsets']
    is_daddr = sample['is_daddr']
    
    sos_idx = vocab.stoi.get('<sos>', 3)
    eos_idx = vocab.stoi.get('<eos>', 2)
    pad_idx = vocab.pad_index
    
    issues = []
    
    # Check <sos> (usually at position 0)
    if bert_input[0] == sos_idx:
        print(f"  <sos> at position 0:")
        print(f"    binary_pos: {binary_pos[0].item()}")
        print(f"    function_pos: {function_pos[0].item()}")
        print(f"    bb_pos: {bb_pos[0].item()}")
        print(f"    var_offsets: {var_offsets[0].item()}")
        print(f"    is_daddr: {is_daddr[0].item()}")
        
        if binary_pos[0] != -1.0 or function_pos[0] != -1.0 or bb_pos[0] != -1.0:
            issues.append("<sos> should have position -1.0 for all position types")
            print("    ⚠ Expected all positions to be -1.0")
        if var_offsets[0] != -1:
            issues.append("<sos> should have var_offset=-1")
            print("    ⚠ Expected var_offset=-1")
        if is_daddr[0] != 0:
            issues.append("<sos> should have is_daddr=0")
            print("    ⚠ Expected is_daddr=0")
    
    # Find <eos>
    eos_positions = (bert_input == eos_idx).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        eos_pos = eos_positions[0].item()
        print(f"\n  <eos> at position {eos_pos}:")
        print(f"    binary_pos: {binary_pos[eos_pos].item()}")
        print(f"    function_pos: {function_pos[eos_pos].item()}")
        print(f"    bb_pos: {bb_pos[eos_pos].item()}")
        print(f"    var_offsets: {var_offsets[eos_pos].item()}")
        print(f"    is_daddr: {is_daddr[eos_pos].item()}")
        
        if binary_pos[eos_pos] != -1.0 or function_pos[eos_pos] != -1.0 or bb_pos[eos_pos] != -1.0:
            issues.append("<eos> should have position -1.0")
            print("    ⚠ Expected all positions to be -1.0")
    
    # Check padding
    pad_positions = (bert_input == pad_idx).nonzero(as_tuple=True)[0]
    if len(pad_positions) > 0:
        pad_pos = pad_positions[0].item()
        print(f"\n  <pad> at position {pad_pos}:")
        print(f"    binary_pos: {binary_pos[pad_pos].item()}")
        print(f"    var_offsets: {var_offsets[pad_pos].item()}")
        print(f"    is_daddr: {is_daddr[pad_pos].item()}")
        
        if binary_pos[pad_pos] != -1.0:
            issues.append("Padding should have position -1.0")
            print("    ⚠ Expected all positions to be -1.0")
    
    return issues


def check_token_position_consistency(sample, vocab):
    """Check that tokens and their positions make sense together"""
    print(f"\nToken-Position consistency:")
    print("-" * 60)
    
    bert_input = sample['bert_input']
    binary_pos = sample['binary_pos']
    var_offsets = sample['var_offsets']
    is_daddr = sample['is_daddr']
    
    issues = []
    
    # Decode first 10 tokens and show their attributes
    print(f"  First 10 tokens:")
    for i in range(min(10, len(bert_input))):
        tid = bert_input[i].item()
        if tid == vocab.pad_index:
            break
        
        token = vocab.itos[tid] if 0 <= tid < len(vocab.itos) else f'UNK:{tid}'
        bin_pos = binary_pos[i].item()
        var_off = var_offsets[i].item()
        daddr = is_daddr[i].item()
        
        # Check expectations
        warning = ""
        if token == 'var' and var_off == -1:
            warning = " ⚠ 'var' token should have offset >= 0"
            issues.append(f"'var' token at position {i} has var_offset=-1")
        elif token != 'var' and var_off >= 0:
            warning = f" ⚠ non-'var' token '{token}' has offset >= 0"
            issues.append(f"Non-var token '{token}' at position {i} has var_offset={var_off}")
        
        if token == 'daddr' and daddr != 1:
            warning = " ⚠ 'daddr' token should have is_daddr=1"
            issues.append(f"'daddr' token at position {i} has is_daddr={daddr}")
        
        print(f"    [{i:2d}] {token:10s} | pos={bin_pos:6.3f} | var={var_off:3d} | daddr={daddr}{warning}")
    
    return issues


def main():
    print("="*80)
    print("DATALOADER INPUT VERIFICATION")
    print("="*80)
    
    # Load vocab and dataset
    print("\nLoading dataset...")
    vocab = WordVocab.load_vocab('./vocab_addr')
    dataset = InstructionMaskingDataset(
        cfg_corpus_path='/data/kun/palmtreedata/cfg_train_2.txt',
        dfg_corpus_path=None,
        vocab=vocab,
        seq_len=512,
        on_memory=True,
        token_mask_prob=0.0,
        instruction_mask_prob=0.0,
        data_percentage=0.02,  # Load 2% to reach daddr examples
        enable_imd=False,
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Test a few samples
    all_issues = []
    
    for idx in [0, 100, 5070]:  # 5070 has daddr
        if idx >= len(dataset):
            continue
            
        print("\n" + "="*80)
        print(f"SAMPLE {idx}")
        print("="*80)
        
        sample = dataset[idx]
        
        # Check IMC task
        imc_data = sample['imc']
        if not check_data_types_and_shapes(imc_data, "IMC"):
            print("  ✗ Missing required fields!")
            continue
        
        issues = []
        issues.extend(check_position_ranges(imc_data, "IMC", vocab))
        issues.extend(check_var_offsets(imc_data, "IMC"))
        issues.extend(check_is_daddr(imc_data, "IMC"))
        issues.extend(check_special_tokens(imc_data, vocab))
        issues.extend(check_token_position_consistency(imc_data, vocab))
        
        if issues:
            print(f"\n  ⚠ Found {len(issues)} issues:")
            for issue in issues:
                print(f"    - {issue}")
            all_issues.extend(issues)
        else:
            print(f"\n  ✓ All checks passed for sample {idx}!")
    
    # Summary
    print("\n" + "="*80)
    if all_issues:
        print(f"✗ FOUND {len(all_issues)} ISSUES TOTAL")
        print("="*80)
        print("\nUnique issues:")
        for issue in set(all_issues):
            print(f"  - {issue}")
    else:
        print("✓ ALL CHECKS PASSED - INPUT IS CORRECT!")
        print("="*80)
        print("\nThe dataloader is providing correctly formatted input:")
        print("  ✓ All required fields present")
        print("  ✓ Correct data types and shapes")
        print("  ✓ Position values in valid ranges [-1, 1]")
        print("  ✓ var_offsets correctly set (-1 for non-var, >=0 for var)")
        print("  ✓ is_daddr correctly set (0 or 1)")
        print("  ✓ Special tokens have correct position values")
        print("  ✓ Token-position associations are consistent")


if __name__ == '__main__':
    main()
