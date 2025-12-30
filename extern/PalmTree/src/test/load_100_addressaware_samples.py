"""
Load 100 samples from Address-Aware DataLoader and display their text and tensor representations

This script loads samples from the address-aware dataset and shows:
- IMC (Instruction Masking CFG): Instructions masked at instruction-level in CFG
- IMD (Instruction Masking DFG): Instructions masked at instruction-level in DFG (if enabled)
- MLM (Masked Language Model): Tokens masked at token-level in CFG
- Original instruction text
- BERT input tensors
- BERT label tensors
- Address embeddings
"""

import torch
import sys
sys.path.insert(0, 'src')

from vocab import WordVocab
from address_aware.dataloader_addressaware import InstructionMaskingDataset


def decode_tokens(token_ids, vocab):
    """Decode token IDs back to words"""
    tokens = []
    for tid in token_ids:
        tid_val = tid.item() if torch.is_tensor(tid) else tid
        if tid_val == vocab.stoi.get('<pad>', 0):
            break  # Stop at padding
        elif 0 <= tid_val < len(vocab.itos):
            tokens.append(vocab.itos[tid_val])
        else:
            tokens.append(f'<UNK:{tid_val}>')
    return tokens


def format_with_masking(tokens, labels):
    """Format tokens showing masking"""
    result = []
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if token == '<pad>':
            break
        if label != -1:  # This token is masked (address-aware uses -1, not -100)
            result.append(f"[{token}→MASKED]")
        else:
            result.append(token)
    return ' '.join(result)


def reconstruct_original_from_labels(tokens, labels, vocab):
    """Reconstruct original tokens from masked tokens and labels"""
    original = []
    for token, label in zip(tokens, labels):
        if token == '<pad>':
            break
        if label != -1 and token == '<mask>':
            # This was masked, show original from label (address-aware uses -1, not -100)
            if 0 <= label < len(vocab.itos):
                original.append(vocab.itos[label])
            else:
                original.append(f'<UNK:{label}>')
        else:
            original.append(token)
    return ' '.join(original)


def count_masked_tokens(bert_input, labels, pad_idx=0):
    """
    Count how many tokens are masked, excluding padding.
    
    Args:
        bert_input: The input token IDs (to identify padding)
        labels: The label values (-1 for non-masked, token_id for masked)
        pad_idx: The padding token ID (default 0)
    
    Returns:
        masked: Number of tokens that are masked (label != -1)
        total: Total number of non-padding tokens
    """
    masked = 0
    total = 0
    
    for inp_token, label in zip(bert_input, labels):
        inp_val = inp_token.item() if hasattr(inp_token, 'item') else inp_token
        label_val = label.item() if hasattr(label, 'item') else label
        
        # Skip padding tokens
        if inp_val == pad_idx:
            continue
            
        # This is a real token (not padding)
        total += 1
        if label_val != -1:
            # This token is masked
            masked += 1
    
    return masked, total


def main():
    output_file = 'addressaware_100_samples_detailed.txt'
    
    # Load vocab
    print("Loading vocab...")
    vocab = WordVocab.load_vocab("./vocab_addr")
    print(f"Vocabulary loaded: {len(vocab)} tokens")
    
    # Create dataset without IMD (only IMC and MLM)
    print("\nCreating Address-Aware dataset (IMD disabled)...")
    train_cfg = "/data/kun/palmtreedata/cfg_train_2.txt"
    train_dfg = "/data/kun/palmtreedata/dfg_train_2.txt"
    
    dataset = InstructionMaskingDataset(
        cfg_corpus_path=train_cfg,
        dfg_corpus_path=train_dfg,
        vocab=vocab,
        seq_len=20,
        on_memory=True,
        token_mask_prob=0.15,  # MLM token masking rate
        instruction_mask_prob=0.25,  # IMC instruction masking rate
        data_percentage=0.01,  # Load only 1% for quick testing
        enable_imd=False,  # Disable DFG instruction masking
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"CFG lines: {len(dataset.cfg_lines)}")
    print(f"DFG lines: {len(dataset.dfg_lines) if dataset.dfg_lines else 0}")
    print(f"IMD enabled: {dataset.enable_imd}")
    
    # Load 100 samples
    num_samples = min(100, len(dataset))
    print(f"\nLoading {num_samples} samples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"Address-Aware Dataset: {num_samples} Samples\n")
        f.write("="*100 + "\n\n")
        f.write("Tasks:\n")
        f.write("- IMC: Instruction Masking CFG (entire instructions masked)\n")
        f.write("- MLM: Masked Language Model (individual tokens masked)\n")
        f.write("Note: IMD (Instruction Masking DFG) is DISABLED\n")
        f.write("\n" + "="*100 + "\n\n")
        
        for idx in range(num_samples):
            sample = dataset[idx]
            
            # Extract IMC data
            imc_data = sample['imc']
            imc_input = imc_data['bert_input']
            imc_label = imc_data['bert_label']
            imc_binary_pos = imc_data['binary_pos']
            imc_function_pos = imc_data['function_pos']
            imc_bb_pos = imc_data['bb_pos']
            imc_var_offsets = imc_data['var_offsets']
            imc_is_daddr = imc_data['is_daddr']
            
            # Extract MLM data
            mlm_data = sample['mlm']
            mlm_input = mlm_data['bert_input']
            mlm_label = mlm_data['bert_label']
            mlm_binary_pos = mlm_data['binary_pos']
            mlm_function_pos = mlm_data['function_pos']
            mlm_bb_pos = mlm_data['bb_pos']
            mlm_var_offsets = mlm_data['var_offsets']
            mlm_is_daddr = mlm_data['is_daddr']
            
            # Decode tokens
            imc_tokens = decode_tokens(imc_input, vocab)
            mlm_tokens = decode_tokens(mlm_input, vocab)
            
            # Count masked tokens
            imc_masked, imc_total = count_masked_tokens(imc_input, imc_label)
            mlm_masked, mlm_total = count_masked_tokens(mlm_input, mlm_label)
            
            imc_mask_pct = (imc_masked / imc_total * 100) if imc_total > 0 else 0
            mlm_mask_pct = (mlm_masked / mlm_total * 100) if mlm_total > 0 else 0
            
            # Write sample header
            f.write(f"{'='*100}\n")
            f.write(f"SAMPLE {idx + 1}/{num_samples}\n")
            f.write(f"{'='*100}\n\n")
            
            # IMC (Instruction Masking CFG)
            f.write("="*80 + "\n")
            f.write("IMC (Instruction Masking CFG)\n")
            f.write("="*80 + "\n")
            f.write(f"Masking: {imc_masked}/{imc_total} tokens ({imc_mask_pct:.1f}%)\n\n")
            
            # Original instructions (reconstruct from labels)
            original_imc = reconstruct_original_from_labels(imc_tokens, imc_label, vocab)
            f.write("ORIGINAL CFG (before instruction masking):\n")
            f.write(f"  {original_imc}\n\n")
            
            # Masked instructions
            masked_imc = format_with_masking(imc_tokens, imc_label)
            f.write("MASKED CFG (after instruction masking):\n")
            f.write(f"  {masked_imc}\n\n")
            
            # Address-related embeddings sample (first 20)
            binary_pos_sample = imc_binary_pos[:20].tolist()
            function_pos_sample = imc_function_pos[:20].tolist()
            bb_pos_sample = imc_bb_pos[:20].tolist()
            var_offsets_sample = imc_var_offsets[:20].tolist()
            is_daddr_sample = imc_is_daddr[:20].tolist()
            f.write(f"Binary positions (first 20): {binary_pos_sample}\n")
            f.write(f"Function positions (first 20): {function_pos_sample}\n")
            f.write(f"BB positions (first 20): {bb_pos_sample}\n")
            f.write(f"Var offsets (first 20): {var_offsets_sample}\n")
            f.write(f"Is daddr (first 20): {is_daddr_sample}\n\n")
            
            # Tensor info
            f.write("TENSOR INFO:\n")
            f.write(f"  bert_input shape: {imc_input.shape}\n")
            f.write(f"  bert_label shape: {imc_label.shape}\n")
            f.write(f"  binary_pos shape: {imc_binary_pos.shape}\n")
            f.write(f"  function_pos shape: {imc_function_pos.shape}\n")
            f.write(f"  bb_pos shape: {imc_bb_pos.shape}\n")
            f.write(f"  var_offsets shape: {imc_var_offsets.shape}\n")
            f.write(f"  is_daddr shape: {imc_is_daddr.shape}\n")
            f.write(f"  bert_input (first 20): {imc_input[:20].tolist()}\n")
            f.write(f"  bert_label (first 20): {imc_label[:20].tolist()}\n\n")
            
            # MLM (Token Masking CFG)
            f.write("="*80 + "\n")
            f.write("MLM (Token Masking CFG)\n")
            f.write("="*80 + "\n")
            f.write(f"Masking: {mlm_masked}/{mlm_total} tokens ({mlm_mask_pct:.1f}%)\n\n")
            
            # Original instructions (reconstruct from labels)
            original_mlm = reconstruct_original_from_labels(mlm_tokens, mlm_label, vocab)
            f.write("ORIGINAL CFG (before token masking):\n")
            f.write(f"  {original_mlm}\n\n")
            
            # Masked instructions
            masked_mlm = format_with_masking(mlm_tokens, mlm_label)
            f.write("MASKED CFG (after token masking):\n")
            f.write(f"  {masked_mlm}\n\n")
            
            # Address-related embeddings sample (first 20)
            binary_pos_sample_mlm = mlm_binary_pos[:20].tolist()
            function_pos_sample_mlm = mlm_function_pos[:20].tolist()
            bb_pos_sample_mlm = mlm_bb_pos[:20].tolist()
            var_offsets_sample_mlm = mlm_var_offsets[:20].tolist()
            is_daddr_sample_mlm = mlm_is_daddr[:20].tolist()
            f.write(f"Binary positions (first 20): {binary_pos_sample_mlm}\n")
            f.write(f"Function positions (first 20): {function_pos_sample_mlm}\n")
            f.write(f"BB positions (first 20): {bb_pos_sample_mlm}\n")
            f.write(f"Var offsets (first 20): {var_offsets_sample_mlm}\n")
            f.write(f"Is daddr (first 20): {is_daddr_sample_mlm}\n\n")
            
            # Tensor info
            f.write("TENSOR INFO:\n")
            f.write(f"  bert_input shape: {mlm_input.shape}\n")
            f.write(f"  bert_label shape: {mlm_label.shape}\n")
            f.write(f"  binary_pos shape: {mlm_binary_pos.shape}\n")
            f.write(f"  function_pos shape: {mlm_function_pos.shape}\n")
            f.write(f"  bb_pos shape: {mlm_bb_pos.shape}\n")
            f.write(f"  var_offsets shape: {mlm_var_offsets.shape}\n")
            f.write(f"  is_daddr shape: {mlm_is_daddr.shape}\n")
            f.write(f"  bert_input (first 20): {mlm_input[:20].tolist()}\n")
            f.write(f"  bert_label (first 20): {mlm_label[:20].tolist()}\n\n")
            
            # IMD (Instruction Masking DFG) - if enabled
            if 'imd' in sample:
                imd_data = sample['imd']
                imd_input = imd_data['bert_input']
                imd_label = imd_data['bert_label']
                imd_binary_pos = imd_data['binary_pos']
                imd_var_offsets = imd_data['var_offsets']
                imd_is_daddr = imd_data['is_daddr']
                
                imd_tokens = decode_tokens(imd_input, vocab)
                imd_masked, imd_total = count_masked_tokens(imd_input, imd_label)
                imd_mask_pct = (imd_masked / imd_total * 100) if imd_total > 0 else 0
                
                f.write("="*80 + "\n")
                f.write("IMD (Instruction Masking DFG)\n")
                f.write("="*80 + "\n")
                f.write(f"Masking: {imd_masked}/{imd_total} tokens ({imd_mask_pct:.1f}%)\n\n")
                
                # Original instructions
                original_imd = reconstruct_original_from_labels(imd_tokens, imd_label, vocab)
                f.write("ORIGINAL DFG (before instruction masking):\n")
                f.write(f"  {original_imd}\n\n")
                
                # Masked instructions
                masked_imd = format_with_masking(imd_tokens, imd_label)
                f.write("MASKED DFG (after instruction masking):\n")
                f.write(f"  {masked_imd}\n\n")
                
                # Address-related embeddings sample (first 20)
                binary_pos_sample_imd = imd_binary_pos[:20].tolist()
                var_offsets_sample_imd = imd_var_offsets[:20].tolist()
                is_daddr_sample_imd = imd_is_daddr[:20].tolist()
                f.write(f"Binary positions (first 20): {binary_pos_sample_imd}\n")
                f.write(f"Var offsets (first 20): {var_offsets_sample_imd}\n")
                f.write(f"Is daddr (first 20): {is_daddr_sample_imd}\n\n")
                
                # Tensor info
                f.write("TENSOR INFO:\n")
                f.write(f"  bert_input shape: {imd_input.shape}\n")
                f.write(f"  bert_label shape: {imd_label.shape}\n")
                f.write(f"  binary_pos shape: {imd_binary_pos.shape}\n")
                f.write(f"  var_offsets shape: {imd_var_offsets.shape}\n")
                f.write(f"  is_daddr shape: {imd_is_daddr.shape}\n")
                f.write(f"  bert_input (first 20): {imd_input[:20].tolist()}\n")
                f.write(f"  bert_label (first 20): {imd_label[:20].tolist()}\n\n")
            
            f.write("\n")
        
        # Summary statistics
        f.write("="*100 + "\n")
        f.write("SUMMARY\n")
        f.write("="*100 + "\n")
        f.write(f"Total samples processed: {num_samples}\n")
        f.write(f"Dataset configuration:\n")
        f.write(f"  - Token mask probability (MLM): {dataset.token_mask_prob}\n")
        f.write(f"  - Instruction mask probability (IMC): {dataset.instruction_mask_prob}\n")
        f.write(f"  - Sequence length: {dataset.seq_len}\n")
        f.write(f"  - Enable IMD: {dataset.enable_imd} (DISABLED)\n")
        f.write(f"\nTasks:\n")
        f.write(f"  - IMC: Instruction-level masking on CFG (~{dataset.instruction_mask_prob*100}% of instructions)\n")
        f.write(f"  - MLM: Token-level masking on CFG (~{dataset.token_mask_prob*100}% of tokens)\n")
    
    print(f"\n✓ Output written to: {output_file}")
    
    # Print quick stats
    print("\nQuick validation:")
    print(f"  Sample 1 has {len(sample.keys())} task(s):")
    for task_name in sample.keys():
        print(f"    - {task_name.upper()}")
    
    print(f"\n  IMC masking rate: ~{dataset.instruction_mask_prob*100}% (instruction-level)")
    print(f"  MLM masking rate: ~{dataset.token_mask_prob*100}% (token-level)")
    print(f"  IMD: DISABLED")


if __name__ == '__main__':
    main()
