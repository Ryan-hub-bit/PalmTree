"""
Test script to find and verify daddr (destination address) handling in the dataloader
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


def main():
    # Load vocab
    print("Loading vocab...")
    vocab = WordVocab.load_vocab("./vocab_addr")
    print(f"Vocabulary loaded: {len(vocab)} tokens")
    
    # Check if 'daddr' is in vocab
    if 'daddr' in vocab.stoi:
        print(f"✓ 'daddr' token found in vocab: ID = {vocab.stoi['daddr']}")
    else:
        print("✗ 'daddr' token NOT found in vocab!")
        return
    
    # Create dataset
    print("\nCreating Address-Aware dataset...")
    train_cfg = "/data/kun/palmtreedata/cfg_train_2.txt"
    
    dataset = InstructionMaskingDataset(
        cfg_corpus_path=train_cfg,
        dfg_corpus_path=None,
        vocab=vocab,
        seq_len=512,
        on_memory=True,
        token_mask_prob=0.0,  # Disable token masking for easier inspection
        instruction_mask_prob=0.0,  # Disable instruction masking for easier inspection
        data_percentage=0.02,  # Load 2% to reach line 5071
        enable_imd=False,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Search for samples with daddr
    print("\n" + "="*80)
    print("Searching for daddr instructions...")
    print("="*80 + "\n")
    
    daddr_token_id = vocab.stoi['daddr']
    found_count = 0
    
    for idx in range(min(1000, len(dataset))):
        sample = dataset[idx]
        imc_data = sample['imc']
        
        imc_input = imc_data['bert_input']
        imc_is_daddr = imc_data['is_daddr']
        imc_binary_pos = imc_data['binary_pos']
        imc_function_pos = imc_data['function_pos']
        imc_bb_pos = imc_data['bb_pos']
        
        # Check if this sample has daddr
        has_daddr = False
        daddr_positions = []
        
        for i, (token_id, is_da) in enumerate(zip(imc_input, imc_is_daddr)):
            tid = token_id.item() if torch.is_tensor(token_id) else token_id
            is_da_val = is_da.item() if torch.is_tensor(is_da) else is_da
            
            if tid == daddr_token_id:
                has_daddr = True
                daddr_positions.append(i)
            
            # Also check is_daddr flag
            if is_da_val == 1 and i not in daddr_positions:
                print(f"⚠ Warning: is_daddr=1 at position {i} but token is not 'daddr'")
        
        if has_daddr:
            found_count += 1
            tokens = decode_tokens(imc_input, vocab)
            
            print(f"Sample {idx}: Found {len(daddr_positions)} daddr token(s)")
            print(f"Instruction sequence: {' '.join(tokens)}")
            print()
            
            # Show detailed info for each daddr
            for pos in daddr_positions:
                is_da_val = imc_is_daddr[pos].item() if torch.is_tensor(imc_is_daddr[pos]) else imc_is_daddr[pos]
                binary = imc_binary_pos[pos].item() if torch.is_tensor(imc_binary_pos[pos]) else imc_binary_pos[pos]
                func = imc_function_pos[pos].item() if torch.is_tensor(imc_function_pos[pos]) else imc_function_pos[pos]
                bb = imc_bb_pos[pos].item() if torch.is_tensor(imc_bb_pos[pos]) else imc_bb_pos[pos]
                
                print(f"  Position {pos}:")
                print(f"    Token: {tokens[pos]}")
                print(f"    is_daddr flag: {is_da_val} {'✓' if is_da_val == 1 else '✗ SHOULD BE 1!'}")
                print(f"    Binary position: {binary}")
                print(f"    Function position: {func}")
                print(f"    BB position: {bb}")
                print()
            
            # Show context around daddr
            for pos in daddr_positions:
                start = max(0, pos - 3)
                end = min(len(tokens), pos + 4)
                context = tokens[start:end]
                is_daddr_context = [imc_is_daddr[i].item() for i in range(start, end)]
                
                print(f"  Context around position {pos}:")
                for i, (tok, is_da) in enumerate(zip(context, is_daddr_context)):
                    marker = "→→→" if start + i == pos else "   "
                    flag = f"[is_daddr={is_da}]"
                    print(f"    {marker} {tok} {flag}")
                print()
            
            print("="*80 + "\n")
            
            if found_count >= 5:
                break
    
    print(f"\nSummary: Found {found_count} samples with daddr tokens (out of {min(1000, len(dataset))} checked)")
    
    if found_count == 0:
        print("\n⚠ No daddr tokens found! This could mean:")
        print("  1. The data doesn't contain daddr instructions in the sampled portion")
        print("  2. The regex pattern is not matching daddr in the data")
        print("  3. The vocab doesn't have 'daddr' token")


if __name__ == '__main__':
    main()
