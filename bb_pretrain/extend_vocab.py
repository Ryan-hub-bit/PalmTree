"""
Extend PalmTree vocabulary with address tokens
Creates a new extended vocabulary file
"""
import pickle
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pre-trained_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import config


class ExtendedVocab:
    """Extended vocabulary class that can be pickled"""
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.unk_index = stoi.get('<unk>', 0)
        self.pad_index = stoi.get('<pad>', 1)
        self.sos_index = stoi.get('<sos>', 2)
        self.eos_index = stoi.get('<eos>', 3)
        self.mask_index = stoi.get('<mask>', 4)
    
    def __len__(self):
        return len(self.stoi)


def extend_palmtree_vocab(original_vocab_path, output_vocab_path):
    """
    Extend PalmTree vocabulary with address tokens
    
    Args:
        original_vocab_path: Path to original PalmTree vocab
        output_vocab_path: Path to save extended vocab
    """
    print("=" * 80)
    print("Extending PalmTree Vocabulary with Address Tokens")
    print("=" * 80)
    
    # Load original vocab
    print(f"\n1. Loading original vocabulary from: {original_vocab_path}")
    try:
        from vocab import WordVocab
        vocab = WordVocab.load_vocab(original_vocab_path)
        print(f"   Original vocab size: {len(vocab)}")
        print(f"   Type: WordVocab object")
        
        # Get the vocabulary dictionary
        if hasattr(vocab, 'stoi'):
            vocab_dict = vocab.stoi.copy()
            itos = vocab.itos.copy()
        else:
            print("   ERROR: Vocab object doesn't have 'stoi' attribute")
            return False
            
    except Exception as e:
        print(f"   Failed to load with WordVocab: {e}")
        print(f"   Trying pickle...")
        
        try:
            with open(original_vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            
            if hasattr(vocab, 'stoi'):
                vocab_dict = vocab.stoi.copy()
                itos = vocab.itos.copy()
            else:
                vocab_dict = vocab.copy()
                itos = {idx: token for token, idx in vocab_dict.items()}
                
            print(f"   Original vocab size: {len(vocab_dict)}")
            print(f"   Loaded as dict")
        except Exception as e2:
            print(f"   ERROR: Failed to load vocab: {e2}")
            return False
    
    # Get next available index
    max_idx = max(vocab_dict.values())
    next_idx = max_idx + 1
    
    print(f"\n2. Adding address tokens (starting from index {next_idx}):")
    
    # Add address tokens
    for token in config.SPECIAL_ADDR_TOKENS:
        if token in vocab_dict:
            print(f"   ⚠️  Token '{token}' already exists at index {vocab_dict[token]}")
        else:
            vocab_dict[token] = next_idx
            itos.append(token)  # Append to list, not assign by index
            print(f"   ✓ Added '{token}' at index {next_idx}")
            next_idx += 1
    
    print(f"\n3. Extended vocab size: {len(vocab_dict)}")
    print(f"   Added {len(config.SPECIAL_ADDR_TOKENS)} new tokens")
    
    # Save extended vocab
    print(f"\n4. Saving extended vocabulary to: {output_vocab_path}")
    
    extended_vocab = ExtendedVocab(vocab_dict, itos)
    
    with open(output_vocab_path, 'wb') as f:
        pickle.dump(extended_vocab, f)
    
    print(f"   ✓ Saved successfully!")
    
    # Verify
    print(f"\n5. Verification:")
    with open(output_vocab_path, 'rb') as f:
        loaded = pickle.load(f)
    print(f"   Loaded vocab size: {len(loaded.stoi)}")
    
    print(f"\n   Address token IDs:")
    for token in config.SPECIAL_ADDR_TOKENS:
        if token in loaded.stoi:
            print(f"     {token:20s} → {loaded.stoi[token]}")
    
    print("\n" + "=" * 80)
    print("✓ Vocabulary extension complete!")
    print("=" * 80)
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extend PalmTree vocabulary with address tokens')
    parser.add_argument('--input_vocab', type=str, 
                        default='../pre-trained_model/palmtree/vocab',
                        help='Path to original PalmTree vocabulary')
    parser.add_argument('--output_vocab', type=str,
                        default='../pre-trained_model/palmtree/vocab_extended',
                        help='Path to save extended vocabulary')
    
    args = parser.parse_args()
    
    success = extend_palmtree_vocab(args.input_vocab, args.output_vocab)
    
    if success:
        print(f"\nNow use this vocab for training:")
        print(f"  python train.py --vocab_file {args.output_vocab} ...")
    else:
        print("\nFailed to extend vocabulary!")
        sys.exit(1)


if __name__ == '__main__':
    main()
