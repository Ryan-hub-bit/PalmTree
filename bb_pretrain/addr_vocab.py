"""
Address Vocabulary - Separate from PalmTree instruction vocabulary
Handles special address tokens and address types
"""
from collections import Counter


class AddrVocab:
    """
    Vocabulary for address-related tokens
    Separate from the main PalmTree instruction vocabulary
    """
    
    def __init__(self):
        # Special address tokens
        self.pad_index = 0
        self.unk_index = 1
        self.addr_start_index = 2
        self.addr_end_index = 3
        self.addr_code_index = 4
        self.addr_data_index = 5
        self.addr_tgt_index = 6
        self.addr_unknown_index = 7
        self.seq_index = 8  # Separator between instructions
        
        # Build stoi (string to index) mapping
        self.stoi = {
            "<pad>": self.pad_index,
            "<unk>": self.unk_index,
            "<addr_start>": self.addr_start_index,
            "<addr_end>": self.addr_end_index,
            "<addr_code>": self.addr_code_index,
            "<addr_data>": self.addr_data_index,
            "<addr_tgt>": self.addr_tgt_index,
            "<addr_unknown>": self.addr_unknown_index,
            "<seq>": self.seq_index,
        }
        
        # Build itos (index to string) mapping
        self.itos = {idx: token for token, idx in self.stoi.items()}
        
        self.vocab_size = len(self.stoi)
    
    def __len__(self):
        return self.vocab_size
    
    def to_index(self, token: str) -> int:
        """Convert token to index"""
        return self.stoi.get(token, self.unk_index)
    
    def to_token(self, index: int) -> str:
        """Convert index to token"""
        return self.itos.get(index, "<unk>")
    
    def is_address_token(self, token: str) -> bool:
        """Check if a token is an address token"""
        return token.startswith("<addr_") or token == "<seq>"
    
    def get_palmtree_replacement(self, token: str) -> str:
        """
        Get the replacement token to use in PalmTree vocab
        For address tokens, return a generic placeholder
        """
        if self.is_address_token(token):
            return "<unk>"  # Or could use a special [ADDR] token if in PalmTree vocab
        return token


def create_addr_vocab():
    """Create and return address vocabulary"""
    return AddrVocab()


if __name__ == "__main__":
    # Test
    vocab = create_addr_vocab()
    print(f"Address Vocabulary Size: {len(vocab)}")
    print("\nTokens:")
    for idx in range(len(vocab)):
        print(f"  {idx}: {vocab.to_token(idx)}")
    
    print("\nTest conversions:")
    test_tokens = ["<addr_start>", "<addr_code>", "<seq>", "some_random_token"]
    for token in test_tokens:
        idx = vocab.to_index(token)
        print(f"  '{token}' -> {idx} -> '{vocab.to_token(idx)}'")
