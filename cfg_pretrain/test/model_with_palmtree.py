#!/usr/bin/env python3
"""
Example: How to use PalmTree embeddings with address tokens

Strategy:
1. Load PalmTree's pre-trained token embeddings
2. Extend embedding matrix for new address tokens
3. Initialize new embeddings (random or from similar tokens)
4. Fine-tune everything together
"""

import torch
import torch.nn as nn

class AddressAwarePalmTree(nn.Module):
    """
    Extends PalmTree with address token support
    """
    
    def __init__(self, palmtree_model, extended_vocab_size, embedding_dim=768):
        super().__init__()
        
        # Store original vocab size
        self.original_vocab_size = palmtree_model.token_embedding.num_embeddings
        self.new_vocab_size = extended_vocab_size
        self.embedding_dim = embedding_dim
        
        print(f"Original vocab size: {self.original_vocab_size}")
        print(f"Extended vocab size: {self.new_vocab_size}")
        print(f"New tokens: {self.new_vocab_size - self.original_vocab_size}")
        
        # Extract PalmTree's embeddings
        original_embeddings = palmtree_model.token_embedding.weight.data
        
        # Create extended embedding matrix
        self.token_embedding = nn.Embedding(self.new_vocab_size, embedding_dim)
        
        # Copy pre-trained weights for existing tokens
        self.token_embedding.weight.data[:self.original_vocab_size] = original_embeddings
        
        # Initialize new address token embeddings
        # Option 1: Random initialization (standard)
        # nn.init.normal_(self.token_embedding.weight.data[self.original_vocab_size:], 
        #                 mean=0.0, std=0.02)
        
        # Option 2: Initialize from similar semantic tokens (RECOMMENDED)
        self._initialize_address_embeddings(palmtree_model)
        
        # Copy other PalmTree components
        self.transformer = palmtree_model.transformer
        # ... copy other layers ...
        
        # NEW: Add address position embeddings (Level 2)
        self.binary_position_embedding = nn.Linear(1, embedding_dim)
        self.function_position_embedding = nn.Linear(1, embedding_dim)
        
        # NEW: Add address-specific prediction heads
        self.addr_type_classifier = nn.Linear(embedding_dim, 4)  # 4 address types
        self.cf_direction_classifier = nn.Linear(embedding_dim, 4)  # 4 directions
        
    def _initialize_address_embeddings(self, palmtree_model):
        """
        Initialize address token embeddings from semantically similar tokens
        
        Strategy:
        - addr_start/addr_end → average of instruction start/end markers (if any)
        - addr_code → average of jump/call instructions (je, jmp, call)
        - addr_data → average of data access patterns (mov with memory)
        """
        
        # Get vocabulary (assuming you have it)
        # vocab = load_vocab()
        
        # Example: Initialize addr_code from jump-related instructions
        # jump_tokens = ['je', 'jne', 'jmp', 'call', 'jz', 'jnz']
        # jump_ids = [vocab.get(t) for t in jump_tokens if vocab.get(t) is not None]
        # 
        # if jump_ids:
        #     addr_code_id = self.original_vocab_size + 2  # Assuming this is addr_code
        #     jump_embeddings = palmtree_model.token_embedding.weight.data[jump_ids]
        #     self.token_embedding.weight.data[addr_code_id] = jump_embeddings.mean(dim=0)
        
        # For now, use random init with smaller std for stability
        nn.init.normal_(
            self.token_embedding.weight.data[self.original_vocab_size:], 
            mean=0.0, 
            std=0.01  # Smaller std for stability
        )
        
        print("✓ Initialized address token embeddings")
    
    def forward(self, input_ids, binary_positions, function_positions, 
                attention_mask=None, segment_ids=None):
        """
        Forward pass with 3-level embeddings
        
        Args:
            input_ids: [batch, seq_len] - Level 1: Semantic (token IDs)
            binary_positions: [batch, seq_len] - Level 2a: Binary position
            function_positions: [batch, seq_len] - Level 2b: Function position
            attention_mask: [batch, seq_len]
            segment_ids: [batch, seq_len]
        
        Returns:
            dict with various predictions
        """
        batch_size, seq_len = input_ids.shape
        
        # Level 1: Semantic embeddings (from PalmTree + new address tokens)
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, dim]
        
        # Level 2: Address position embeddings (ONLY for address tokens)
        binary_embeds = self.binary_position_embedding(
            binary_positions.unsqueeze(-1)
        )  # [batch, seq_len, dim]
        
        function_embeds = self.function_position_embedding(
            function_positions.unsqueeze(-1)
        )  # [batch, seq_len, dim]
        
        # Combine all embeddings
        combined_embeds = token_embeds + binary_embeds + function_embeds
        
        # Level 3: Sequence position (handled by transformer)
        # Add standard positional encoding here if needed
        
        # Pass through transformer
        hidden_states = self.transformer(
            combined_embeds,
            attention_mask=attention_mask,
            # ... other args
        )
        
        # Get pooled representation (e.g., first token or mean pooling)
        pooled_output = hidden_states[:, 0]  # CLS-style pooling
        
        # Predictions
        addr_type_logits = self.addr_type_classifier(pooled_output)
        cf_direction_logits = self.cf_direction_classifier(pooled_output)
        
        return {
            'hidden_states': hidden_states,
            'pooled_output': pooled_output,
            'addr_type_logits': addr_type_logits,
            'cf_direction_logits': cf_direction_logits,
        }


def load_palmtree_and_extend(palmtree_checkpoint: str, extended_vocab_size: int):
    """
    Load PalmTree checkpoint and extend for address tokens
    
    Args:
        palmtree_checkpoint: Path to PalmTree's pre-trained model
        extended_vocab_size: New vocabulary size (original + 4 address tokens)
    
    Returns:
        AddressAwarePalmTree model
    """
    
    # Load PalmTree
    print(f"Loading PalmTree from: {palmtree_checkpoint}")
    palmtree_model = torch.load(palmtree_checkpoint)
    
    # Create extended model
    model = AddressAwarePalmTree(
        palmtree_model=palmtree_model,
        extended_vocab_size=extended_vocab_size,
        embedding_dim=768  # PalmTree's dimension
    )
    
    # Freeze PalmTree's parameters (optional - for initial training)
    # for param in model.transformer.parameters():
    #     param.requires_grad = False
    
    print("✓ Model ready for address-focused training")
    
    return model


# Example usage
if __name__ == "__main__":
    # Assuming PalmTree vocab size is 5000, we add 4 address tokens
    original_vocab_size = 5000
    extended_vocab_size = 5004
    
    # Load and extend
    # model = load_palmtree_and_extend(
    #     palmtree_checkpoint="path/to/palmtree.pth",
    #     extended_vocab_size=extended_vocab_size
    # )
    
    print("\nTraining Strategy:")
    print("=" * 60)
    print("Phase 1: Freeze PalmTree, train only address components")
    print("  - New token embeddings (addr_start, addr_end, addr_code, addr_data)")
    print("  - Address position embeddings (binary_pos, function_pos)")
    print("  - Address prediction heads")
    print()
    print("Phase 2: Fine-tune everything end-to-end")
    print("  - Unfreeze PalmTree")
    print("  - Joint training with smaller learning rate for PalmTree layers")
    print()
