"""
Function Similarity Model for Fine-tuning

This model takes a pre-trained AddressAwareBERT and fine-tunes it for 
function similarity tasks using contrastive learning.

Uses the function blocks and ground truth pairs from funcsim dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add strupos to path to import models
strupos_path = os.path.join(os.path.dirname(__file__), '..', '..', 'strupos')
if strupos_path not in sys.path:
    sys.path.insert(0, strupos_path)

# Import from strupos using absolute import to avoid circular import
import importlib.util
spec = importlib.util.spec_from_file_location("strupos_model", os.path.join(strupos_path, "model.py"))
strupos_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strupos_model)
AddressAwareBERT = strupos_model.AddressAwareBERT


class FunctionSimilarityModel(nn.Module):
    """
    Function Similarity model using contrastive learning.
    
    Architecture:
    1. Pre-trained AddressAwareBERT encoder
    2. Function embedding layer (mean pooling over sequence)
    3. Similarity computation (cosine similarity or euclidean distance)
    
    Training:
    - Positive pairs: Same function at different optimization levels
    - Negative pairs: Different functions
    - Loss: Contrastive loss or triplet loss
    """
    
    def __init__(
        self,
        vocab_size,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        max_len=60,
        use_address_embedding=True,
        use_var_embedding=True,
        embedding_dim=256,
        freeze_bert=False
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            hidden: BERT hidden dimension
            n_layers: Number of BERT layers
            attn_heads: Number of attention heads
            dropout: Dropout rate
            max_len: Maximum sequence length
            use_address_embedding: Use address-aware embeddings
            use_var_embedding: Use variable offset embeddings
            embedding_dim: Dimension of function embedding (output)
            freeze_bert: Whether to freeze BERT weights during fine-tuning
        """
        super().__init__()
        
        # BERT encoder
        self.bert = AddressAwareBERT(
            vocab_size=vocab_size,
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            max_len=max_len,
            use_address_embedding=use_address_embedding,
            use_var_embedding=use_var_embedding
        )
        
        # Freeze BERT if requested (useful for initial training)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Projection layer to function embedding space
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim)
        )
        
        # L2 normalize embeddings for cosine similarity
        self.normalize = True
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets=None, 
                num_instructions=None, instruction_boundaries=None):
        """
        Forward pass with chunked processing.
        
        Format: <sos> instr1 instr2 instr3 ... <eos> (no separators between instructions)
        
        Process:
        1. Split function into chunks of 8 instructions
        2. For each chunk (max 60 tokens): get BERT embedding and mean pool
        3. Mean pool all chunk embeddings -> function embedding
        
        Args:
            token_ids: [batch_size, seq_len]
            segment_labels: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len]
            function_pos: [batch_size, seq_len]
            bb_pos: [batch_size, seq_len]
            var_offsets: [batch_size, seq_len] - Variable offsets for var(0xXX) tokens (optional)
            num_instructions: [batch_size] or int - Number of instructions per sample (optional)
            instruction_boundaries: List of lists - Start positions of each instruction (optional)
            
        Returns:
            embeddings: [batch_size, embedding_dim] - Function embeddings
        """
        batch_size = token_ids.size(0)
        chunk_size = 8  # Instructions per chunk
        max_chunk_len = 60  # Max tokens per chunk
        
        # If no instruction info provided, fall back to simple pooling
        if num_instructions is None or instruction_boundaries is None:
            bert_output = self.bert(token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets)
            pad_mask = (token_ids != 0).float().unsqueeze(-1)
            pooled = (bert_output * pad_mask).sum(dim=1) / (pad_mask.sum(dim=1) + 1e-9)
            embeddings = self.projection(pooled)
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
        
        # Process each sample in batch
        function_embeddings = []
        
        for b in range(batch_size):
            # Get instruction info for this sample
            n_instr = num_instructions[b] if isinstance(num_instructions, (list, torch.Tensor)) else num_instructions
            boundaries = instruction_boundaries[b] if isinstance(instruction_boundaries, list) else instruction_boundaries
            
            # Split instructions into chunks of 8
            chunk_embeddings = []
            
            for chunk_idx in range(0, n_instr, chunk_size):
                chunk_end_idx = min(chunk_idx + chunk_size, n_instr)
                
                # Get token range for this chunk
                start_token_pos = boundaries[chunk_idx]
                # End position: start of next chunk or end of sequence
                if chunk_end_idx < n_instr:
                    end_token_pos = boundaries[chunk_end_idx]
                else:
                    # Last chunk: find <eos> position
                    eos_pos = (token_ids[b] == 2).nonzero(as_tuple=True)[0]
                    end_token_pos = eos_pos[0].item() + 1 if len(eos_pos) > 0 else token_ids.size(1)
                
                # Extract chunk tokens (limit to max_chunk_len)
                chunk_len = min(end_token_pos - start_token_pos, max_chunk_len)
                chunk_token_ids = token_ids[b, start_token_pos:start_token_pos+chunk_len].unsqueeze(0)
                chunk_segments = segment_labels[b, start_token_pos:start_token_pos+chunk_len].unsqueeze(0)
                chunk_binary_pos = binary_pos[b, start_token_pos:start_token_pos+chunk_len].unsqueeze(0)
                chunk_function_pos = function_pos[b, start_token_pos:start_token_pos+chunk_len].unsqueeze(0)
                chunk_bb_pos = bb_pos[b, start_token_pos:start_token_pos+chunk_len].unsqueeze(0)
                chunk_var_offsets = var_offsets[b, start_token_pos:start_token_pos+chunk_len].unsqueeze(0) if var_offsets is not None else None
                
                # Get BERT embedding for this chunk
                chunk_bert_output = self.bert(
                    chunk_token_ids, chunk_segments,
                    chunk_binary_pos, chunk_function_pos, chunk_bb_pos,
                    chunk_var_offsets
                )  # [1, chunk_len, hidden]
                
                # Mean pool the chunk
                chunk_mask = (chunk_token_ids != 0).float().unsqueeze(-1)  # [1, chunk_len, 1]
                chunk_emb = (chunk_bert_output * chunk_mask).sum(dim=1) / (chunk_mask.sum(dim=1) + 1e-9)  # [1, hidden]
                chunk_embeddings.append(chunk_emb.squeeze(0))  # [hidden]
            
            # Mean pool all chunk embeddings
            if len(chunk_embeddings) > 0:
                all_chunks = torch.stack(chunk_embeddings)  # [num_chunks, hidden]
                func_emb = all_chunks.mean(dim=0)  # [hidden]
            else:
                func_emb = torch.zeros(self.bert.hidden, device=token_ids.device)
            
            function_embeddings.append(func_emb)
        
        # Stack into batch
        pooled_output = torch.stack(function_embeddings)  # [batch_size, hidden]
        
        # Project to embedding dimension
        embeddings = self.projection(pooled_output)  # [batch_size, embedding_dim]
        
        # L2 normalize for cosine similarity
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_similarity(self, emb1, emb2, metric='cosine'):
        """
        Compute similarity between two embeddings.
        
        Args:
            emb1: [batch_size, embedding_dim]
            emb2: [batch_size, embedding_dim]
            metric: 'cosine' or 'euclidean'
            
        Returns:
            similarity: [batch_size] - Similarity scores
        """
        if metric == 'cosine':
            # Cosine similarity (higher is more similar)
            return F.cosine_similarity(emb1, emb2, dim=1)
        elif metric == 'euclidean':
            # Negative euclidean distance (higher is more similar)
            return -torch.norm(emb1 - emb2, p=2, dim=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def load_pretrained_bert(self, checkpoint_path):
        """
        Load pre-trained BERT weights and detect if it has address/var embeddings.
        
        Args:
            checkpoint_path: Path to pre-trained BERT checkpoint
            
        Returns:
            dict with 'has_address' and 'has_var' flags
        """
        print(f"[INFO] Loading pre-trained BERT from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Checkpoint is a dictionary
            if 'bert_state_dict' in checkpoint:
                state_dict = checkpoint['bert_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Extract BERT weights (remove 'bert.' prefix if present)
                bert_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('bert.'):
                        bert_state_dict[k[5:]] = v
                    elif not k.startswith('MLM.') and not k.startswith('NSP.'):
                        bert_state_dict[k] = v
                state_dict = bert_state_dict
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            # Checkpoint is a model object directly
            print(f"[INFO] Checkpoint is a model object (type: {type(checkpoint).__name__})")
            state_dict = checkpoint.state_dict()
        
        # Detect if checkpoint has address/var embeddings
        # Look for specific keys in the embedding module
        has_address = any('address_position' in k or 'address_embedding' in k for k in state_dict.keys())
        has_var = any('var_position' in k or 'var_embedding' in k for k in state_dict.keys())
        
        print(f"[INFO] Checkpoint detection:")
        print(f"  - Has address embeddings: {has_address}")
        print(f"  - Has var embeddings: {has_var}")
        
        # Debug: show some keys to help verify
        embedding_keys = [k for k in state_dict.keys() if 'embedding' in k or 'position' in k]
        if embedding_keys:
            print(f"[DEBUG] Found {len(embedding_keys)} embedding-related keys:")
            for key in embedding_keys[:5]:  # Show first 5
                print(f"    {key}")
            if len(embedding_keys) > 5:
                print(f"    ... and {len(embedding_keys) - 5} more")
        
        # Load weights
        missing_keys, unexpected_keys = self.bert.load_state_dict(state_dict, strict=False)
        
        print("[INFO] Pre-trained BERT loaded successfully")
        
        return {'has_address': has_address, 'has_var': has_var}


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for function similarity.
    
    Pulls together positive pairs (same function, different opt levels)
    Pushes apart negative pairs (different functions)
    """
    
    def __init__(self, margin=1.0, metric='cosine'):
        """
        Args:
            margin: Margin for negative pairs
            metric: 'cosine' or 'euclidean'
        """
        super().__init__()
        self.margin = margin
        self.metric = metric
    
    def forward(self, emb1, emb2, labels):
        """
        Compute contrastive loss.
        
        Args:
            emb1: [batch_size, embedding_dim]
            emb2: [batch_size, embedding_dim]
            labels: [batch_size] - 1 for positive pairs, 0 for negative pairs
            
        Returns:
            loss: Scalar loss value
        """
        if self.metric == 'cosine':
            # Cosine similarity (range: -1 to 1)
            similarity = F.cosine_similarity(emb1, emb2, dim=1)
            # Convert to distance (range: 0 to 2)
            distance = 1 - similarity
        else:
            # Euclidean distance
            distance = torch.norm(emb1 - emb2, p=2, dim=1)
        
        # Contrastive loss
        # Positive pairs: minimize distance
        # Negative pairs: maximize distance (up to margin)
        pos_loss = labels.float() * distance.pow(2)
        neg_loss = (1 - labels.float()) * F.relu(self.margin - distance).pow(2)
        
        loss = (pos_loss + neg_loss).mean()
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for function similarity.
    
    Anchor: Query function
    Positive: Same function at different opt level
    Negative: Different function
    
    Loss encourages: distance(anchor, positive) + margin < distance(anchor, negative)
    """
    
    def __init__(self, margin=1.0, metric='cosine'):
        """
        Args:
            margin: Margin between positive and negative distances
            metric: 'cosine' or 'euclidean'
        """
        super().__init__()
        self.margin = margin
        self.metric = metric
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor: [batch_size, embedding_dim]
            positive: [batch_size, embedding_dim]
            negative: [batch_size, embedding_dim]
            
        Returns:
            loss: Scalar loss value
        """
        if self.metric == 'cosine':
            # Cosine distance (1 - similarity)
            pos_distance = 1 - F.cosine_similarity(anchor, positive, dim=1)
            neg_distance = 1 - F.cosine_similarity(anchor, negative, dim=1)
        else:
            # Euclidean distance
            pos_distance = torch.norm(anchor - positive, p=2, dim=1)
            neg_distance = torch.norm(anchor - negative, p=2, dim=1)
        
        # Triplet loss
        loss = F.relu(pos_distance - neg_distance + self.margin).mean()
        
        return loss
