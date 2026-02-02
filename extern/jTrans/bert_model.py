"""
Address-Aware BERT Model for jTrans
Self-contained implementation without external dependencies
"""

import torch
import torch.nn as nn
from transformer_components import TransformerBlock, MaskedLanguageModel, NextSentencePrediction
from pretrain.address_aware.address_embedding import AddressAwareBERTEmbedding


class AddressAwareBERT(nn.Module):
    """
    Address-aware BERT for binary code understanding.
    
    Uses hierarchical address positions (binary, function, basic block)
    in addition to standard token embeddings.
    """
    
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, max_len=512, 
                 use_address_embedding=True, use_var_embedding=True):
        """
        Args:
            vocab_size: Size of the vocabulary
            hidden: Hidden size / embedding dimension
            n_layers: Number of transformer layers
            attn_heads: Number of attention heads
            dropout: Dropout rate
            max_len: Maximum sequence length
            use_address_embedding: Whether to use address-aware positional embeddings
            use_var_embedding: Whether to use var offset embeddings for var(0xXX) tokens
        """
        super().__init__()
        
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding
        
        # Address-aware embedding layer
        self.embedding = AddressAwareBERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            dropout=dropout,
            max_len=max_len,
            use_address_embedding=use_address_embedding,
            use_var_embedding=use_var_embedding
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len]
            segment_labels: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len]
            function_pos: [batch_size, seq_len]
            bb_pos: [batch_size, seq_len]
            var_offsets: [batch_size, seq_len] (optional, for var positional embedding)
            
        Returns:
            output: [batch_size, seq_len, hidden]
        """
        # Create attention mask (mask out padding tokens)
        mask = (token_ids > 0).unsqueeze(1).repeat(1, token_ids.size(1), 1).unsqueeze(1)
        
        # Get embeddings with address and var awareness
        x = self.embedding(token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x


class AddressAwareBERTForMLM(nn.Module):
    """
    Address-aware BERT with MLM and JTP heads for pretraining (jTrans style).
    Keeps MLM + JTP tasks for compatibility with existing jTrans pretrain.
    """
    
    def __init__(self, bert_model, vocab_size, max_len=512):
        """
        Args:
            bert_model: AddressAwareBERT model
            vocab_size: Vocabulary size
            max_len: Maximum sequence length (for JTP)
        """
        super().__init__()
        
        self.bert = bert_model
        self.vocab_size = vocab_size
        self.hidden = bert_model.hidden
        self.max_len = max_len
        
        # MLM head (predicts masked tokens)
        self.mlm_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, vocab_size)
        )
        
        # JTP head (predicts jump target positions)
        self.jtp_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, max_len)
        )
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Forward pass for MLM + JTP.
        
        Returns:
            mlm_logits: [batch_size, seq_len, vocab_size]
            jtp_logits: [batch_size, seq_len, max_len]
        """
        # Get BERT output
        sequence_output = self.bert(token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets)
        
        # MLM predictions
        mlm_logits = self.mlm_head(sequence_output)
        
        # JTP predictions
        jtp_logits = self.jtp_head(sequence_output)
        
        return mlm_logits, jtp_logits


class FunctionSimilarityModel(nn.Module):
    """
    Function Similarity model using contrastive learning (strupos style).
    
    Architecture:
    1. Pre-trained AddressAwareBERT encoder
    2. Function embedding layer (mean pooling over chunks)
    3. Similarity computation (cosine similarity)
    """
    
    def __init__(
        self,
        vocab_size,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        max_len=512,
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
        
        # Freeze BERT if requested
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
        
        Args:
            token_ids: [batch_size, seq_len]
            segment_labels: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len]
            function_pos: [batch_size, seq_len]
            bb_pos: [batch_size, seq_len]
            var_offsets: [batch_size, seq_len]
            num_instructions: [batch_size] or list - Number of instructions per sample
            instruction_boundaries: List of lists - Start positions of each instruction
            
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
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings
        
        # Process each sample in batch with chunking
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
                )
                
                # Mean pool the chunk
                chunk_mask = (chunk_token_ids != 0).float().unsqueeze(-1)
                chunk_emb = (chunk_bert_output * chunk_mask).sum(dim=1) / (chunk_mask.sum(dim=1) + 1e-9)
                chunk_embeddings.append(chunk_emb.squeeze(0))
            
            # Mean pool all chunk embeddings
            if len(chunk_embeddings) > 0:
                all_chunks = torch.stack(chunk_embeddings)
                func_emb = all_chunks.mean(dim=0)
            else:
                func_emb = torch.zeros(self.bert.hidden, device=token_ids.device)
            
            function_embeddings.append(func_emb)
        
        # Stack into batch
        pooled_output = torch.stack(function_embeddings)
        
        # Project to embedding dimension
        embeddings = self.projection(pooled_output)
        
        # L2 normalize for cosine similarity
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def load_pretrained_bert(self, checkpoint_path):
        """
        Load pre-trained BERT weights from checkpoint.
        Auto-detects if checkpoint has address/var embeddings.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dict with detection results: {'has_address': bool, 'has_var': bool}
        """
        import torch
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'bert_state_dict' in checkpoint:
                state_dict = checkpoint['bert_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Extract BERT weights (remove 'bert.' prefix if present)
                bert_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('bert.'):
                        bert_state_dict[k[5:]] = v
                    elif not k.startswith('mlm_head') and not k.startswith('jtp_head'):
                        bert_state_dict[k] = v
                state_dict = bert_state_dict
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()
        
        # Detect if checkpoint has address/var embeddings
        has_address = any('address_position' in k or 'address_embedding' in k for k in state_dict.keys())
        has_var = any('var_position' in k or 'var_embedding' in k for k in state_dict.keys())
        
        print(f"[INFO] Checkpoint detection:")
        print(f"  - Has address embeddings: {has_address}")
        print(f"  - Has var embeddings: {has_var}")
        
        # Load weights
        missing_keys, unexpected_keys = self.bert.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"[INFO] Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[INFO] Unexpected keys: {len(unexpected_keys)}")
        
        print("[INFO] Pre-trained BERT loaded successfully")
        
        return {'has_address': has_address, 'has_var': has_var}


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for function similarity.
    Pulls together positive pairs, pushes apart negative pairs.
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
        Args:
            emb1: [batch_size, embedding_dim]
            emb2: [batch_size, embedding_dim]
            labels: [batch_size] - 1 for positive pairs, 0 for negative pairs
            
        Returns:
            loss: scalar
        """
        if self.metric == 'cosine':
            # Cosine similarity (embeddings are already normalized)
            similarity = torch.sum(emb1 * emb2, dim=1)
            
            # Positive pairs: maximize similarity
            positive_loss = (1 - similarity) * labels.float()
            
            # Negative pairs: push apart if similarity > (1 - margin)
            negative_loss = torch.clamp(similarity - (1 - self.margin), min=0) * (1 - labels.float())
            
            loss = (positive_loss + negative_loss).mean()
        else:
            # Euclidean distance
            distance = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1))
            
            # Positive pairs: minimize distance
            positive_loss = distance * labels.float()
            
            # Negative pairs: maximize distance (up to margin)
            negative_loss = torch.clamp(self.margin - distance, min=0) * (1 - labels.float())
            
            loss = (positive_loss + negative_loss).mean()
        
        return loss
