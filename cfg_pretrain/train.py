"""
CFG Pretraining with 3-Level Embeddings
Building on PalmTree's pre-trained semantic embeddings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pickle
from tqdm import tqdm
import json

from data_loader import CFGPretrainDataset
from config import (
    VOCAB_FILE, BB_PAIRS_FILE, MAX_SEQ_LEN, EMBED_DIM,
    NUM_HEADS, NUM_LAYERS, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, NUM_EPOCHS, NUM_WORKERS, SAVE_EVERY,
    MLM_PROBABILITY, TASK_WEIGHTS
)


class SinCosPositionEncoding(nn.Module):
    """
    Sin/Cos position encoding for continuous position values
    Uses multiple frequencies to capture patterns at different scales
    """
    def __init__(self, embed_dim, base=10000.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        
        # Create frequency bands
        # Lower indices = lower frequencies (long-range patterns)
        # Higher indices = higher frequencies (short-range patterns)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-torch.log(torch.tensor(base)) / embed_dim)
        )
        self.register_buffer('div_term', div_term)
        
    def forward(self, positions, scale=1.0):
        """
        Args:
            positions: [batch, seq_len] - Position values (can be negative)
            scale: Scaling factor for position values (default: 1.0)
        
        Returns:
            encodings: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len = positions.size()
        device = positions.device
        
        # Initialize encoding
        pe = torch.zeros(batch_size, seq_len, self.embed_dim, device=device)
        
        # Scale positions
        scaled_pos = positions.unsqueeze(-1) * scale  # [batch, seq_len, 1]
        
        # Apply sin/cos with different frequencies
        # Each pair (2i, 2i+1) uses the same frequency but different functions
        for i in range(len(self.div_term)):
            pe[:, :, 2*i] = torch.sin(scaled_pos.squeeze(-1) * self.div_term[i])
            pe[:, :, 2*i+1] = torch.cos(scaled_pos.squeeze(-1) * self.div_term[i])
        
        return pe


class ThreeLevelEmbedding(nn.Module):
    """
    3-Level Embedding Architecture:
    Level 1: Semantic (from PalmTree's pre-trained embeddings)
    Level 2a: Binary Position (global, full 128-dim encoding)
    Level 2b: Function Position (local, full 128-dim encoding)
    Level 3: Sequence Position (standard positional encoding)
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len, palmtree_embeddings=None):
        super().__init__()
        
        # Level 1: Semantic embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize with PalmTree's pre-trained embeddings if provided
        if palmtree_embeddings is not None:
            print(f"Loading PalmTree embeddings: {palmtree_embeddings.shape}")
            original_vocab_size = palmtree_embeddings.shape[0]
            
            # Copy PalmTree's embeddings for original tokens
            with torch.no_grad():
                self.token_embedding.weight[:original_vocab_size] = palmtree_embeddings
            
            # Initialize new address tokens (6631-6634) 
            # Use random initialization or average of similar tokens
            print(f"Initializing {vocab_size - original_vocab_size} new address token embeddings")
            # Keep default random initialization for now
        
        # Level 2: Two separate position encoders for hierarchical address positions
        # Binary position: global location in binary (0-1), uses lower base frequency
        self.binary_position_encoder = SinCosPositionEncoding(embed_dim, base=10000.0)
        
        # Function position: local offset within function (can be negative), uses higher base frequency
        self.function_position_encoder = SinCosPositionEncoding(embed_dim, base=1000.0)
        
        # Level 3: Sequence position embeddings
        self.sequence_position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, token_ids, binary_positions, function_positions, sequence_positions):
        """
        Args:
            token_ids: [batch, seq_len] - Token IDs from vocabulary
            binary_positions: [batch, seq_len] - Binary-normalized positions (0-1) for addr tokens, 0 for others
            function_positions: [batch, seq_len] - Function-normalized positions for addr tokens, 0 for others
            sequence_positions: [batch, seq_len] - Sequential position indices
        
        Returns:
            embeddings: [batch, seq_len, embed_dim]
        """
        # Level 1: Semantic embeddings
        semantic_emb = self.token_embedding(token_ids)  # [batch, seq_len, embed_dim]
        
        # Level 2a: Binary position encoding (global, full 128 dims)
        # Scale binary positions [0, 1] to [0, 2π] for full rotation coverage
        binary_emb = self.binary_position_encoder(binary_positions, scale=2*torch.pi)
        
        # Level 2b: Function position encoding (local, full 128 dims)
        # Scale function positions (can be negative) with π for appropriate coverage
        function_emb = self.function_position_encoder(function_positions, scale=torch.pi)
        
        # Level 3: Sequence position embeddings
        seq_pos_emb = self.sequence_position_embedding(sequence_positions)  # [batch, seq_len, embed_dim]
        
        # Combine all levels (direct addition, like original Transformer)
        # Each level contributes full 128-dimensional information
        combined = semantic_emb + binary_emb + function_emb + seq_pos_emb
        
        # Normalize and dropout
        embeddings = self.dropout(self.layer_norm(combined))
        
        return embeddings


class CFGPretrainModel(nn.Module):
    """
    CFG Pretraining Model with Multiple Tasks
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, palmtree_embeddings=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # 3-Level embedding
        self.embedding = ThreeLevelEmbedding(vocab_size, embed_dim, max_seq_len, palmtree_embeddings)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Task-specific heads
        
        # Task 1: Masked Language Modeling (MLM)
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        
        # Task 2: Control Flow Prediction (binary classification: reachable/not reachable)
        self.cfg_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # Concat source & target representations
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1)
        )
        
        # Task 3: Address Type Classification (for masked address tokens)
        # 4 types: addr_start, addr_end, addr_code, addr_data
        self.addr_type_head = nn.Linear(embed_dim, 4)
        
        # Initialize prediction heads with smaller weights to prevent gradient explosion
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with smaller scale to prevent gradient explosion"""
        # MLM head
        nn.init.normal_(self.mlm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlm_head.bias)
        
        # CFG head
        for module in self.cfg_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
        
        # Address type head
        nn.init.normal_(self.addr_type_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.addr_type_head.bias)
        
    def forward(self, token_ids, binary_positions, function_positions, sequence_positions, attention_mask=None):
        """
        Forward pass through embedding + transformer
        
        Returns:
            hidden_states: [batch, seq_len, embed_dim]
        """
        # Get 3-level embeddings
        embeddings = self.embedding(token_ids, binary_positions, function_positions, sequence_positions)
        
        # Create attention mask for padding
        if attention_mask is not None:
            # Convert to transformer format: 1 for tokens to mask, 0 for valid tokens
            attention_mask = (attention_mask == 0).float()
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
        
        # Transformer encoding
        hidden_states = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        return hidden_states
    
    def predict_mlm(self, hidden_states):
        """Predict masked tokens"""
        return self.mlm_head(hidden_states)
    
    def predict_cfg(self, source_repr, target_repr):
        """Predict if target is reachable from source"""
        combined = torch.cat([source_repr, target_repr], dim=-1)
        return self.cfg_head(combined).squeeze(-1)
    
    def predict_addr_type(self, hidden_states):
        """Predict address token type"""
        return self.addr_type_head(hidden_states)


def load_palmtree_embeddings(model_path):
    """
    Load PalmTree's pre-trained token embeddings from BERT model
    PalmTree uses BERT architecture with embedding.token layer
    """
    try:
        # Load PalmTree BERT model (weights_only=False to allow unpickling custom classes)
        print(f"Loading PalmTree model from: {model_path}")
        palmtree_model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # PalmTree model is a BERT instance with embedding.token.weight
        if hasattr(palmtree_model, 'embedding') and hasattr(palmtree_model.embedding, 'token'):
            embeddings = palmtree_model.embedding.token.weight.data
            print(f"✓ Loaded PalmTree token embeddings: {embeddings.shape}")
            print(f"  Vocabulary size: {embeddings.shape[0]}")
            print(f"  Embedding dimension: {embeddings.shape[1]}")
            return embeddings
        else:
            print("Warning: Could not find embedding.token in PalmTree model")
            print(f"Model attributes: {dir(palmtree_model)}")
            return None
            
    except Exception as e:
        print(f"Error loading PalmTree model: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_epoch(model, dataloader, optimizer, device, epoch, vocab, palmtree_model=None):
    """
    Train for one epoch with multiple tasks
    Focus on address tokens and control flow, not general semantics
    Added contrastive loss to keep embeddings close to PalmTree
    """
    model.train()
    total_loss = 0
    mlm_loss_sum = 0
    cfg_loss_sum = 0
    addr_loss_sum = 0
    contrastive_loss_sum = 0
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore non-masked tokens
    cfg_criterion = nn.BCEWithLogitsLoss()
    addr_criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore non-address tokens
    contrastive_criterion = nn.CosineEmbeddingLoss()  # Keep embeddings close to PalmTree
    
    # Get address token IDs from vocab
    addr_token_ids = {
        'addr_start': vocab.stoi.get('addr_start', -1),
        'addr_end': vocab.stoi.get('addr_end', -1),
        'addr_code': vocab.stoi.get('addr_code', -1),
        'addr_data': vocab.stoi.get('addr_data', -1),
    }
    addr_id_to_class = {v: i for i, (k, v) in enumerate(addr_token_ids.items())}  # Map token_id -> class index
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        token_ids = batch['input_ids'].to(device)
        binary_positions = batch['binary_positions'].to(device)
        function_positions = batch['function_positions'].to(device)
        sequence_positions = batch['sequence_positions'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        mlm_labels = batch['mlm_labels'].to(device)
        cfg_label = batch['cfg_label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        hidden_states = model(token_ids, binary_positions, function_positions, sequence_positions, attention_mask)
        
        # Task 1: MLM Loss (focus on address and control flow tokens)
        mlm_logits = model.predict_mlm(hidden_states)
        mlm_logits = mlm_logits.view(-1, model.vocab_size)
        mlm_labels_flat = mlm_labels.view(-1)
        
        # Clamp logits to prevent overflow before softmax
        mlm_logits = torch.clamp(mlm_logits, min=-100, max=100)
        mlm_loss = mlm_criterion(mlm_logits, mlm_labels_flat)
        
        # Check for NaN in MLM loss
        if torch.isnan(mlm_loss):
            print(f"\nWARNING: NaN in MLM loss, skipping batch")
            mlm_loss = torch.tensor(0.0, device=device)
        
        # Task 2: CFG Reachability Loss
        # Use mean pooling over non-padding tokens for BB representation
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        bb_repr = sum_embeddings / sum_mask  # [batch, embed_dim]
        
        # Split into source and target (assuming batch is organized this way)
        batch_size = bb_repr.size(0)
        if batch_size % 2 == 0:
            source_repr = bb_repr[:batch_size//2]
            target_repr = bb_repr[batch_size//2:]
            cfg_labels = cfg_label[:batch_size//2]
            
            cfg_logits = model.predict_cfg(source_repr, target_repr)
            cfg_loss = cfg_criterion(cfg_logits, cfg_labels.float())
        else:
            cfg_loss = torch.tensor(0.0, device=device)
        
        # Task 3: Address Type Prediction (4-class classification for masked address tokens)
        # Create labels: for each masked position, if it's an address token, set the class (0-3), else -100
        addr_labels = torch.full_like(mlm_labels, -100)  # [batch, seq_len], default ignore
        for addr_id, class_idx in addr_id_to_class.items():
            addr_labels[mlm_labels == addr_id] = class_idx  # Set class index for address tokens
        
        addr_logits = model.predict_addr_type(hidden_states)  # [batch, seq_len, 4]
        addr_logits = addr_logits.view(-1, 4)  # [batch*seq_len, 4]
        addr_labels_flat = addr_labels.view(-1)  # [batch*seq_len]
        
        # Clamp addr logits to prevent overflow
        addr_logits = torch.clamp(addr_logits, min=-100, max=100)
        addr_loss = addr_criterion(addr_logits, addr_labels_flat)
        
        # Check for NaN in address loss
        if torch.isnan(addr_loss):
            print(f"\nWARNING: NaN in Address loss, skipping batch")
            addr_loss = torch.tensor(0.0, device=device)
        
        # Task 4: Contrastive Loss (keep NON-ADDRESS embeddings close to PalmTree's semantic space)
        # Address tokens are new and should learn freely; only preserve semantics for existing tokens
        contrastive_loss = torch.tensor(0.0, device=device)
        if palmtree_model is not None:
            # Clip token IDs to PalmTree's vocab range for embedding lookup
            # Address tokens (6632-6635) will be clipped to 6630 (last valid PalmTree token)
            token_ids_clipped = torch.clamp(token_ids, 0, 6630)
            
            with torch.no_grad():
                # Get PalmTree's embeddings using clipped tokens
                palmtree_embeddings = palmtree_model.embedding.token(token_ids_clipped)  # [batch, seq_len, embed_dim]
            
            # Get our enhanced embeddings (just the token embedding part, not full hidden states)
            our_token_embeddings = model.embedding.token_embedding(token_ids)  # [batch, seq_len, embed_dim]
            
            # Calculate contrastive loss: encourage similarity FOR NON-ADDRESS TOKENS ONLY
            # Reshape for contrastive loss
            batch_size, seq_len, embed_dim = our_token_embeddings.shape
            our_flat = our_token_embeddings.view(-1, embed_dim)  # [batch*seq_len, embed_dim]
            palmtree_flat = palmtree_embeddings.view(-1, embed_dim)  # [batch*seq_len, embed_dim]
            token_ids_flat = token_ids.view(-1)  # [batch*seq_len]
            
            # Create mask: 1 for non-padding AND non-address tokens
            attention_mask_flat = attention_mask.view(-1)  # [batch*seq_len]
            
            # Identify address token IDs
            addr_token_id_set = set(addr_token_ids.values())
            
            # Create mask: True for valid non-address tokens
            non_addr_mask = attention_mask_flat.bool()  # Start with non-padding
            for addr_id in addr_token_id_set:
                if addr_id >= 0:  # Valid address token ID
                    non_addr_mask = non_addr_mask & (token_ids_flat != addr_id)
            
            # Only compute contrastive loss on non-address tokens
            if non_addr_mask.sum() > 0:
                our_non_addr = our_flat[non_addr_mask]
                palmtree_non_addr = palmtree_flat[non_addr_mask]
                target = torch.ones(our_non_addr.size(0), device=device)  # All 1s = similar
                contrastive_loss = contrastive_criterion(our_non_addr, palmtree_non_addr, target)
        
        # Combined loss with task weights
        loss = (TASK_WEIGHTS['mlm'] * mlm_loss + 
                TASK_WEIGHTS['cfg_prediction'] * cfg_loss + 
                TASK_WEIGHTS['addr_prediction'] * addr_loss +
                TASK_WEIGHTS['contrastive'] * contrastive_loss)
        
        # Check for NaN in combined loss
        if torch.isnan(loss):
            print(f"\nWARNING: NaN in combined loss, skipping batch")
            print(f"  MLM: {mlm_loss.item():.4f}, CFG: {cfg_loss.item():.4f}, Addr: {addr_loss.item():.4f}, Contra: {contrastive_loss.item():.4f}")
            continue
        
        # Backward pass
        loss.backward()
        
        # More aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        mlm_loss_sum += mlm_loss.item()
        if isinstance(cfg_loss, torch.Tensor):
            cfg_loss_sum += cfg_loss.item()
        addr_loss_sum += addr_loss.item()
        contrastive_loss_sum += contrastive_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mlm': f'{mlm_loss.item():.4f}',
            'cfg': f'{cfg_loss.item():.4f}' if isinstance(cfg_loss, torch.Tensor) else '0.0',
            'addr': f'{addr_loss.item():.4f}',
            'contra': f'{contrastive_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_mlm = mlm_loss_sum / len(dataloader)
    avg_cfg = cfg_loss_sum / len(dataloader)
    avg_addr = addr_loss_sum / len(dataloader)
    avg_contra = contrastive_loss_sum / len(dataloader)
    
    return avg_loss, avg_mlm, avg_cfg, avg_addr, avg_contra


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary>
    vocab_path = os.path.join(os.path.dirname(__file__), VOCAB_FILE)
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load PalmTree's pre-trained embeddings
    palmtree_model_path = os.path.join(os.path.dirname(__file__), "..", "pre-trained_model", "palmtree", "transformer.ep19")
    palmtree_embeddings = None
    palmtree_model = None
    
    if os.path.exists(palmtree_model_path):
        print(f"\n{'='*80}")
        print("Loading PalmTree Pre-trained Model")
        print(f"{'='*80}")
        palmtree_embeddings = load_palmtree_embeddings(palmtree_model_path)
        
        # Also load the full PalmTree model for contrastive learning
        print("Loading full PalmTree model for contrastive loss...")
        palmtree_model = torch.load(palmtree_model_path, map_location=device, weights_only=False)
        palmtree_model.eval()  # Set to eval mode, we only use it for reference
        for param in palmtree_model.parameters():
            param.requires_grad = False  # Freeze PalmTree model
        print("✓ PalmTree model loaded and frozen")
    else:
        print(f"Warning: PalmTree model not found at {palmtree_model_path}")
        print("Training from scratch without pre-trained embeddings")
        print("⚠ Contrastive loss will be disabled")
    
    # Create dataset and split into train/val
    data_file = os.path.join(os.path.dirname(__file__), "..", BB_PAIRS_FILE)
    print(f"Loading training data from: {data_file}")
    full_dataset = CFGPretrainDataset(
        data_file=data_file,
        vocab_file=vocab_path,
        max_seq_length=MAX_SEQ_LEN,
        mlm_probability=MLM_PROBABILITY
    )
    # Split: 90% train, 10% val
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    def validate_epoch(model, dataloader, device, vocab, palmtree_model=None):
        """
        Validate for one epoch with all 4 tasks (including contrastive)
        """
        model.eval()
        total_loss = 0
        mlm_loss_sum = 0
        cfg_loss_sum = 0
        addr_loss_sum = 0
        contrastive_loss_sum = 0
        addr_loss_sum = 0
        
        mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        cfg_criterion = nn.BCEWithLogitsLoss()
        addr_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        contrastive_criterion = nn.CosineEmbeddingLoss()
        
        # Get address token IDs from vocab
        addr_token_ids = {
            'addr_start': vocab.stoi.get('addr_start', -1),
            'addr_end': vocab.stoi.get('addr_end', -1),
            'addr_code': vocab.stoi.get('addr_code', -1),
            'addr_data': vocab.stoi.get('addr_data', -1),
        }
        addr_id_to_class = {v: i for i, (k, v) in enumerate(addr_token_ids.items())}
        
        with torch.no_grad():
            for batch in dataloader:
                token_ids = batch['input_ids'].to(device)
                binary_positions = batch['binary_positions'].to(device)
                function_positions = batch['function_positions'].to(device)
                sequence_positions = batch['sequence_positions'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                mlm_labels = batch['mlm_labels'].to(device)
                cfg_label = batch['cfg_label'].to(device)
                
                hidden_states = model(token_ids, binary_positions, function_positions, sequence_positions, attention_mask)
                
                # Task 1: MLM Loss
                mlm_logits = model.predict_mlm(hidden_states)
                mlm_logits = mlm_logits.view(-1, model.vocab_size)
                mlm_labels_flat = mlm_labels.view(-1)
                mlm_loss = mlm_criterion(mlm_logits, mlm_labels_flat)
                
                # Task 2: CFG Loss
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                bb_repr = sum_embeddings / sum_mask
                
                batch_size = bb_repr.size(0)
                if batch_size % 2 == 0:
                    source_repr = bb_repr[:batch_size//2]
                    target_repr = bb_repr[batch_size//2:]
                    cfg_labels = cfg_label[:batch_size//2]
                    cfg_logits = model.predict_cfg(source_repr, target_repr)
                    cfg_loss = cfg_criterion(cfg_logits, cfg_labels.float())
                else:
                    cfg_loss = torch.tensor(0.0, device=device)
                
                # Task 3: Address Type Prediction
                addr_labels = torch.full_like(mlm_labels, -100)
                for addr_id, class_idx in addr_id_to_class.items():
                    addr_labels[mlm_labels == addr_id] = class_idx
                
                addr_logits = model.predict_addr_type(hidden_states)
                addr_logits = addr_logits.view(-1, 4)
                addr_labels_flat = addr_labels.view(-1)
                addr_loss = addr_criterion(addr_logits, addr_labels_flat)
                
                # Task 4: Contrastive Loss (only for non-address tokens)
                contrastive_loss = torch.tensor(0.0, device=device)
                if palmtree_model is not None:
                    # Clip token IDs to PalmTree's vocab range
                    token_ids_clipped = torch.clamp(token_ids, 0, 6630)
                    palmtree_embeddings_batch = palmtree_model.embedding.token(token_ids_clipped)
                    our_token_embeddings = model.embedding.token_embedding(token_ids)
                    
                    batch_size, seq_len, embed_dim = our_token_embeddings.shape
                    our_flat = our_token_embeddings.view(-1, embed_dim)
                    palmtree_flat = palmtree_embeddings_batch.view(-1, embed_dim)
                    token_ids_flat = token_ids.view(-1)
                    
                    attention_mask_flat = attention_mask.view(-1)
                    
                    # Identify address token IDs and create mask for non-address tokens
                    addr_token_id_set = set(addr_token_ids.values())
                    non_addr_mask = attention_mask_flat.bool()
                    for addr_id in addr_token_id_set:
                        if addr_id >= 0:
                            non_addr_mask = non_addr_mask & (token_ids_flat != addr_id)
                    
                    if non_addr_mask.sum() > 0:
                        our_non_addr = our_flat[non_addr_mask]
                        palmtree_non_addr = palmtree_flat[non_addr_mask]
                        target = torch.ones(our_non_addr.size(0), device=device)
                        contrastive_loss = contrastive_criterion(our_non_addr, palmtree_non_addr, target)
                
                # Combined loss
                loss = (TASK_WEIGHTS['mlm'] * mlm_loss + 
                        TASK_WEIGHTS['cfg_prediction'] * cfg_loss + 
                        TASK_WEIGHTS['addr_prediction'] * addr_loss +
                        TASK_WEIGHTS['contrastive'] * contrastive_loss)
                
                total_loss += loss.item()
                mlm_loss_sum += mlm_loss.item()
                if isinstance(cfg_loss, torch.Tensor):
                    cfg_loss_sum += cfg_loss.item()
                addr_loss_sum += addr_loss.item()
                contrastive_loss_sum += contrastive_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_mlm = mlm_loss_sum / len(dataloader)
        avg_cfg = cfg_loss_sum / len(dataloader)
        avg_addr = addr_loss_sum / len(dataloader)
        avg_contra = contrastive_loss_sum / len(dataloader)
        
        return avg_loss, avg_mlm, avg_cfg, avg_addr, avg_contra
    
    # Create model
    model = CFGPretrainModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        palmtree_embeddings=palmtree_embeddings
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Embedding dim: {EMBED_DIM}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Max seq len: {MAX_SEQ_LEN}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    best_loss = float('inf')
    
    best_val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        avg_loss, avg_mlm, avg_cfg, avg_addr, avg_contra = train_epoch(
            model, train_loader, optimizer, device, epoch, vocab, palmtree_model
        )
        
        # Validate
        val_loss, val_mlm, val_cfg, val_addr, val_contra = validate_epoch(
            model, val_loader, device, vocab, palmtree_model
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {avg_loss:.4f} | MLM: {avg_mlm:.4f} | CFG: {avg_cfg:.4f} | Addr: {avg_addr:.4f} | Contra: {avg_contra:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | MLM: {val_mlm:.4f} | CFG: {val_cfg:.4f} | Addr: {val_addr:.4f} | Contra: {val_contra:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'mlm_loss': avg_mlm,
            'cfg_loss': avg_cfg,
        }
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", f"cfg_pretrain_epoch_{epoch}.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(os.path.dirname(__file__), "checkpoints", "cfg_pretrain_best.pth")
            torch.save(checkpoint, best_path)
            print(f"  Best model updated: {best_path}")
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Quick test: print cfg_label distribution for a few batches using atilibusb.so_bb_pairs.txt
    import sys
    if '--test-cfg-labels' in sys.argv:
        data_file = os.path.join(os.path.dirname(__file__), "..", "bb_pairs_output", "atilibusb.so_bb_pairs.txt")
        vocab_path = os.path.join(os.path.dirname(__file__), VOCAB_FILE)
        dataset = CFGPretrainDataset(
            data_file=data_file,
            vocab_file=vocab_path,
            max_seq_length=MAX_SEQ_LEN,
            mlm_probability=MLM_PROBABILITY
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        print("Testing negative pair generation (cfg_label distribution):")
        from collections import Counter
        label_counter = Counter()
        for i, batch in enumerate(dataloader):
            labels = batch['cfg_label'].tolist()
            label_counter.update(labels)
            print(f"Batch {i+1}: cfg_label counts: {Counter(labels)}")
            if i >= 4:
                break
        print(f"Total cfg_label counts in 5 batches: {label_counter}")
        sys.exit(0)
    main()
