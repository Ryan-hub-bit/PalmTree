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
    MLM_PROBABILITY
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


def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train for one epoch with multiple tasks
    """
    model.train()
    total_loss = 0
    mlm_loss_sum = 0
    cfg_loss_sum = 0
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    cfg_criterion = nn.BCEWithLogitsLoss()
    
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
        
        # Task 1: MLM Loss
        mlm_logits = model.predict_mlm(hidden_states)
        mlm_logits = mlm_logits.view(-1, model.vocab_size)
        mlm_labels_flat = mlm_labels.view(-1)
        mlm_loss = mlm_criterion(mlm_logits, mlm_labels_flat)
        
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
        
        # Combined loss
        loss = mlm_loss + 0.5 * cfg_loss  # Weight CFG task less
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        mlm_loss_sum += mlm_loss.item()
        if isinstance(cfg_loss, torch.Tensor):
            cfg_loss_sum += cfg_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mlm': f'{mlm_loss.item():.4f}',
            'cfg': f'{cfg_loss.item():.4f}' if isinstance(cfg_loss, torch.Tensor) else '0.0'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_mlm = mlm_loss_sum / len(dataloader)
    avg_cfg = cfg_loss_sum / len(dataloader)
    
    return avg_loss, avg_mlm, avg_cfg


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab_path = os.path.join(os.path.dirname(__file__), VOCAB_FILE)
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load PalmTree's pre-trained embeddings
    palmtree_model_path = os.path.join(os.path.dirname(__file__), "..", "pre-trained_model", "palmtree", "transformer.ep19")
    palmtree_embeddings = None
    
    if os.path.exists(palmtree_model_path):
        print(f"\n{'='*80}")
        print("Loading PalmTree Pre-trained Model")
        print(f"{'='*80}")
        palmtree_embeddings = load_palmtree_embeddings(palmtree_model_path)
    else:
        print(f"Warning: PalmTree model not found at {palmtree_model_path}")
        print("Training from scratch without pre-trained embeddings")
    
    # Create dataset and dataloader
    data_file = os.path.join(os.path.dirname(__file__), "..", BB_PAIRS_FILE)
    print(f"Loading training data from: {data_file}")
    
    dataset = CFGPretrainDataset(
        data_file=data_file,
        vocab_file=vocab_path,
        max_seq_length=MAX_SEQ_LEN,
        mlm_probability=MLM_PROBABILITY
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)} BB pairs")
    print(f"Number of batches: {len(dataloader)}")
    
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
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        avg_loss, avg_mlm, avg_cfg = train_epoch(model, dataloader, optimizer, device, epoch)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  MLM Loss: {avg_mlm:.4f}")
        print(f"  CFG Loss: {avg_cfg:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        if epoch % SAVE_EVERY == 0 or avg_loss < best_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'mlm_loss': avg_mlm,
                'cfg_loss': avg_cfg,
            }
            
            checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", f"cfg_pretrain_epoch_{epoch}.pth")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(os.path.dirname(__file__), "checkpoints", "cfg_pretrain_best.pth")
                torch.save(checkpoint, best_path)
                print(f"  Best model updated: {best_path}")
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
