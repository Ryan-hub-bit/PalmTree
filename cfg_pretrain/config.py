"""
Configuration for CFG-based pretraining
"""

# Data paths
DATA_DIR = "bb_pairs_output"
BB_PAIRS_FILE = "all_bb_pairs.txt"  # Main training data file
VOCAB_FILE = "vocab_extended"  # Extended vocabulary with address tokens

# Model parameters (aligned with PalmTree)
MAX_SEQ_LEN = 512
MAX_SEQ_LENGTH = 512  # Alias for compatibility
EMBED_DIM = 128  # PalmTree's embedding dimension (verified from transformer.ep19)
NUM_HEADS = 8     # Number of attention heads
NUM_LAYERS = 6    # Number of transformer layers
INTERMEDIATE_SIZE = 512  # Feedforward dimension (4x EMBED_DIM)
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_PROBS_DROPOUT_PROB = 0.1

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 4  # DataLoader workers
SAVE_EVERY = 2   # Save checkpoint every N epochs

# MLM parameters
MLM_PROBABILITY = 0.15  # Probability of masking tokens

# Pretraining tasks weights
TASK_WEIGHTS = {
    'mlm': 1.0,           # Masked Language Modeling
    'cfg_prediction': 1.0, # CFG edge prediction (predict successor BB)
    'addr_prediction': 0.5 # Address value prediction
}

# MLM parameters
MLM_PROBABILITY = 0.15
MLM_MASK_TOKEN_PROB = 0.8
MLM_RANDOM_TOKEN_PROB = 0.1
MLM_UNCHANGED_PROB = 0.1

# Device
DEVICE = "cuda"  # or "cpu"

# Logging
LOG_INTERVAL = 100
SAVE_INTERVAL = 1000
CHECKPOINT_DIR = "checkpoints/cfg_pretrain"
