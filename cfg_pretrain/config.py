"""
Configuration for CFG-based pretraining
"""

# Data paths
DATA_DIR = "bb_pairs_output"
BB_PAIRS_FILE = "all_bb_pairs.txt"  # Main training data file
#BB_PAIRS_FILE = "bb_pairs_output/ircat_bb_pairs.txt"  # Main training data file
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
BATCH_SIZE = 512  # Further reduced to 4 to avoid OOM on busy GPU
LEARNING_RATE = 5e-5  # Reduced from 1e-4 to prevent gradient explosion and NaN
NUM_EPOCHS = 3 
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.5  # Reduced from 1.0 for more aggressive gradient clipping
NUM_WORKERS = 4  # DataLoader workers
SAVE_EVERY = 2   # Save checkpoint every N epochs

# MLM parameters
MLM_PROBABILITY = 0.15  # Probability of masking tokens
MLM_MASK_TOKEN_PROB = 0.8
MLM_RANDOM_TOKEN_PROB = 0.1
MLM_UNCHANGED_PROB = 0.1

# Negative pair sampling probability
NEGATIVE_PAIR_PROB = 0.5  # Probability of sampling a negative pair (CFG label = 0)

# ============================================================================
# Task Configuration - Enable/Disable different pretraining objectives
# ============================================================================
# You can customize which tasks to train by setting True/False:
#
# Example configurations:
#
# 1. All tasks (default):
#    ENABLE_TASKS = {'mlm': True, 'cfg_prediction': True, 'addr_prediction': True, 'contrastive': True}
#
# 2. Only MLM (standard BERT-style):
#    ENABLE_TASKS = {'mlm': True, 'cfg_prediction': False, 'addr_prediction': False, 'contrastive': False}
#
# 3. MLM + CFG (focus on control flow):
#    ENABLE_TASKS = {'mlm': True, 'cfg_prediction': True, 'addr_prediction': False, 'contrastive': False}
#
# 4. No contrastive (let embeddings drift from PalmTree):
#    ENABLE_TASKS = {'mlm': True, 'cfg_prediction': True, 'addr_prediction': True, 'contrastive': False}
#
ENABLE_TASKS = {
    'mlm': True,            # Masked Language Modeling - predicts masked tokens
    'cfg_prediction': True, # CFG edge prediction - predicts if BB2 follows BB1
    'addr_prediction': True,# Address type classification - classifies address tokens
    'contrastive': False     # Contrastive loss - keeps embeddings close to PalmTree
}

# Task weights (only applied if task is enabled)
TASK_WEIGHTS = {
    'mlm': 1.0,            # Masked Language Modeling
    'cfg_prediction': 1.0, # CFG edge prediction (predict successor BB)
    'addr_prediction': 0.5,# Address value prediction
    'contrastive': 0.3     # Contrastive loss to keep embeddings close to PalmTree
}

# Device
DEVICE = "cuda"  # or "cpu"

# ============================================================================
# Multi-GPU Configuration
# ============================================================================
# Enable multi-GPU training using DataParallel (simple, works on single node)
# Set to True to use all available GPUs, or specify GPU IDs: [0, 1, 2, 3]
USE_MULTI_GPU = True  # Set to True to enable multi-GPU
GPU_IDS = None  # None = use all available GPUs, or specify list like [0, 1, 2]

# Logging
LOG_INTERVAL = 100
SAVE_INTERVAL = 1000
CHECKPOINT_DIR = "checkpoints/cfg_pretrain"



