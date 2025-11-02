"""
Configuration for Basic Block Pretraining
"""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PALMTREE_ROOT = os.path.dirname(PROJECT_ROOT)
PRETRAINED_MODEL_PATH = os.path.join(PALMTREE_ROOT, "pre-trained_model")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Model Configuration
PALMTREE_VOCAB_SIZE = 20000  # Original PalmTree vocab size
MAX_SEQ_LENGTH = 128  # Max tokens in a BB pair (reduced from 512 for memory)
HIDDEN_SIZE = 128  # PalmTree hidden dimension (from pre-trained model)
NUM_LAYERS = 12  # PalmTree transformer layers
NUM_HEADS = 12  # PalmTree attention heads
INTERMEDIATE_SIZE = 512  # 4 * HIDDEN_SIZE

# Address Encoding Configuration
ADDRESS_ENCODING_DIM = 64  # Dimension for sin/cos address encoding (reduced to fit with 128 hidden size)
ADDRESS_FREQ_BASE = 10000  # Base for sinusoidal encoding (similar to positional encoding)

# Special Address Tokens (add to vocabulary)
SPECIAL_ADDR_TOKENS = [
    "<addr_start>",
    "<addr_end>",
    "<addr_tgt>",
    "<addr_code>",
    "<addr_data>",
    "<addr_unknown>",
    "<seq>",  # Separator between instructions
]

# Training Configuration
BATCH_SIZE = 8  # Reduced from 32 for memory
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Task Weights (multi-task learning)
TASK_WEIGHTS = {
    "addr_type": 1.0,         # Masked address type prediction (PRIMARY)
    "edge_type": 0.8,         # Edge type classification
    "addr_distance": 0.5,     # Address distance prediction (spatial relationships)
    "next_bb": 0.3,           # Next BB prediction (optional, lower weight)
}

# Data Configuration
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
MIN_BB_LENGTH = 1  # Minimum instructions in a BB
MAX_BB_LENGTH = 100  # Maximum instructions in a BB

# Address Type Labels
ADDR_TYPE_LABELS = {
    "code": 0,
    "data": 1,
    "start": 2,
    "end": 3,
    "tgt": 4,
    "unknown": 5,
}

# Edge Type Labels
EDGE_TYPE_LABELS = {
    "fallthrough": 0,
    "branch": 1,
    "call": 2,
    "return": 3,
    "indirect": 4,
}

# Address Distance Labels (for learning spatial relationships)
ADDR_DISTANCE_LABELS = {
    "same_bb": 0,      # Within same BB (addr_start to addr_end)
    "near": 1,         # < 1KB distance
    "medium": 2,       # 1KB - 64KB distance  
    "far": 3,          # > 64KB distance
}

# Masking probability for address type prediction
ADDR_MASK_PROB = 0.15  # 15% of addresses will be masked during training

# Logging
LOG_INTERVAL = 100
SAVE_INTERVAL = 1000
EVAL_INTERVAL = 500
