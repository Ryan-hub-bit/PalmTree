"""
Configuration file.
"""

VOCAB_SIZE = 10000
USE_CUDA = True
DEVICES = [0]
CUDA_DEVICE = DEVICES[0]
VERSION = 1
MAXLEN = 10

# Semantic instruction format options
# "none": use original assembly instructions
# "semantic-only": use only semantic format (LD/ST/CP direction markers)
# "augment": mix 50% original + 50% semantic during training
USE_SEMANTIC = "none"  # Change to "semantic-only" to test semantic format
USE_SEMANTIC = "semantic-only"  # Change to "semantic-only" to test semantic format
AUGMENT_RATIO = 0.5    # Only used when USE_SEMANTIC="augment"
