"""Top-level probe utilities for testing embeddings from Palmtree checkpoints.

Placed at the repository root so it can be used alongside `cfg_pretrain/`.
This package provides a small loader and a linear probing script to test whether
basic-block embeddings contain linear information about function-relative
position.
"""

__all__ = ["load_checkpoint", "probe_linear_relationship"]
