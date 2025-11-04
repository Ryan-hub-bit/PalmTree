# Palmtree top-level probe

This small probe package helps you check whether a Palmtree checkpoint encodes
function-relative positions (or any scalar attribute) linearly inside basic-block
embeddings. The package is placed at repository root so it's alongside
`cfg_pretrain/` for convenience.

Quick example (precomputed embeddings):

1. Prepare two numpy files:

   - `inputs.npy` shape (N, D) — the basic-block embeddings (e.g. 10000 x 128)
   - `positions.npy` shape (N,) — the scalar position per block (values in [0,1])

2. Run the linear probe:

```bash
python probe/probe_linear_relationship.py \
  --embeddings /path/to/inputs.npy \
  --positions /path/to/positions.npy \
  --epochs 200 --lr 0.01
```

Notes:

- The loader `probe/load_checkpoint.py` can load full-model checkpoints saved with
  `torch.save(model, path)` and the repo's pickled vocab files.
- If your checkpoint is a `state_dict` only, reconstruct the model using
  `palmtree.model.bert.BERT(...)` and `load_state_dict` before extracting embeddings.

If you want, I can also add a helper that runs a small test dataset through the
model to produce `inputs.npy` automatically — tell me which checkpoint (name)
and which test data to use and I'll implement and run it.
