"""Linear probe script for testing whether embeddings encode function-relative position.

Place this at repo root next to `cfg_pretrain/` so it's easy to run from the project
root. Example usage (precomputed embeddings):

    python probe/probe_linear_relationship.py \
        --embeddings /path/to/inputs.npy \
        --positions /path/to/positions.npy

For a quick start, save your h_BB (10000 x 128) to `inputs.npy` and p_func (10000,) to `positions.npy`
and run in precomputed mode.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from typing import Tuple


def train_linear_probe(h: np.ndarray, p: np.ndarray, device: torch.device, epochs=200, lr=1e-2, verbose=True) -> Tuple[nn.Module, dict]:
    """Train a linear probe (nn.Linear) to predict p from h.

    Args:
        h: numpy array of shape (N, D)
        p: numpy array of shape (N,)
    Returns:
        trained nn.Module and metrics dict
    """
    X = torch.from_numpy(h).float().to(device)
    y = torch.from_numpy(p).float().unsqueeze(1).to(device)

    model = nn.Linear(X.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                pred_np = pred.cpu().numpy().reshape(-1)
                r2 = r2_score(p, pred_np)
                mae = mean_absolute_error(p, pred_np)
                pr, _ = pearsonr(p, pred_np)
                print(f"epoch {epoch:4d}: loss={loss.item():.6f} r2={r2:.4f} mae={mae:.6f} pearson={pr:.4f}")

    # Final metrics
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy().reshape(-1)

    metrics = {
        "r2": float(r2_score(p, pred)),
        "mae": float(mean_absolute_error(p, pred)),
        "pearson": float(pearsonr(p, pred)[0])
    }

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Linear probe for function-relative position encoded in embeddings")
    parser.add_argument("--embeddings", type=str, required=True, help=".npy file with embeddings (N x D)")
    parser.add_argument("--positions", type=str, required=True, help=".npy file with positions (N,)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("Loading embeddings and positions...")
    h = np.load(args.embeddings)
    p = np.load(args.positions)

    assert h.ndim == 2, "embeddings must be a 2D array"
    assert p.ndim == 1, "positions must be a 1D array"
    assert h.shape[0] == p.shape[0], "number of embeddings and positions must match"

    print(f"Loaded {h.shape[0]} embeddings of dim {h.shape[1]}")

    # Normalize positions to [0,1] if not already
    if p.min() < 0 or p.max() > 1:
        print("Normalizing positions to [0,1]")
        p = (p - p.min()) / (p.max() - p.min())

    probe, metrics = train_linear_probe(h, p, device, epochs=args.epochs, lr=args.lr)

    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Save probe weights for inspection
    import os
    out_dir = os.path.dirname(args.embeddings) or "."
    torch.save(probe.state_dict(), os.path.join(out_dir, "linear_probe.pth"))
    print(f"Saved probe weights to {os.path.join(out_dir, 'linear_probe.pth')}")


if __name__ == "__main__":
    main()
