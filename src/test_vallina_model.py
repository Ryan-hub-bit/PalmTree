#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch

# ---- Global patch: force legacy torch.load behavior everywhere ----
print("[init] Patching torch.load to default to weights_only=False + map_location=cpu")
_real_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    kwargs.setdefault("map_location", "cpu")
    return _real_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# -------------------------------------------------------------------

import torch.nn as nn
from torch.utils.data import DataLoader

import bert_pytorch
from palmtree.model import BERTLM2
from palmtree.dataset.dataset2 import BERTDataset2
from palmtree.dataset import WordVocab


# =========================
# Vocab utilities
# =========================
def load_vocab(vocab_path):
    """
    Load a WordVocab. Prefer WordVocab.load_vocab; fall back to torch.load and validate.
    """
    # First try the official loader
    try:
        vocab = WordVocab.load_vocab(vocab_path)
        print(f"[vocab] Loaded WordVocab via WordVocab.load_vocab: {vocab_path}")
        return vocab
    except Exception as e:
        print(f"[vocab] WordVocab.load_vocab failed: {e} — trying torch.load")

    # Fallback to torch.load (old saves)
    obj = torch.load(vocab_path)
    # If it's already a WordVocab, good
    if isinstance(obj, WordVocab):
        print(f"[vocab] Loaded WordVocab via torch.load: {vocab_path}")
        return obj

    # If it's a dict with likely fields (very defensive)
    if isinstance(obj, dict) and ("stoi" in obj or "itos" in obj or "freqs" in obj):
        vocab = WordVocab()
        # fill minimally if present
        vocab.stoi = obj.get("stoi", {})
        vocab.itos = obj.get("itos", [])
        vocab.freqs = obj.get("freqs", None)
        # common indices if provided
        for name in ("pad_index", "unk_index", "mask_index", "cls_index", "sep_index"):
            if name in obj:
                setattr(vocab, name, obj[name])
        print(f"[vocab] Reconstructed WordVocab from dict fields in {vocab_path}")
        return vocab

    # If it's an int (your current case), that’s just vocab size — not enough for dataset
    if isinstance(obj, int):
        raise TypeError(
            f"[vocab] Got an int from {vocab_path} (value={obj}). "
            "That is a vocab size, not a WordVocab. Re-save your vocab using WordVocab.save_vocab(path) "
            "and pass that file to --vocab_path."
        )

    # Last resort: clear error
    raise TypeError(
        f"[vocab] Unsupported vocab object type from {vocab_path}: {type(obj)}. "
        "Please provide a WordVocab file."
    )


# =========================
# Data utilities
# =========================
def create_test_dataloader(vocab, test_data_path, batch_size=32, num_workers=4):
    """Create test dataloader for CFG + DFG."""
    test_cfg_path = os.path.join(test_data_path, "cfg_test.txt")
    test_dfg_path = os.path.join(test_data_path, "dfg_test.txt")

    if not os.path.exists(test_cfg_path):
        raise FileNotFoundError(f"Missing file: {test_cfg_path}")
    if not os.path.exists(test_dfg_path):
        raise FileNotFoundError(f"Missing file: {test_dfg_path}")

    dataset = BERTDataset2(
        test_cfg_path,
        test_dfg_path,
        vocab,
        seq_len=20,
        corpus_lines=None,
        on_memory=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )


# =========================
# Evaluation
# =========================
def evaluate_epoch(model, data_loader, device):
    """
    Evaluate one pass over the test set.
    Returns a dict of averaged losses/metrics.
    """
    model.eval()

    total_dfg_next_loss = 0.0
    total_cfg_next_loss = 0.0
    total_mask_loss = 0.0
    total_dfg_correct = 0
    total_cfg_correct = 0
    total_samples = 0

    dfg_next_criterion = nn.NLLLoss()
    cfg_next_criterion = nn.NLLLoss()
    masked_criterion = nn.NLLLoss(ignore_index=0)  # padding id 0

    progress = tqdm(
        enumerate(data_loader),
        desc="Evaluating",
        total=len(data_loader),
        bar_format="{l_bar}{r_bar}",
    )

    with torch.no_grad():
        for i, batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            dfg_next_out, cfg_next_out, mask_lm_out = model(
                batch["dfg_bert_input"],
                batch["dfg_segment_label"],
                batch["cfg_bert_input"],
                batch["cfg_segment_label"],
            )

            # MLM loss (expects [N, C, L]); mask_lm_out is [N, L, C]
            mask_loss = masked_criterion(
                mask_lm_out.transpose(1, 2), batch["dfg_bert_label"]
            )
            dfg_next_loss = dfg_next_criterion(dfg_next_out, batch["dfg_is_next"])
            cfg_next_loss = cfg_next_criterion(cfg_next_out, batch["cfg_is_next"])

            # NSP accuracies
            dfg_pred = torch.argmax(dfg_next_out, dim=1)
            cfg_pred = torch.argmax(cfg_next_out, dim=1)
            dfg_correct = (dfg_pred == batch["dfg_is_next"]).sum().item()
            cfg_correct = (cfg_pred == batch["cfg_is_next"]).sum().item()

            bsz = batch["dfg_bert_input"].size(0)
            total_samples += bsz
            total_mask_loss += mask_loss.item() * bsz
            total_dfg_next_loss += dfg_next_loss.item() * bsz
            total_cfg_next_loss += cfg_next_loss.item() * bsz
            total_dfg_correct += dfg_correct
            total_cfg_correct += cfg_correct

            if (i % 100 == 0) and total_samples > 0:
                progress.set_postfix({
                    "MLM": f"{total_mask_loss/total_samples:.4f}",
                    "DFG Acc": f"{total_dfg_correct/total_samples:.4f}",
                    "CFG Acc": f"{total_cfg_correct/total_samples:.4f}",
                })

    avg_mlm_loss = total_mask_loss / max(1, total_samples)
    avg_dfg_nsp_loss = total_dfg_next_loss / max(1, total_samples)
    avg_cfg_nsp_loss = total_cfg_next_loss / max(1, total_samples)
    dfg_nsp_acc = total_dfg_correct / max(1, total_samples)
    cfg_nsp_acc = total_cfg_correct / max(1, total_samples)
    perplexity = float(np.exp(avg_mlm_loss))
    total_loss = avg_dfg_nsp_loss + avg_cfg_nsp_loss + avg_mlm_loss

    return {
        "total_loss": total_loss,
        "mlm_loss": avg_mlm_loss,
        "perplexity": perplexity,
        "dfg_nsp_loss": avg_dfg_nsp_loss,
        "cfg_nsp_loss": avg_cfg_nsp_loss,
        "dfg_nsp_acc": dfg_nsp_acc,
        "cfg_nsp_acc": cfg_nsp_acc,
    }


# =========================
# Checkpoint loading
# =========================
def _strip_prefixes(state_dict):
    """Remove common prefixes like 'module.' and 'bert.'."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("bert."):
            k = k[len("bert."):]
        out[k] = v
    return out


def build_base_bert(vocab_size):
    """Build the exact BERT2 architecture used during training (adjust if needed)."""
    return bert_pytorch.BERT2(
        vocab_size=vocab_size,
        hidden=128,
        n_layers=12,
        attn_heads=8,
        dropout=0.0,
    )


def load_and_wrap_model(model_path, vocab_size, device):
    """
    Build BERT2, load checkpoint (global patch ensures weights_only=False),
    then wrap with BERTLM. Returns eval() model on the chosen device.
    """
    print(f"Loading model from {model_path}")
    bert = build_base_bert(vocab_size)

    ckpt = torch.load(model_path)

    # Extract state_dict in a tolerant way
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt  # assume it's already a raw state_dict
    else:
        if hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
        else:
            raise RuntimeError("Unrecognized checkpoint format (no state_dict).")

    state_dict = _strip_prefixes(state_dict)
    missing, unexpected = bert.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")

    bert = bert.to(device)
    model = BERTLM2(bert, vocab_size).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    model.eval()
    print("Model loaded successfully.")
    return model


# =========================
# Device
# =========================
def setup_device(use_cuda_flag):
    use_cuda = bool(use_cuda_flag) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using device: {device}")
    return device


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Evaluate BERT (CFG+DFG) on test data")
    parser.add_argument("--model_dir", required=True, help="Directory with checkpoints")
    parser.add_argument("--test_data", required=True, help="Directory with test txts")
    parser.add_argument("--vocab_path", required=True, help="Path to WordVocab file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default="evaluation_results", help="Save dir")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.model_dir):
            raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
        if not os.path.exists(args.test_data):
            raise FileNotFoundError(f"Test data directory not found: {args.test_data}")
        if not os.path.exists(args.vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {args.vocab_path}")

        os.makedirs(args.output_dir, exist_ok=True)

        print("\nStarting evaluation...")
        print(f"Model directory: {args.model_dir}")
        print(f"Test data:      {args.test_data}")
        print(f"Output dir:     {args.output_dir}\n")

        device = setup_device(args.cuda)

        # ---- Load vocab as WordVocab (critical for BERTDataset2) ----
        vocab = load_vocab(args.vocab_path)
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")

        print("Creating test dataloader...")
        test_loader = create_test_dataloader(
            vocab,
            args.test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Accept both transformer.epN and transformer_mlp.epN naming
        ckpt_re = re.compile(r"^(transformer(?:_mlp)?\.ep)(\d+)$")
        checkpoints = []
        for fname in os.listdir(args.model_dir):
            m = ckpt_re.match(fname)
            if m:
                checkpoints.append((fname, int(m.group(2))))
        checkpoints.sort(key=lambda x: x[1])

        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints named transformer.ep* or transformer_mlp.ep* found in {args.model_dir}"
            )

        print(f"Found {len(checkpoints)} checkpoints")

        metrics_accum = {
            "total_loss": [],
            "mlm_loss": [],
            "perplexity": [],
            "dfg_nsp_loss": [],
            "cfg_nsp_loss": [],
            "dfg_nsp_acc": [],
            "cfg_nsp_acc": [],
        }

        progress = tqdm(checkpoints, desc="Evaluating epochs", unit="epoch")
        for fname, epoch in progress:
            progress.set_description(f"Evaluating epoch {epoch:02d}")
            ckpt_path = os.path.join(args.model_dir, fname)
            try:
                model = load_and_wrap_model(ckpt_path, vocab_size, device)
                epoch_metrics = evaluate_epoch(model, test_loader, device)

                # save per-epoch metrics
                out_epoch = {
                    "epoch": epoch,
                    "checkpoint": fname,
                    **epoch_metrics,
                }
                with open(os.path.join(args.output_dir, f"epoch_{epoch:02d}.json"), "w") as f:
                    json.dump(out_epoch, f, indent=2)

                for k, v in epoch_metrics.items():
                    metrics_accum[k].append(v)

                progress.set_postfix({"Loss": f"{epoch_metrics['total_loss']:.4f}"})

            except Exception as e:
                print(f"\n[error] Epoch {epoch} ({fname}) failed: {e}")
                continue

        summary = {
            "metrics": metrics_accum,
            "settings": {
                "model_dir": args.model_dir,
                "test_data": args.test_data,
                "vocab_path": args.vocab_path,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
        }
        all_path = os.path.join(args.output_dir, "all_metrics.json")
        with open(all_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\nEvaluation complete!")
        print(f"Per-epoch results: {args.output_dir}/epoch_XX.json")
        print(f"Aggregated metrics: {all_path}")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
