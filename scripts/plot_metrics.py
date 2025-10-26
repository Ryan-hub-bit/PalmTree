#!/usr/bin/env python3
import os
import re
import json
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import csv

# ----------------------------
# Config defaults
# ----------------------------
DEFAULT_BASE_DIR = "/home/louie/PalmTree/evaluation_results"
MODELS = ["address_aware", "vanilla"]
METRICS = [
    "total_loss",
    "mlm_loss",
    "perplexity",
    "dfg_nsp_loss",
    "cfg_nsp_loss",
    "dfg_nsp_acc",
    "cfg_nsp_acc",
]

EPOCH_RE = re.compile(r"epoch_(\d+)\.json$", re.IGNORECASE)


def find_epoch_files(folder: str):
    """
    Return a list of (epoch_num:int, filepath:str) for files like epoch_00.json ... epoch_19.json,
    sorted by numeric epoch.
    """
    if not os.path.isdir(folder):
        return []

    out = []
    for name in os.listdir(folder):
        m = EPOCH_RE.match(name)
        if m:
            epoch = int(m.group(1))
            out.append((epoch, os.path.join(folder, name)))
    out.sort(key=lambda x: x[0])
    return out


def load_model_metrics(model_dir: str):
    """
    Read all epoch json files and return:
      epochs: [int, ...]
      metrics: {metric_name: [values aligned with epochs]}
    """
    files = find_epoch_files(model_dir)
    print(f"[INFO] Scanning {model_dir}")
    if not files:
        print("  -> No epoch_XX.json files found.")
        return [], {m: [] for m in METRICS}

    print("  -> Found files:")
    print("     " + " → ".join([os.path.basename(p) for _, p in files]))

    epochs = []
    metric_values = {m: [] for m in METRICS}

    for ep, path in files:
        with open(path, "r") as f:
            j = json.load(f)

        # Prefer epoch from filename; fallback to JSON if absent
        epoch_from_json = j.get("epoch", ep)
        epochs.append(int(epoch_from_json))

        for m in METRICS:
            if m not in j:
                raise KeyError(f"Metric '{m}' missing in {path}")
            metric_values[m].append(j[m])

    # Ensure epochs (x-axis) are strictly increasing and aligned
    paired = list(zip(epochs, *[metric_values[m] for m in METRICS]))
    paired.sort(key=lambda row: row[0])
    epochs_sorted = [row[0] for row in paired]
    metrics_sorted = {m: [row[i+1] for row in paired] for i, m in enumerate(METRICS)}

    return epochs_sorted, metrics_sorted


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_metric(metric: str, data, out_path: str):
    """
    Plot a single metric across epochs for both models on one figure.
    (One chart per metric; no subplots.)
    """
    plt.figure(figsize=(8, 5))
    for model_name, content in data.items():
        epochs = content["epochs"]
        values = content["metrics"][metric]
        if not epochs:
            continue
        # Default matplotlib colors/styles; do not set explicit colors
        plt.plot(epochs, values, marker="o", label=model_name)

    plt.title(f"{metric} vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_combined_csv(csv_path: str, model_data: dict):
    """
    Write a long-form CSV with columns:
      model, epoch, total_loss, mlm_loss, perplexity, dfg_nsp_loss, cfg_nsp_loss, dfg_nsp_acc, cfg_nsp_acc
    """
    rows = []
    for model_name, content in model_data.items():
        epochs = content["epochs"]
        for idx, ep in enumerate(epochs):
            row = {"model": model_name, "epoch": ep}
            for m in METRICS:
                row[m] = content["metrics"][m][idx]
            rows.append(row)

    rows.sort(key=lambda r: (r["model"], r["epoch"]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "epoch"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics across epochs for two models."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base directory containing model folders (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=MODELS,
        help=f"Model folder names under base_dir (default: {MODELS})",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    models = args.models

    # Load all data
    model_data = {}
    for model in models:
        model_dir = os.path.join(base_dir, model)
        epochs, metrics_dict = load_model_metrics(model_dir)
        model_data[model] = {"epochs": epochs, "metrics": metrics_dict}

    # Output dirs
    out_dir = os.path.join(base_dir, "comparison_plots")
    ensure_out_dir(out_dir)

    # Plot each metric
    for metric in METRICS:
        out_path = os.path.join(out_dir, f"{metric}_comparison.png")
        plot_metric(metric, model_data, out_path)
        print(f"[OK] Saved {metric} plot -> {out_path}")

    # Also export a combined CSV
    csv_path = os.path.join(base_dir, "metrics_combined.csv")
    write_combined_csv(csv_path, model_data)
    print(f"[OK] Wrote combined CSV -> {csv_path}")

    print("\nDone. Check the 'comparison_plots' folder for images.")


if __name__ == "__main__":
    main()
