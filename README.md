# 🌴 PalmTree — assembly language model

PalmTree is a pretrained language model for assembly instructions (instruction embedding). The implementation and training code live in this repository. Current platform support: x86.

**Pretrained model:** [Google Drive link](https://drive.google.com/file/d/1yC3M-kVTFWql6hCgM_QCbKtc1PbdVdvp/view?usp=sharing)

If you use PalmTree in your research, please consider citing:

Xuezixiang Li, Yu Qu, and Heng Yin, "PalmTree: Learning an Assembly Language Model for Instruction Embedding", CCS 2021.

## BibTeX

```bibtex
@inproceedings{li2021palmtree,
  title={Palmtree: learning an assembly language model for instruction embedding},
  author={Li, Xuezixiang and Qu, Yu and Yin, Heng},
  booktitle={Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security},
  pages={3236--3251},
  year={2021}
}
```

## Quick overview

- Purpose: train and evaluate language-style models over assembly to produce instruction embeddings useful for downstream analyses (outlier detection, matching, etc.).
- Location: training and data utilities are under `src/` and `data/`. Pretrained artifacts are in `pre-trained_model/` and `model/`.

## Requirements

- CUDA (tested with CUDA 10.1+)
- PyTorch (>= 1.3.1)
- Binary Ninja (optional) — used only for dataset generation via its Python API

> Note: exact package versions and Python requirements are not pinned in a requirements file. Create a virtual environment and install PyTorch according to your CUDA version. If you want, I can add a minimal `requirements.txt`.

## Quick start

1. Create an environment and install PyTorch (follow official instructions for your CUDA version).
2. Install other dependencies you need for data generation (Binary Ninja) or evaluation.
3. Training: the training entrypoint is `src/train_palmtree.py` — adjust configuration in `src/config.py` or the `config.py` in `pre-trained_model/`.
4. Evaluation: utility scripts and evaluation helpers are under `pre-trained_model/` (see `eval_utils.py` and `how2use.py`).

Example (conceptual):

```bash
# from repository root
python src/train_palmtree.py   # uses settings in src/config.py
python pre-trained_model/eval_utils.py   # run evaluation helpers
```

If you'd like, I can add runnable examples with exact flags for training and evaluation.

## Dataset generation

This project supports generating datasets from binary code. Two dataset flavors are used:

- Control-flow graphs (CFGs)
- Data-flow graphs (DFGs)

Generator scripts (examples mentioned in project): `dataflow_gen.py` and `control_flow_gen.py` (these use Binary Ninja). There is also a `data/` folder with prepared training splits and a `scripts/` helper folder (for combining files).

Binary Ninja is optional — if you don't have it you can still use the included preprocessed datasets in `data/`.

### Dataset format

Datasets are plain text (TXT). Graph-based samples are converted into instruction sequences by random walks and then split into instruction pairs.

Example: given the instruction sequence

```asm
push rbp
mov rbp, rsp
sub rsp, 0x20
```

the generator produces instruction pairs (tab-separated) such as:

```text
push rbp\tmov rbp rsp
mov rbp rsp\tsub rsp 0x20
```

Note: tokens are simplified (commas removed, spacing normalized) and pairs are separated by a single tab character (`\\t`).

## Evaluations

This repo includes code for both intrinsic and extrinsic evaluations.

Intrinsic examples:

- Opcode outlier detection
- Operand outlier detection
- Basic-block matching

Extrinsic examples:

- Gemini
- EKALVYA

See `pre-trained_model/` and the `extrinsic_evaluation/` folder under `src/` for the evaluation code.

## Acknowledgements

This implementation builds on [bert-pytorch](https://github.com/codertimo/BERT-pytorch). We extended the codebase to support CFG/DFG training tasks and graph-derived data.

## TODO / Roadmap

- Support additional binary analysis backends (Ghidra, IDA Pro)
- Add a Dockerfile and reproducible environment
- Provide step-by-step examples for training/evaluation with flags

## Contributing & contact

If you'd like to contribute, please open an issue or a pull request. If you want me to add more examples (CLI flags, Dockerfile, `requirements.txt`, or runnable notebooks), tell me which you'd prefer and I can add them.

## Repository layout (key files)

The following list highlights important files and folders you may want to inspect or run:

- `src/` — main source code
  - `src/train_palmtree.py` — training entrypoint
  - `src/config.py` — training / runtime configuration
  - `src/data_generator/` — scripts to generate datasets (e.g. `control_flow_gen.py`, `dataflow_gen.py`)
  - `src/palmtree/` — model implementation and dataset utilities
    - `src/palmtree/model/` — model definitions (transformer, BERT-style embeddings)
    - `src/palmtree/dataset/` — dataset and vocab code

- `pre-trained_model/` — pretrained artifacts and evaluation helpers
  - `pre-trained_model/eval_utils.py`
  - `pre-trained_model/how2use.py`
  - `pre-trained_model/config.py`

- `model/` — saved model checkpoints and vocabulary files (e.g. `transformer.ep19`, `vocab`)

- `data/` — prepared training data and per-project splits

- `scripts/` — helper scripts (e.g. `scripts/combine_cfg_files.py`)

- `src/extrinsic_evaluation/` — extrinsic evaluation suites
  - `src/extrinsic_evaluation/gemini/` — Gemini evaluation code
  - `src/extrinsic_evaluation/EKLAVYA/` — EKLAVYA evaluation code and data

If you'd like, I can expand any of these bullets into runnable examples (exact command lines, expected outputs, or minimal configs).
