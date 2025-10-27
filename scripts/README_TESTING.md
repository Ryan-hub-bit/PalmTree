# Testing Both Models (Vanilla vs Semantic)

## Overview
This script tests both vanilla (raw assembly) and semantic (LD/ST/CP) models across all training epochs (0-19).

## Usage

```bash
cd /home/louie/PalmTree/scripts
python3 test_both_models.py
```

## What it does

1. **Vanilla Test**: Tests models from `data/baseoutput/transformer.ep{0-19}`
   - Uses: `data/test/cfg_2.txt` and `data/test/dfg_2.txt`
   - Vocab: `data/baseoutput/vocab`
   - Results: `evaluation_results/vanilla/`

2. **Semantic Test**: Tests models from `data/movoutput/transformer.ep{0-19}`
   - Uses: `data/test/cfg_2_semantic.txt` and `data/test/dfg_2_semantic.txt`
   - Vocab: `data/movoutput/vocab`
   - Results: `evaluation_results/semantic/`

## Output Structure

```
evaluation_results/
├── vanilla/
│   ├── epoch_00.json
│   ├── epoch_01.json
│   ├── ...
│   ├── epoch_19.json
│   └── all_metrics.json
└── semantic/
    ├── epoch_00.json
    ├── epoch_01.json
    ├── ...
    ├── epoch_19.json
    └── all_metrics.json
```

## Metrics Tracked

- **MLM Loss**: Masked Language Model loss
- **Perplexity**: exp(MLM loss)
- **DFG NSP**: Data Flow Graph Next Sentence Prediction
- **CFG NSP**: Control Flow Graph Next Sentence Prediction
- **Accuracies**: DFG and CFG prediction accuracies

## Requirements

- Test data must exist:
  - `data/test/cfg_2.txt` and `data/test/dfg_2.txt`
  - `data/test/cfg_2_semantic.txt` and `data/test/dfg_2_semantic.txt`
- Model checkpoints must exist:
  - `data/baseoutput/transformer.ep{0-19}`
  - `data/movoutput/transformer.ep{0-19}`
- Vocabulary files must exist at the respective locations
