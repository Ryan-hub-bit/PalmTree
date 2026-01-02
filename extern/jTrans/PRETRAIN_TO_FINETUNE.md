# From Pretraining to Fine-tuning

## Overview
After baseline pretraining, you can load the model for fine-tuning on downstream tasks (e.g., function similarity).

## Pretraining Output Structure

After running `bash run_baseline_pretrain.sh`, you'll get:

```
/home/kun/Document/AAE/output/baseline_pretrain/
├── checkpoint_epoch_1/          # Saved every N epochs (--save_every 1)
│   ├── config.json
│   ├── pytorch_model.bin        # Model weights
│   └── training_info.json       # Training metrics
├── checkpoint_epoch_2/
├── ...
├── checkpoint_epoch_10/
└── best_model/                  # Best model based on validation loss
    ├── config.json
    ├── pytorch_model.bin
    └── training_info.json
```

## Loading Pretrained Model for Fine-tuning

### Method 1: Use existing finetune.py

```python
# In finetune.py line 224:
model = BinBertModel.from_pretrained(args.model_path)

# Run with your pretrained model:
python finetune.py \
    --model_path /home/kun/Document/AAE/output/baseline_pretrain/best_model \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/baseline \
    --train_path /path/to/paired_train_data \
    --eval_path /path/to/paired_test_data \
    --output_path ./output/finetune \
    --batch_size 64 \
    --lr 1e-5 \
    --freeze_cnt 10
```

### Method 2: Load in custom script

```python
from transformers import BertModel, BertTokenizer

# Load pretrained model
model_path = "/home/kun/Document/AAE/output/baseline_pretrain/best_model"
model = BertModel.from_pretrained(model_path)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "/home/kun/Document/AAE/extern/jTrans/pretrain/baseline"
)

# Now use model for your task
model.to(device)
model.eval()  # or model.train() for fine-tuning
```

## Key Points

✅ **Saved Model Format**: Hugging Face format (config.json + pytorch_model.bin)
✅ **Compatible**: Can load with `BertModel.from_pretrained()`
✅ **Tokenizer**: Use the same tokenizer from pretraining
✅ **Architecture**: Uses position embeddings = word embeddings (jTrans style)

## Fine-tuning Tasks

The existing `finetune.py` does:
- **Task**: Triplet loss for function similarity
- **Data**: Paired functions across optimizations (O0, O1, O2, O3, Os)
- **Loss**: Cosine similarity triplet loss
- **Freezing**: Can freeze first N layers (`--freeze_cnt`)

## What's Saved During Pretraining

```python
# Line 390 in train_baseline.py:
model.bert.save_pretrained(checkpoint_dir)
```

This saves:
- `config.json`: Model architecture (hidden_size, num_layers, etc.)
- `pytorch_model.bin`: Trained weights
- `training_info.json`: Custom training metrics (MLM loss, JTP loss, etc.)

## Compatibility Note

Your baseline model uses **standard BERT architecture**:
- Vocabulary: 2,902 tokens
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- Max length: 512

The finetune.py expects `BinBertModel` which:
```python
class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        # Key change: position_embeddings = word_embeddings
        self.embeddings.position_embeddings = self.embeddings.word_embeddings
```

Since your baseline model already saves as standard BERT format, you can:
1. Load with `BertModel.from_pretrained()` ✅
2. Wrap in `BinBertModel` if needed for position embedding trick ✅
3. Use directly for any BERT-compatible task ✅
