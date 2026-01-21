# jTrans_instr Complete Pipeline

## Complete Steps (Run in Order)

### Step 1: Generate Pickle Files and JSON
```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
bash generate_baseline.sh
```

### Step 2: Generate Text File
```bash
bash generate_text.sh
```

### Step 3: Build Vocabulary
```bash
bash build_vocab.sh
```

## Output Files
- `/data/kun/jtrans_instr/instr_pretrain.txt` - Pretraining data
- `/home/kun/Document/AAE/extern/jTrans_instr/jtrans_tokenizer/vocab.txt` - Vocabulary

## Done!
