# Address-Aware Training Tasks Summary

## Task Distribution: CFG vs DFG

### CFG (Control Flow Graph)
Sequences follow **control flow edges** → Instructions are **continuous/nearby in memory**

| Task | Applied? | Purpose |
|------|----------|---------|
| **MLM** | ✅ Yes | Learn instruction semantics |
| **NSP_CFG** | ✅ Yes | Learn control flow order (uses addresses implicitly) |
| **ADP** | ✅ Yes | Learn address distance magnitude (explicit supervision) |
| ~~**AOP**~~ | ❌ **REMOVED** | **Redundant - NSP_CFG already learns order via addresses** |

**Loss**: `mlm_loss + nsp_cfg_loss + 0.5*adp_loss`

### DFG (Data Flow Graph)
Sequences follow **data dependencies** → Instructions are **scattered in memory**

| Task | Applied? | Reason |
|------|----------|--------|
| **MLM** | ❌ No | Not used (PalmTree design) |
| **NSP_DFG** | ✅ Yes | Learn data dependency relationships |
| **ADP** | ❌ No | Distance meaningless (not continuous) |
| ~~**AOP**~~ | ❌ **REMOVED** | **Redundant with NSP** |

**Loss**: `nsp_dfg_loss` only

---

## Why ADP/AOP are CFG-Only

### CFG Example (Continuous):
```
Instruction Sequence (following control flow):
0x401000: endbr64           (binary=0.21, fnorm=0.00, bbnorm=0.00)
0x401004: sub rsp, 8        (binary=0.21, fnorm=0.15, bbnorm=0.22)  ← 4 bytes later
0x401008: mov rax, [rel]    (binary=0.21, fnorm=0.30, bbnorm=0.44)  ← 4 bytes later
0x40100f: test rax, rax     (binary=0.21, fnorm=0.56, bbnorm=0.83)  ← 7 bytes later

Distance patterns:
- fnorm increases monotonically: 0.00 → 0.15 → 0.30 → 0.56
- Instructions are nearby in memory (4-7 bytes apart)
- ADP can learn: "close fnorm values → nearby instructions"
- AOP can learn: "lower binary_pos → comes first in memory"
```

### DFG Example (Scattered):
```
Instruction Sequence (following def-use chains):
0x401000: mov rax, [rbx]    (binary=0.21, fnorm=0.00)  # defines rax
0x401234: add rbx, 8        (binary=0.45, fnorm=0.82)  # uses rbx (564 bytes later!)
0x401008: test rax, rax     (binary=0.21, fnorm=0.30)  # uses rax (back to 0x401008)
0x401450: mov rcx, rax      (binary=0.67, fnorm=0.15)  # uses rax (1104 bytes later!)

Distance patterns:
- binary_pos jumps: 0.21 → 0.45 → 0.21 → 0.67 (non-monotonic)
- fnorm jumps: 0.00 → 0.82 → 0.30 → 0.15 (random order)
- Instructions are scattered (564-1104 bytes apart)
- ADP is meaningless: distance doesn't correlate with dependency
- AOP is meaningless: memory order doesn't match data flow order
```

**Conclusion**: Address distance/order only makes sense when instructions are **spatially correlated** (CFG), not when they're **logically correlated** (DFG).

---

## Complete Training Loop

```python
def train_epoch(model, dataloader, optimizer, device):
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    
    for batch in dataloader:
        # === CFG Forward: MLM + NSP + ADP ===
        cfg_mlm, cfg_nsp, cfg_adp = model(
            batch['cfg_bert_input'].to(device),
            batch['cfg_segment_label'].to(device),
            batch['cfg_binary_pos'].to(device),
            batch['cfg_function_pos'].to(device),
            batch['cfg_bb_pos'].to(device),
            corpus_type='cfg',
            return_address_tasks=True  # Returns 3 outputs for CFG
        )
        
        # === DFG Forward: NSP only ===
        _, dfg_nsp = model(
            batch['dfg_bert_input'].to(device),
            batch['dfg_segment_label'].to(device),
            batch['dfg_binary_pos'].to(device),
            batch['dfg_function_pos'].to(device),
            batch['dfg_bb_pos'].to(device),
            corpus_type='dfg',
            return_address_tasks=False  # Returns 2 outputs for DFG
        )
        
        # === Compute Losses ===
        # CFG losses
        mlm_loss = mlm_criterion(
            cfg_mlm.transpose(1, 2), 
            batch['cfg_bert_label'].to(device)
        )
        nsp_cfg_loss = nsp_criterion(
            cfg_nsp, 
            batch['cfg_is_next'].to(device)
        )
        adp_loss = mse_criterion(
            cfg_adp, 
            batch['cfg_adp_labels'].to(device)  # [binary_dist, fnorm_dist, bbnorm_dist]
        )
        
        # DFG loss
        nsp_dfg_loss = nsp_criterion(
            dfg_nsp, 
            batch['dfg_is_next'].to(device)
        )
        
        # === Combined Loss ===
        total_loss = (
            1.0 * mlm_loss +       # Token prediction (CFG)
            1.0 * nsp_cfg_loss +   # Control flow order (CFG, uses addresses implicitly)
            1.0 * nsp_dfg_loss +   # Data dependency coherence (DFG)
            0.5 * adp_loss         # Address distance magnitude (CFG only, explicit)
        )
        
        # === Backward Pass ===
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

---

## Task Purposes

| Task | Type | CFG | DFG | Purpose |
|------|------|-----|-----|---------|
| **MLM** | Classification (vocab_size) | ✅ | ❌ | Learn token semantics and context |
| **NSP_CFG** | Classification (2) | ✅ | ❌ | Learn "is this the correct execution order?" (uses addresses implicitly) |
| **NSP_DFG** | Classification (2) | ❌ | ✅ | Learn "is this the correct data dependency?" |
| **ADP** | Regression (3) | ✅ | ❌ | Learn "how far apart are these instructions?" (explicit distance magnitude) |
| ~~**AOP**~~ | ~~Classification (2)~~ | ❌ | ❌ | **REMOVED - redundant with NSP_CFG** |

---

## Expected Benefits

### For CFG:
1. **MLM**: Strong instruction-level understanding
2. **NSP_CFG**: Understands control flow semantics (jumps, branches)
3. **ADP**: Understands instruction proximity and locality
4. **AOP**: Understands memory layout and function boundaries

### For DFG:
1. **NSP_DFG**: Understands data dependencies (def-use chains)
2. **No ADP/AOP**: Avoids learning meaningless address patterns

### Combined:
- **Better position encoding**: ADP/AOP explicitly teach model to encode addresses
- **Better probing results**: Linear probes should show R² > 0.5 after training
- **Task-specific understanding**: CFG learns spatial patterns, DFG learns logical patterns
