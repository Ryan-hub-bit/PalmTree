# Address-Focused Training Tasks

## Current Tasks in data_loader.py
1. **MLM (Masked Language Modeling)** - Predict masked tokens
2. **CFG Prediction** - Is target BB a valid successor?
3. **Address Feature Aggregation** - Extract address statistics

## Suggested Address-Specific Tasks

### 1. Address Type Classification
**Goal**: Predict the type of masked address token (addr_start, addr_end, addr_code, addr_data)

**Implementation**:
```python
def _get_address_type_labels(self, token_ids: List[int]) -> List[int]:
    """
    Label each position with address type:
    0 = non-address, 1 = addr_start, 2 = addr_end, 3 = addr_code, 4 = addr_data
    """
    labels = []
    for token_id in token_ids:
        if token_id == self.vocab.get('addr_start'):
            labels.append(1)
        elif token_id == self.vocab.get('addr_end'):
            labels.append(2)
        elif token_id == self.vocab.get('addr_code'):
            labels.append(3)
        elif token_id == self.vocab.get('addr_data'):
            labels.append(4)
        else:
            labels.append(0)
    return labels
```

**Why**: Helps model learn to distinguish different address behaviors

---

### 2. Address Distance Regression
**Goal**: Predict the relative distance between two address tokens

**Implementation**:
```python
def _compute_address_distance_pairs(self, positions: List[Tuple[float, float]], 
                                    token_ids: List[int]) -> List[Dict]:
    """
    For each pair of address tokens, compute their distance in binary space
    Returns: [(idx1, idx2, binary_distance, function_distance), ...]
    """
    addr_indices = [i for i, tid in enumerate(token_ids) 
                   if tid in self.addr_token_ids]
    
    pairs = []
    for i in range(len(addr_indices)):
        for j in range(i+1, len(addr_indices)):
            idx1, idx2 = addr_indices[i], addr_indices[j]
            bin_dist = abs(positions[idx1][0] - positions[idx2][0])
            func_dist = positions[idx1][1] - positions[idx2][1]  # Can be negative
            pairs.append({
                'idx1': idx1, 'idx2': idx2,
                'binary_distance': bin_dist,
                'function_distance': func_dist
            })
    return pairs
```

**Why**: Model learns spatial relationships between addresses

---

### 3. Control Flow Direction Prediction
**Goal**: Predict if target address is forward/backward/external relative to source

**Implementation**:
```python
def _get_control_flow_direction(self, source_addrs: List[Dict], 
                                target_addrs: List[Dict]) -> int:
    """
    0 = same function (local jump)
    1 = forward jump (func_norm > 0)
    2 = backward jump (func_norm < 0)
    3 = external (func_norm outside [-1, 2])
    """
    # Find addr_start in target
    target_start = next((a for a in target_addrs if a['type'] == 'addr_start'), None)
    if target_start is None:
        return 3
    
    func_norm = target_start['function_norm']
    if 0 <= func_norm <= 1:
        return 0  # Same function
    elif func_norm > 1:
        return 1  # Forward
    elif -1 <= func_norm < 0:
        return 2  # Backward
    else:
        return 3  # External
```

**Why**: Learns CFG patterns (loops, calls, local jumps)

---

### 4. Address Reachability Prediction
**Goal**: Given two BBs, predict if there's a direct control flow edge

**Current**: You already have `cfg_label` (always 1 for valid pairs)

**Enhancement**: Add negative samples
```python
def __getitem__(self, idx: int):
    # ... existing code ...
    
    # 50% chance: use actual CFG pair (label=1)
    # 50% chance: use random non-successor BB (label=0)
    if random.random() < 0.5:
        # Use actual pair
        source_bb, target_bb = self.pairs[idx]
        cfg_label = 1
    else:
        # Random non-successor
        source_bb = self.pairs[idx][0]
        target_bb = random.choice([p[1] for p in self.pairs if p[0] != source_bb])
        cfg_label = 0
```

**Why**: Model learns which address patterns indicate valid edges

---

### 5. Address Span Prediction
**Goal**: Predict the size/span of a basic block from addr_start to addr_end

**Implementation**:
```python
def _compute_bb_span(self, addresses: List[Dict]) -> float:
    """
    Compute the span of BB in binary space
    """
    addr_start = next((a for a in addresses if a['type'] == 'addr_start'), None)
    addr_end = next((a for a in addresses if a['type'] == 'addr_end'), None)
    
    if addr_start and addr_end:
        return addr_end['binary_norm'] - addr_start['binary_norm']
    return 0.0
```

**Why**: Learns typical BB sizes and patterns

---

### 6. Next Address Prediction
**Goal**: Given sequence of addresses in BB, predict the next address

**Implementation**:
```python
def _get_next_address_labels(self, token_ids: List[int], 
                             positions: List[Tuple[float, float]]) -> List[Tuple]:
    """
    For each address token, label is the NEXT address token's position
    """
    addr_indices = [i for i, tid in enumerate(token_ids) 
                   if tid in self.addr_token_ids]
    
    labels = [(0.0, 0.0)] * len(token_ids)
    for i in range(len(addr_indices) - 1):
        curr_idx = addr_indices[i]
        next_idx = addr_indices[i + 1]
        labels[curr_idx] = positions[next_idx]
    
    return labels
```

**Why**: Learns sequential address patterns within BBs

---

## Recommended Implementation Priority

### High Priority (Most Impactful)
1. **Mask ONLY address tokens** for MLM (already implemented above)
2. **Control Flow Direction Prediction** - Key for CFG understanding
3. **Address Reachability with negative samples** - Critical for learning valid edges

### Medium Priority  
4. **Address Type Classification** - Helps distinguish addr_start/end/code/data
5. **Address Distance Regression** - Useful for spatial reasoning

### Low Priority (Nice to Have)
6. **Address Span Prediction** - Less critical but can help
7. **Next Address Prediction** - Good for sequential modeling

---

## How to Add These Tasks to data_loader.py

In `__getitem__()` method, add new return fields:

```python
return {
    # Existing fields
    'input_ids': torch.tensor(masked_ids, dtype=torch.long),
    'binary_positions': torch.tensor(binary_positions, dtype=torch.float),
    'function_positions': torch.tensor(function_positions, dtype=torch.float),
    'sequence_positions': torch.tensor(sequence_positions, dtype=torch.long),
    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
    'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
    'cfg_label': torch.tensor(cfg_label, dtype=torch.long),
    
    # NEW: Address-specific tasks
    'addr_type_labels': torch.tensor(addr_type_labels, dtype=torch.long),
    'control_flow_direction': torch.tensor(cf_direction, dtype=torch.long),
    'bb_span': torch.tensor(bb_span, dtype=torch.float),
    'source_addr_features': torch.tensor(source_addr_features, dtype=torch.float),
    'target_addr_features': torch.tensor(target_addr_features, dtype=torch.float),
}
```

---

## Model Architecture Changes Needed

Your model will need additional prediction heads:

```python
class PalmTreeWithAddressTasks(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing embeddings ...
        
        # NEW: Address-specific heads
        self.addr_type_classifier = nn.Linear(hidden_size, 5)  # 5 types
        self.cf_direction_classifier = nn.Linear(hidden_size, 4)  # 4 directions
        self.bb_span_predictor = nn.Linear(hidden_size, 1)  # Regression
        
    def forward(self, ...):
        # ... existing forward pass ...
        
        # NEW: Address task predictions
        addr_type_logits = self.addr_type_classifier(pooled_output)
        cf_direction_logits = self.cf_direction_classifier(pooled_output)
        bb_span_pred = self.bb_span_predictor(pooled_output)
        
        return {
            'mlm_logits': mlm_logits,
            'cfg_logits': cfg_logits,
            'addr_type_logits': addr_type_logits,
            'cf_direction_logits': cf_direction_logits,
            'bb_span_pred': bb_span_pred,
        }
```
