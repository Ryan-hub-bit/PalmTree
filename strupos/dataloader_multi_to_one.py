"""
Multi-to-One NSP DataLoader for Address-Aware BERT

Uses N-1 instructions to predict the Nth instruction for CFG NSP.

For a line with 8 instructions:
- MLM: Uses all 8 instructions (one sample per line)
- NSP-CFG: Uses first 7 instructions to predict the 8th
- NSP-DFG: Uses DFG as-is (already in pair format)

This provides richer context for NSP prediction.
"""

import torch
from torch.utils.data import Dataset
import re
import random
from tqdm import tqdm


class MultiToOneDataset(Dataset):
    """
    Dataset that uses N-1 instructions to predict the Nth instruction for NSP.
    
    Strategy:
    - MLM: Iterate through CFG lines (one MLM sample per line)
    - NSP-CFG: Use first N-1 instructions to predict Nth (e.g., [1-7] predict 8)
    - NSP-DFG: Use DFG lines as-is (already in pair format)
    
    Segmentation modes:
    - instruction_level_segment=False: Context=segment1, Target=segment2 (standard BERT)
    - instruction_level_segment=True: Each instruction gets its own segment ID
    """
    
    def __init__(
        self,
        cfg_corpus_path,
        dfg_corpus_path,
        vocab,
        seq_len=512,
        nsp_content_max=20,
        encoding="utf-8",
        on_memory=True,
        nsp_prob=0.5,
        mask_prob=0.15,
        data_percentage=1.0,
        train_split=1.0,
        is_train=True,
        instruction_level_segment=False,
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.nsp_content_max = nsp_content_max
        self.on_memory = on_memory
        self.nsp_prob = nsp_prob
        self.instruction_level_segment = instruction_level_segment
        self.mask_prob = mask_prob
        self.encoding = encoding
        
        # Regex patterns for parsing instructions with addresses
        self.addr_pattern = re.compile(
            r"^([a-zA-Z0-9_.]+)\(([^:]+):([^:]+):([^:]+):([^)]+)\)"
        )
        self.nested_addr_pattern = re.compile(
            r"^address\(([^:]+):([^:]+):([^:]+):([^)]+)\)(.*)$"
        )
        
        # Load CFG and DFG data
        print(f"Loading CFG corpus from {cfg_corpus_path}...")
        with open(cfg_corpus_path, "r", encoding=encoding) as f:
            cfg_lines = [line.strip() for line in f if line.strip()]
        
        print(f"Loading DFG corpus from {dfg_corpus_path}...")
        with open(dfg_corpus_path, "r", encoding=encoding) as f:
            dfg_lines = [line.strip() for line in f if line.strip()]
        
        # Sample data if needed
        if data_percentage < 1.0:
            cfg_sample_size = max(1, int(len(cfg_lines) * data_percentage))
            dfg_sample_size = max(1, int(len(dfg_lines) * data_percentage))
            cfg_lines = cfg_lines[:cfg_sample_size]
            dfg_lines = dfg_lines[:dfg_sample_size]
            print(f"Sampled {data_percentage*100}% of data: {len(cfg_lines)} CFG lines, {len(dfg_lines)} DFG lines")
        
        self.cfg_lines = cfg_lines
        self.dfg_lines = dfg_lines
        
        # Create NSP samples using multi-to-one strategy
        print("Creating multi-to-one NSP samples from CFG...")
        self.nsp_cfg_samples = []
        for line_idx, line in enumerate(tqdm(cfg_lines, desc="Processing CFG")):
            sample = self._create_multi_to_one_sample(line, line_idx)
            if sample:
                self.nsp_cfg_samples.append(sample)
        
        print(f"Dataset statistics:")
        print(f"  CFG lines (for MLM): {len(self.cfg_lines)}")
        print(f"  NSP-CFG samples: {len(self.nsp_cfg_samples)}")
        print(f"  DFG lines (for NSP): {len(self.dfg_lines)}")
    
    def _create_multi_to_one_sample(self, line, line_idx):
        """
        Create a multi-to-one NSP sample: use first N-1 instructions to predict Nth.
        
        Args:
            line: Tab-separated instructions
            line_idx: Index of this line in cfg_lines
        
        Returns:
            (line_idx, context_instructions, target_instruction, label) or None
        """
        instructions = line.split('\t')
        
        # Need at least 2 instructions (ideally 8 for full context)
        if len(instructions) < 2:
            return None
        
        # Context = all but last instruction, Target = last instruction
        context_insts = instructions[:-1]  # [0, 1, 2, ..., N-2]
        target_original = instructions[-1]  # N-1
        
        # Decide if positive or negative
        if random.random() < self.nsp_prob:
            # NEGATIVE: Replace target with random instruction
            target = target_original
            max_attempts = 10
            attempts = 0
            while target == target_original and attempts < max_attempts:
                random_line = random.choice(self.cfg_lines)
                random_instructions = random_line.split('\t')
                target = random.choice(random_instructions)
                attempts += 1
            label = 0
        else:
            # POSITIVE: Keep the original last instruction
            target = target_original
            label = 1
        
        # Join context instructions with tab
        context = '\t'.join(context_insts)
        
        return (line_idx, context, target, label)
    
    def _parse_instruction(self, inst_text):
        """Parse a single instruction and extract tokens + positions"""
        tokens = []
        positions = []
        
        # Match opcode with address
        match = self.addr_pattern.match(inst_text)
        if not match:
            return [inst_text], [(0.0, 0.0, 0.0)]
        
        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))
        
        tokens.append(opcode)
        positions.append((binary_pos, function_pos, bb_pos))
        
        # Parse operands
        operands_text = inst_text[match.end():].strip()
        if operands_text:
            for operand in operands_text.split():
                nested_match = self.nested_addr_pattern.match(operand)
                if nested_match:
                    nested_binary_pos = float(nested_match.group(2))
                    nested_function_pos = float(nested_match.group(3))
                    nested_bb_pos = float(nested_match.group(4))
                    rest = nested_match.group(5)
                    
                    if rest:
                        tokens.append(rest)
                        positions.append((nested_binary_pos, nested_function_pos, nested_bb_pos))
                else:
                    tokens.append(operand)
                    positions.append((binary_pos, function_pos, bb_pos))
        
        return tokens, positions
    
    def _parse_instruction_sequence(self, inst_sequence_text):
        """Parse a sequence of tab-separated instructions"""
        instructions = inst_sequence_text.split('\t')
        all_tokens = []
        all_positions = []
        
        for inst in instructions:
            tokens, positions = self._parse_instruction(inst)
            all_tokens.extend(tokens)
            all_positions.extend(positions)
        
        return all_tokens, all_positions
    
    def __len__(self):
        # Return max of all task sample counts
        return max(len(self.cfg_lines), len(self.nsp_cfg_samples), len(self.dfg_lines))
    
    def __getitem__(self, idx):
        # For each index, we need to return samples for all enabled tasks
        # We'll cycle through each dataset
        
        # MLM from CFG
        cfg_idx = idx % len(self.cfg_lines)
        mlm_output = self.get_mlm_sample(cfg_idx)
        
        # NSP-CFG (multi-to-one)
        nsp_cfg_idx = idx % len(self.nsp_cfg_samples)
        nsp_cfg_output = self.get_nsp_cfg_sample(nsp_cfg_idx)
        
        # NSP-DFG
        dfg_idx = idx % len(self.dfg_lines)
        nsp_dfg_output = self.get_nsp_dfg_sample(dfg_idx)
        
        return {
            "mlm": mlm_output,
            "nsp_cfg": nsp_cfg_output,
            "nsp_dfg": nsp_dfg_output,
        }
    
    def get_mlm_sample(self, idx):
        """Get MLM sample from CFG line"""
        line = self.cfg_lines[idx]
        tokens, positions = self._parse_instruction_sequence(line)
        
        # Convert to token IDs
        token_ids = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in tokens]
        
        # Truncate or pad to seq_len
        if len(token_ids) > self.seq_len:
            token_ids = token_ids[:self.seq_len]
            positions = positions[:self.seq_len]
        else:
            pad_len = self.seq_len - len(token_ids)
            token_ids.extend([self.vocab.pad_index] * pad_len)
            positions.extend([(0.0, 0.0, 0.0)] * pad_len)
        
        # Apply masking
        output_label = []
        for i, token_id in enumerate(token_ids):
            prob = random.random()
            if prob < self.mask_prob and token_id not in [self.vocab.pad_index, self.vocab.sos_index, self.vocab.eos_index]:
                prob /= self.mask_prob
                
                if prob < 0.8:
                    token_ids[i] = self.vocab.mask_index
                elif prob < 0.9:
                    token_ids[i] = random.randrange(len(self.vocab))
                
                output_label.append(token_id)
            else:
                output_label.append(0)
        
        return {
            "bert_input": token_ids,
            "bert_label": output_label,
            "binary_pos": [p[0] for p in positions],
            "function_pos": [p[1] for p in positions],
            "bb_pos": [p[2] for p in positions],
            "segment_label": [1] * self.seq_len,
        }
    
    def get_nsp_cfg_sample(self, idx):
        """Get NSP-CFG sample using multi-to-one strategy"""
        line_idx, context, target, label = self.nsp_cfg_samples[idx]
        
        if self.instruction_level_segment:
            return self._get_nsp_cfg_instruction_level(context, target, label)
        else:
            return self._get_nsp_cfg_standard(context, target, label)
    
    def _get_nsp_cfg_standard(self, context, target, label):
        """Standard BERT segmentation: context=1, target=2"""
        # Parse context (N-1 instructions)
        context_tokens, context_positions = self._parse_instruction_sequence(context)
        
        # Parse target (1 instruction)
        target_tokens, target_positions = self._parse_instruction(target)
        
        # Truncate context if too long (leave room for target)
        max_context_len = self.nsp_content_max - len(target_tokens) - 3  # -3 for [CLS] and 2 [SEP]
        if len(context_tokens) > max_context_len:
            context_tokens = context_tokens[:max_context_len]
            context_positions = context_positions[:max_context_len]
        
        # Build token sequence: [CLS] context [SEP] target [SEP]
        token_ids = [self.vocab.sos_index]
        positions = [(0.0, 0.0, 0.0)]
        segment_labels = [1]
        
        # Add context
        for token in context_tokens:
            token_ids.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        positions.extend(context_positions)
        segment_labels.extend([1] * len(context_tokens))
        
        # Add SEP
        token_ids.append(self.vocab.eos_index)
        positions.append((0.0, 0.0, 0.0))
        segment_labels.append(1)
        
        # Add target
        for token in target_tokens:
            token_ids.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        positions.extend(target_positions)
        segment_labels.extend([2] * len(target_tokens))
        
        # Add final SEP
        token_ids.append(self.vocab.eos_index)
        positions.append((0.0, 0.0, 0.0))
        segment_labels.append(2)
        
        # Truncate or pad to nsp_content_max
        if len(token_ids) > self.nsp_content_max:
            token_ids = token_ids[:self.nsp_content_max]
            positions = positions[:self.nsp_content_max]
            segment_labels = segment_labels[:self.nsp_content_max]
        else:
            pad_len = self.nsp_content_max - len(token_ids)
            token_ids.extend([self.vocab.pad_index] * pad_len)
            positions.extend([(0.0, 0.0, 0.0)] * pad_len)
            segment_labels.extend([0] * pad_len)
        
        return {
            "bert_input": token_ids,
            "segment_label": segment_labels,
            "nsp_label": label,
            "binary_pos": [p[0] for p in positions],
            "function_pos": [p[1] for p in positions],
            "bb_pos": [p[2] for p in positions],
        }
    
    def _get_nsp_cfg_instruction_level(self, context, target, label):
        """Instruction-level segmentation: Each instruction gets its own segment ID"""
        # Parse context and target as individual instructions
        context_instructions = context.split('\t')
        
        # Build token sequence: [CLS] inst1 [SEP] inst2 [SEP] ... instN [SEP]
        token_ids = [self.vocab.sos_index]
        positions = [(0.0, 0.0, 0.0)]
        segment_labels = [0]  # [CLS] gets segment 0
        
        current_segment = 1
        
        # Add each context instruction with its own segment ID
        for inst in context_instructions:
            inst_tokens, inst_positions = self._parse_instruction(inst)
            
            # Check if we have room
            if len(token_ids) + len(inst_tokens) + 1 >= self.nsp_content_max - 10:  # Leave room for target
                break
            
            # Add instruction tokens
            for token in inst_tokens:
                token_ids.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            positions.extend(inst_positions)
            segment_labels.extend([current_segment] * len(inst_tokens))
            
            # Add SEP
            token_ids.append(self.vocab.sep_index)
            positions.append((0.0, 0.0, 0.0))
            segment_labels.append(current_segment)
            
            current_segment += 1
        
        # Add target instruction with its own segment ID
        target_tokens, target_positions = self._parse_instruction(target)
        for token in target_tokens:
            token_ids.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        positions.extend(target_positions)
        segment_labels.extend([current_segment] * len(target_tokens))
        
        # Add final SEP
        token_ids.append(self.vocab.sep_index)
        positions.append((0.0, 0.0, 0.0))
        segment_labels.append(current_segment)
        
        # Truncate or pad to nsp_content_max
        if len(token_ids) > self.nsp_content_max:
            token_ids = token_ids[:self.nsp_content_max]
            positions = positions[:self.nsp_content_max]
            segment_labels = segment_labels[:self.nsp_content_max]
        else:
            pad_len = self.nsp_content_max - len(token_ids)
            token_ids.extend([self.vocab.pad_index] * pad_len)
            positions.extend([(0.0, 0.0, 0.0)] * pad_len)
            segment_labels.extend([0] * pad_len)  # Padding gets segment 0
        
        return {
            "bert_input": token_ids,
            "segment_label": segment_labels,
            "nsp_label": label,
            "binary_pos": [p[0] for p in positions],
            "function_pos": [p[1] for p in positions],
            "bb_pos": [p[2] for p in positions],
        }
    
    def get_nsp_dfg_sample(self, idx):
        """Get NSP-DFG sample (DFG already in pair format)"""
        line = self.dfg_lines[idx]
        parts = line.split('\t')
        
        if len(parts) != 3:
            # Fallback for malformed lines
            inst1 = parts[0] if len(parts) > 0 else ""
            inst2 = parts[1] if len(parts) > 1 else ""
            label = 1
        else:
            inst1, inst2, label = parts
            label = int(label)
        
        # Parse both instructions
        tokens1, positions1 = self._parse_instruction(inst1)
        tokens2, positions2 = self._parse_instruction(inst2)
        
        # Build token sequence: [CLS] inst1 [SEP] inst2 [SEP]
        token_ids = [self.vocab.sos_index]
        positions = [(0.0, 0.0, 0.0)]
        segment_labels = [1]
        
        # Add inst1
        for token in tokens1:
            token_ids.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        positions.extend(positions1)
        segment_labels.extend([1] * len(tokens1))
        
        # Add SEP
        token_ids.append(self.vocab.eos_index)
        positions.append((0.0, 0.0, 0.0))
        segment_labels.append(1)
        
        # Add inst2
        for token in tokens2:
            token_ids.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        positions.extend(positions2)
        segment_labels.extend([2] * len(tokens2))
        
        # Add final SEP
        token_ids.append(self.vocab.eos_index)
        positions.append((0.0, 0.0, 0.0))
        segment_labels.append(2)
        
        # Truncate or pad to nsp_content_max
        if len(token_ids) > self.nsp_content_max:
            token_ids = token_ids[:self.nsp_content_max]
            positions = positions[:self.nsp_content_max]
            segment_labels = segment_labels[:self.nsp_content_max]
        else:
            pad_len = self.nsp_content_max - len(token_ids)
            token_ids.extend([self.vocab.pad_index] * pad_len)
            positions.extend([(0.0, 0.0, 0.0)] * pad_len)
            segment_labels.extend([0] * pad_len)
        
        return {
            "bert_input": token_ids,
            "segment_label": segment_labels,
            "nsp_label": label,
            "binary_pos": [p[0] for p in positions],
            "function_pos": [p[1] for p in positions],
            "bb_pos": [p[2] for p in positions],
        }

def multi_to_one_collate_fn(batch):
    """
    Custom collate function to convert nested batch structure to flat structure
    expected by the training loop.
    
    Input: List of dicts with structure:
        {
            "mlm": {"bert_input": [...], "bert_label": [...], ...},
            "nsp_cfg": {"bert_input": [...], "segment_label": [...], "nsp_label": ..., ...},
            "nsp_dfg": {"bert_input": [...], "segment_label": [...], "nsp_label": ..., ...}
        }
    
    Output: Single dict with structure:
        {
            "cfg_mlm_input": tensor,
            "cfg_mlm_label": tensor,
            "cfg_mlm_binary_pos": tensor,
            "cfg_mlm_function_pos": tensor,
            "cfg_mlm_bb_pos": tensor,
            "cfg_nsp_input": tensor,
            "cfg_segment_label": tensor,
            "cfg_is_next": tensor,
            "cfg_nsp_binary_pos": tensor,
            "cfg_nsp_function_pos": tensor,
            "cfg_nsp_bb_pos": tensor,
            "dfg_nsp_input": tensor,
            "dfg_segment_label": tensor,
            "dfg_is_next": tensor,
            "dfg_nsp_binary_pos": tensor,
            "dfg_nsp_function_pos": tensor,
            "dfg_nsp_bb_pos": tensor,
        }
    """
    # Extract MLM data
    mlm_data = [item["mlm"] for item in batch]
    cfg_mlm_input = torch.tensor([d["bert_input"] for d in mlm_data], dtype=torch.long)
    cfg_mlm_label = torch.tensor([d["bert_label"] for d in mlm_data], dtype=torch.long)
    cfg_mlm_binary_pos = torch.tensor([d["binary_pos"] for d in mlm_data], dtype=torch.float)
    cfg_mlm_function_pos = torch.tensor([d["function_pos"] for d in mlm_data], dtype=torch.float)
    cfg_mlm_bb_pos = torch.tensor([d["bb_pos"] for d in mlm_data], dtype=torch.float)
    
    # Extract NSP-CFG data
    nsp_cfg_data = [item["nsp_cfg"] for item in batch]
    cfg_nsp_input = torch.tensor([d["bert_input"] for d in nsp_cfg_data], dtype=torch.long)
    cfg_segment_label = torch.tensor([d["segment_label"] for d in nsp_cfg_data], dtype=torch.long)
    cfg_is_next = torch.tensor([d["nsp_label"] for d in nsp_cfg_data], dtype=torch.long)
    cfg_nsp_binary_pos = torch.tensor([d["binary_pos"] for d in nsp_cfg_data], dtype=torch.float)
    cfg_nsp_function_pos = torch.tensor([d["function_pos"] for d in nsp_cfg_data], dtype=torch.float)
    cfg_nsp_bb_pos = torch.tensor([d["bb_pos"] for d in nsp_cfg_data], dtype=torch.float)
    
    # Extract NSP-DFG data
    nsp_dfg_data = [item["nsp_dfg"] for item in batch]
    dfg_nsp_input = torch.tensor([d["bert_input"] for d in nsp_dfg_data], dtype=torch.long)
    dfg_segment_label = torch.tensor([d["segment_label"] for d in nsp_dfg_data], dtype=torch.long)
    dfg_is_next = torch.tensor([d["nsp_label"] for d in nsp_dfg_data], dtype=torch.long)
    dfg_nsp_binary_pos = torch.tensor([d["binary_pos"] for d in nsp_dfg_data], dtype=torch.float)
    dfg_nsp_function_pos = torch.tensor([d["function_pos"] for d in nsp_dfg_data], dtype=torch.float)
    dfg_nsp_bb_pos = torch.tensor([d["bb_pos"] for d in nsp_dfg_data], dtype=torch.float)
    
    return {
        "cfg_mlm_input": cfg_mlm_input,
        "cfg_mlm_label": cfg_mlm_label,
        "cfg_mlm_binary_pos": cfg_mlm_binary_pos,
        "cfg_mlm_function_pos": cfg_mlm_function_pos,
        "cfg_mlm_bb_pos": cfg_mlm_bb_pos,
        "cfg_nsp_input": cfg_nsp_input,
        "cfg_segment_label": cfg_segment_label,
        "cfg_is_next": cfg_is_next,
        "cfg_nsp_binary_pos": cfg_nsp_binary_pos,
        "cfg_nsp_function_pos": cfg_nsp_function_pos,
        "cfg_nsp_bb_pos": cfg_nsp_bb_pos,
        "dfg_nsp_input": dfg_nsp_input,
        "dfg_segment_label": dfg_segment_label,
        "dfg_is_next": dfg_is_next,
        "dfg_nsp_binary_pos": dfg_nsp_binary_pos,
        "dfg_nsp_function_pos": dfg_nsp_function_pos,
        "dfg_nsp_bb_pos": dfg_nsp_bb_pos,
    }
