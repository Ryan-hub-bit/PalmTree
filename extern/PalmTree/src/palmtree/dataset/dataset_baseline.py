"""
Baseline Dataset - Strips position information and normalizes to simple tokens

This dataset is for the baseline comparison where we:
1. Remove ALL position information (no hierarchical positions)
2. Normalize daddr -> address (no distinction between code/data addresses)
3. Flatten var(0xOFFSET) -> var_0xOFFSET (simple vocabulary tokens)
4. Use standard BERT architecture (no address/position embeddings)

Example transformation:
Input:  call(0x401234:0.12:0.45:0.78) address(0x402000:0.23:0.56:0.89) daddr(0x600000:0.11:0.22:0.33) var(0x20)
Output: call address address var_0x20

This creates a clean baseline without any structural advantages from the new format.
"""

import torch
from torch.utils.data import Dataset
import re
import random
from tqdm import tqdm


class BaselineDataset(Dataset):
    """
    Baseline dataset that masks position numbers but keeps token structure.
    
    This allows fair comparison by:
    - Using same data format
    - Masking position information (so model can't use it)
    - Using standard BERT (no address embeddings)
    """
    
    def __init__(
        self,
        cfg_corpus_path,
        dfg_corpus_path,
        vocab,
        seq_len=512,
        encoding="utf-8",
        on_memory=True,
        token_mask_prob=0.15,
        instruction_mask_prob=0.25,
        data_percentage=1.0,
        enable_imd=False,
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.token_mask_prob = token_mask_prob
        self.instruction_mask_prob = instruction_mask_prob
        self.enable_imd = enable_imd
        
        # Special token IDs
        self.pad_idx = vocab.stoi.get('<pad>', 0)
        self.unk_idx = vocab.stoi.get('<unk>', 1)
        self.eos_idx = vocab.stoi.get('<eos>', 2)
        self.sos_idx = vocab.stoi.get('<sos>', 3)
        self.mask_idx = vocab.stoi.get('<mask>', 4)
        
        # Regex patterns - same as address-aware version
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
        
        # Load CFG data
        self.cfg_lines = []
        if cfg_corpus_path:
            print(f"Loading CFG corpus from {cfg_corpus_path}")
            self.cfg_lines = self._load_corpus(cfg_corpus_path)
            print(f"Loaded {len(self.cfg_lines)} CFG lines")
        
        # Load DFG data if enabled
        self.dfg_lines = []
        if self.enable_imd and dfg_corpus_path:
            print(f"Loading DFG corpus from {dfg_corpus_path}")
            self.dfg_lines = self._load_corpus(dfg_corpus_path)
            print(f"Loaded {len(self.dfg_lines)} DFG lines")
        
        # Apply data percentage
        if data_percentage < 1.0:
            cfg_size = int(len(self.cfg_lines) * data_percentage)
            self.cfg_lines = self.cfg_lines[:cfg_size]
            if self.dfg_lines:
                dfg_size = int(len(self.dfg_lines) * data_percentage)
                self.dfg_lines = self.dfg_lines[:dfg_size]
        
        print(f"Baseline dataset size:")
        print(f"  CFG lines: {len(self.cfg_lines)}")
        if self.dfg_lines:
            print(f"  DFG lines: {len(self.dfg_lines)}")
        print(f"  Format: Simple tokens (no positions, daddr->address, var->var_0xOFFSET)")
    
    def _load_corpus(self, path):
        """Load corpus file"""
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines
    
    def _mask_positions(self, text):
        """
        Mask all position numbers and normalize format for baseline.
        
        Transformations:
          1. daddr(0xADDR:pos:pos:pos) -> address (convert data addresses to generic address token)
          2. address(0xADDR:pos:pos:pos) -> address (remove position info)
          3. opcode(0xADDR:pos:pos:pos) -> opcode (remove position info from opcodes)
          4. var(0xOFFSET) -> var_0xOFFSET (flatten var to simple token)
        
        Example:
          Input:  call(0x401234:0.12:0.45:0.78) daddr(0x600000:0.11:0.22:0.33) var(0x20)
          Output: call address var_0x20
        
        This creates a true baseline without any position or structural information.
        """
        # Replace daddr with position info -> just "address" token
        text = self.daddr_pattern.sub('address', text)
        
        # Replace address with position info -> just "address" token
        text = self.nested_addr_pattern.sub('address', text)
        
        # Replace opcode(addr:pos:pos:pos) -> just opcode
        def replace_opcode(match):
            return match.group(1)  # Return only the opcode
        text = self.addr_pattern.sub(replace_opcode, text)
        
        # Replace var(0xOFFSET) -> var_0xOFFSET
        def replace_var(match):
            offset = match.group(1)
            return f"var_{offset}"
        text = self.var_pattern.sub(replace_var, text)
        
        return text
    
    def _parse_instruction_baseline(self, inst_text):
        """
        Parse instruction for baseline model.
        After _mask_positions(), the text is already simplified:
        - No position info
        - daddr -> address
        - var(0xOFFSET) -> var_0xOFFSET
        
        Just split into tokens.
        """
        # Apply transformations
        inst_text = self._mask_positions(inst_text)
        
        # Simple split - everything is already normalized
        return inst_text.split()
    
    def _parse_line(self, line):
        """Parse a line into instructions"""
        instructions = line.split('\t')
        parsed_instructions = []
        
        for inst in instructions:
            inst = inst.strip()
            if not inst:
                continue
            
            # Parse instruction tokens (positions masked)
            tokens = self._parse_instruction_baseline(inst)
            parsed_instructions.append(tokens)
        
        return parsed_instructions
    
    def _mask_instruction(self, tokens):
        """Mask entire instruction (all tokens)"""
        masked_tokens = [self.mask_idx] * len(tokens)
        labels = tokens.copy()
        return masked_tokens, labels
    
    def _mask_tokens_mlm(self, tokens):
        """Standard MLM: mask individual tokens at token_mask_prob"""
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # -100 = ignore in loss
        
        for i in range(len(tokens)):
            if random.random() < self.token_mask_prob:
                # Mask this token
                labels[i] = tokens[i]
                
                # 80% mask, 10% random, 10% keep
                rand = random.random()
                if rand < 0.8:
                    masked_tokens[i] = self.mask_idx
                elif rand < 0.9:
                    masked_tokens[i] = random.randint(5, len(self.vocab) - 1)
                # else: keep original
        
        return masked_tokens, labels
    
    def _create_sequence(self, instructions):
        """Create sequence from instructions with masking"""
        all_tokens = []
        all_labels = []
        
        # Add SOS token
        all_tokens.append(self.sos_idx)
        all_labels.append(-100)
        
        # Process each instruction
        for inst_tokens in instructions:
            # Convert tokens to IDs
            token_ids = [self.vocab.stoi.get(t, self.unk_idx) for t in inst_tokens]
            
            # Decide whether to mask entire instruction
            if random.random() < self.instruction_mask_prob:
                # Mask entire instruction (IMC/IMD task)
                masked_ids, labels = self._mask_instruction(token_ids)
            else:
                # Apply standard MLM masking
                masked_ids, labels = self._mask_tokens_mlm(token_ids)
            
            all_tokens.extend(masked_ids)
            all_labels.extend(labels)
        
        # Add EOS token
        all_tokens.append(self.eos_idx)
        all_labels.append(-100)
        
        # Truncate or pad to seq_len
        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
        else:
            padding_len = self.seq_len - len(all_tokens)
            all_tokens.extend([self.pad_idx] * padding_len)
            all_labels.extend([-100] * padding_len)
        
        return all_tokens, all_labels
    
    def __len__(self):
        return len(self.cfg_lines)
    
    def __getitem__(self, idx):
        # Get CFG line
        cfg_line = self.cfg_lines[idx]
        cfg_instructions = self._parse_line(cfg_line)
        cfg_tokens, cfg_labels = self._create_sequence(cfg_instructions)
        
        output = {
            'bert_input': torch.tensor(cfg_tokens, dtype=torch.long),
            'bert_label': torch.tensor(cfg_labels, dtype=torch.long),
        }
        
        # Add DFG if enabled
        if self.enable_imd and self.dfg_lines:
            dfg_idx = idx % len(self.dfg_lines)
            dfg_line = self.dfg_lines[dfg_idx]
            dfg_instructions = self._parse_line(dfg_line)
            dfg_tokens, dfg_labels = self._create_sequence(dfg_instructions)
            
            output['dfg_bert_input'] = torch.tensor(dfg_tokens, dtype=torch.long)
            output['dfg_bert_label'] = torch.tensor(dfg_labels, dtype=torch.long)
        
        return output


if __name__ == "__main__":
    # Test the dataset
    import pickle
    
    print("Testing BaselineDataset...")
    
    # Load vocab
    vocab_path = './vocab.pkl'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Create dataset
    dataset = BaselineDataset(
        cfg_corpus_path='/data/kun/palmtreedata/cfg_train_2.txt',
        dfg_corpus_path=None,
        vocab=vocab,
        seq_len=512,
        data_percentage=0.01,  # Use 1% for testing
    )
    
    # Test one sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Input shape: {sample['bert_input'].shape}")
    print(f"Label shape: {sample['bert_label'].shape}")
    print(f"Input tokens (first 20): {sample['bert_input'][:20]}")
    print(f"Labels (first 20): {sample['bert_label'][:20]}")
    
    # Decode some tokens
    print("\nDecoded tokens (first 10):")
    for i in range(10):
        token_id = sample['bert_input'][i].item()
        if token_id < len(vocab.itos):
            print(f"  {i}: {vocab.itos[token_id]}")
    
    print("\nBaseline dataset test complete!")
