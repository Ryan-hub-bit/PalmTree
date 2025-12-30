"""
Baseline Dataset - Strips position information and normalizes to simple tokens

This dataset is for the baseline comparison where we:
1. Remove ALL position information (no hierarchical positions)
2. Normalize daddr -> address (no distinction between code/data addresses)
3. Flatten var(0xOFFSET) -> var_0xOFFSET (simple vocabulary tokens)
4. Use standard BERT architecture (no address/position embeddings)

Tasks implemented (same as original PalmTree):
- CFG: MLM (Masked Language Modeling) + CWP (Control flow Walk Prediction via is_next)
- DFG: MLM (Masked Language Modeling) + DUP (Data Use Prediction via is_next)

Example transformation:
Input:  call(0x401234:0.12:0.45:0.78) address(0x402000:0.23:0.56:0.89) daddr(0x600000:0.11:0.22:0.33) var(0x20)
Output: call address address var_0x20

Structure: Identical to dataset.py, only difference is the _mask_positions() preprocessing.
"""

from torch.utils.data import Dataset
import tqdm
import torch
import random
import pickle as pkl
import re


class BaselineDataset(Dataset):
    def __init__(self, dfg_corpus_path, cfg_corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, token_mask_prob=0.15, data_percentage=1.0):
        self.vocab = vocab
        self.seq_len = seq_len
        self.token_mask_prob = token_mask_prob  # Masking probability for MLM
        self.data_percentage = data_percentage  # Percentage of data to use

        self.bb_len = 50

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.dfg_corpus_path = dfg_corpus_path
        self.cfg_corpus_path = cfg_corpus_path
        self.encoding = encoding

        # Regex patterns for masking positions (BASELINE SPECIFIC)
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')

        # load DFG sequences 
        with open(dfg_corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.dfg_lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
                self.corpus_lines = len(self.dfg_lines)
       
       # load CFG sequences 
        with open(cfg_corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.cfg_lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
                if self.corpus_lines > len(self.cfg_lines):    
                    self.corpus_lines = len(self.cfg_lines)
        
        # Apply data_percentage sampling
        if self.data_percentage < 1.0 and on_memory:
            original_size = self.corpus_lines
            self.corpus_lines = int(self.corpus_lines * self.data_percentage)
            self.dfg_lines = self.dfg_lines[:self.corpus_lines]
            self.cfg_lines = self.cfg_lines[:self.corpus_lines]
            print(f"Using {self.data_percentage*100}% of data: {self.corpus_lines}/{original_size} samples")
        
        # Note: on_memory=False mode is not fully supported for dual CFG+DFG dataset
        # The original implementation had a bug (corpus_path undefined)
        # Keeping this for compatibility but it won't work correctly
        if not on_memory:
            # Bug: Should use self.cfg_corpus_path or self.dfg_corpus_path
            # This mode is not recommended - use on_memory=True instead
            self.file = open(self.cfg_corpus_path, "r", encoding=encoding)
            self.random_file = open(self.cfg_corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()


    def __len__(self):
        return self.corpus_lines

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

    def __getitem__(self, item):
        c1, c2, c_label, d1, d2, d_label = self.random_sent(item)

        # Apply position masking to CFG sequences (BASELINE SPECIFIC)
        c1 = self._mask_positions(c1)
        c2 = self._mask_positions(c2)

        # Apply MLM masking to CFG (for MLM task)
        c1_random, c1_label = self.random_word(c1)
        c2_random, c2_label = self.random_word(c2)

        c1 = [self.vocab.sos_index] + c1_random + [self.vocab.eos_index]
        c2 = c2_random + [self.vocab.eos_index]

        # Apply position masking to DFG sequences (BASELINE SPECIFIC)
        d1 = self._mask_positions(d1)
        d2 = self._mask_positions(d2)

        # DFG: NO masking, only for DUP (next sentence prediction)
        d1 = [self.vocab.sos_index] + [self.vocab.stoi.get(token, self.vocab.unk_index) for token in d1.split()] + [self.vocab.eos_index]
        d2 = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in d2.split()] + [self.vocab.eos_index]

        # CFG labels for MLM
        c1_label = [self.vocab.pad_index] + c1_label + [self.vocab.pad_index]
        c2_label = c2_label + [self.vocab.pad_index]

        cfg_segment_label = ([1 for _ in range(len(c1))] + [2 for _ in range(len(c2))])[:self.seq_len]
        dfg_segment_label = ([1 for _ in range(len(d1))] + [2 for _ in range(len(d2))])[:self.seq_len]
        cfg_bert_input = (c1 + c2)[:self.seq_len]
        cfg_bert_label = (c1_label + c2_label)[:self.seq_len]

        dfg_bert_input = (d1 + d2)[:self.seq_len]

        cfg_padding = [self.vocab.pad_index for _ in range(self.seq_len - len(cfg_bert_input))]
        cfg_bert_input.extend(cfg_padding), cfg_bert_label.extend(cfg_padding), cfg_segment_label.extend(cfg_padding)
        dfg_padding = [self.vocab.pad_index for _ in range(self.seq_len - len(dfg_bert_input))]
        dfg_bert_input.extend(dfg_padding), dfg_segment_label.extend(dfg_padding)

        output = {"cfg_bert_input": cfg_bert_input,
                  "cfg_bert_label": cfg_bert_label,
                  "cfg_segment_label": cfg_segment_label,
                  "cfg_is_next": c_label,
                  "dfg_bert_input": dfg_bert_input,
                  "dfg_segment_label": dfg_segment_label,
                  "dfg_is_next": d_label
                  }

        return {key: torch.tensor(value) for key, value in output.items()}


    def random_bb(self):
        prob = random.random()
        if prob > 0.5:
            bb_pair = self.bb_pairs[random.choice(list(self.bb_pairs.keys()))]
            return bb_pair, 1
        else:
            neg_keys = random.choices(list(self.bb_pairs.keys()), k=2)
            bb_pair = (self.bb_pairs[neg_keys[0]][0], self.bb_pairs[neg_keys[1]][1])
            return bb_pair, 0 


    def get_index_bb(self, bb_pair):
        tokens1 = [self.vocab.sos_index]
        segment1 = [1]
        i = 1
        for ins in bb_pair[0].split(";")[-5:]:
            if ins:
                for token in ins.split():
                    tokens1.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    segment1.append(i)
                tokens1.append(self.vocab.eos_index)
                segment1.append(i)
                i += 1
         
        tokens2 = [self.vocab.sos_index]
        segment2 = [1]
        j = 1
        for ins in bb_pair[0].split(";")[-5:]:
            if ins:
                for token in ins.split():
                    tokens2.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    segment2.append(j)
                tokens2.append(self.vocab.eos_index)
                segment2.append(j)
                j += 1

        tokens1 = tokens1[:self.bb_len]
        tokens2 = tokens2[:self.bb_len]

        segment1 = segment1[:self.bb_len]
        segment2 = segment2[:self.bb_len]

        padding1 = [self.vocab.pad_index for _ in range(self.bb_len - len(tokens1))]
        padding2 = [self.vocab.pad_index for _ in range(self.bb_len - len(tokens2))]

        tokens1.extend(padding1)
        tokens2.extend(padding2)

        segment1.extend(padding1)
        segment2.extend(padding2)

        return tokens1, tokens2, segment1, segment2
     

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < self.token_mask_prob:
                prob /= self.token_mask_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
        
        return tokens, output_label


    def random_sent(self, index):
        c1, c2, d1, d2 = self.get_corpus_line(index)
        dice = random.random() # TODO: should throw the dice twice here. 
        if dice < 0.25:
            return c1, c2, 1, d1, d2, 1
        elif 0.25 <= dice < 0.5:
            return c1, self.get_random_line(), 0, d1, d2, 1
        elif 0.5 <= dice < 0.75:
            return c1, c2, 1, d2, d1, 0
        else:
            return c1, self.get_random_line(), 0, d2, d1, 0


    def get_corpus_line(self, item):
        if self.on_memory:
            return self.cfg_lines[item][0], self.cfg_lines[item][1], self.dfg_lines[item][0], self.dfg_lines[item][1]

        # now only on_memory copurs are supported
        # else:
        #     line = self.file.__next__()
        #     if line is None:
        #         self.file.close()
        #         self.file = open(self.corpus_path, "r", encoding=self.encoding)
        #         line = self.file.__next__()

        #     t1, t2 = line[:-1].split("\t")
        #     return t1, t2 


    def get_random_line(self):
        if self.on_memory:
            l = self.cfg_lines[random.randrange(len(self.cfg_lines))]
            return l[1]

        # now only on_memory copurs are supported
        # line = self.file.__next__()
        # if line is None:
        #     self.file.close()
        #     self.file = open(self.corpus_path, "r", encoding=self.encoding)
        #     for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
        #         self.random_file.__next__()
        #     line = self.random_file.__next__()
        # return line[:-1].split("\t")[1] 