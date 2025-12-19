from torch.utils.data import Dataset
import tqdm
import torch
import random
import re


class BERTDatasetAddressAware(Dataset):
    """
    Address-aware BERT dataset that parses inline address information.
    Format: opcode(addr:bnorm:fnorm:bbnorm) operand1 operand2 ...
    
    Extracts:
    - Tokens (opcode, operands)
    - Address positions (binary_pos, function_pos, bb_pos) for each token
    - Var offsets for var(0xXX) tokens
    """
    def __init__(self, dfg_corpus_path, cfg_corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.dfg_corpus_path = dfg_corpus_path
        self.cfg_corpus_path = cfg_corpus_path
        self.encoding = encoding

        # Load DFG sequences 
        with open(dfg_corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.dfg_lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading DFG Dataset", total=corpus_lines)]
                
                self.corpus_lines = len(self.dfg_lines)
       
        # Load CFG sequences 
        with open(cfg_corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.cfg_lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading CFG Dataset", total=corpus_lines)]
                
                if self.corpus_lines > len(self.cfg_lines):    
                    self.corpus_lines = len(self.cfg_lines)


    def __len__(self):
        return self.corpus_lines


    def parse_instruction_with_address(self, instruction):
        """
        Parse instruction with inline address information.
        Format examples:
        - mov(0x1000:0.123456:0.234567:0.345678) rax rbx
        - call(0x2000:0.456789:0.567890:0.678901) symbol(0x3000:0.111111:0.222222:0.333333)
        - lea(0x4000:0.444444:0.555555:0.666666) rax var(0x10)
        - mov daddr(0x5000:0.777777:0.888888:0.999999) rax  # Data section address
        
        Returns:
            tokens: list of token strings (opcode, operands without address info)
            binary_positions: list of binary-level normalized positions
            function_positions: list of function-level normalized positions
            bb_positions: list of BB-level normalized positions
            var_offsets: list of variable offsets (-1 for non-var tokens)
            data_binary_positions: list of data section binary-level positions (-1 for non-data addresses)
        """
        tokens = []
        binary_positions = []
        function_positions = []
        bb_positions = []
        var_offsets = []
        data_binary_positions = []
        
        parts = instruction.strip().split()
        
        for part in parts:
            # Check if this is an opcode with address: opcode(addr:bnorm:fnorm:bbnorm)
            opcode_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\((0x[0-9a-fA-F]+):([\d.]+):([\d.]+):([\d.]+)\)', part)
            if opcode_match:
                opcode = opcode_match.group(1)
                # addr = opcode_match.group(2)  # Not used in embedding
                bnorm = float(opcode_match.group(3))
                fnorm = float(opcode_match.group(4))
                bbnorm = float(opcode_match.group(5))
                
                tokens.append(opcode)
                binary_positions.append(bnorm)
                function_positions.append(fnorm)
                bb_positions.append(bbnorm)
                var_offsets.append(-1)
                data_binary_positions.append(-1.0)
                continue
            
            # Check for data section address: daddr(addr:bnorm:fnorm:bbnorm)
            daddr_match = re.match(r'daddr\((0x[0-9a-fA-F]+):([\d.]+):([\d.]+):([\d.]+)\)', part)
            if daddr_match:
                # addr = daddr_match.group(1)  # Not used
                bnorm = float(daddr_match.group(2))
                fnorm = float(daddr_match.group(3))
                bbnorm = float(daddr_match.group(4))
                
                tokens.append('daddr')
                binary_positions.append(-1.0)  # Code positions not applicable for data addresses
                function_positions.append(-1.0)
                bb_positions.append(-1.0)
                var_offsets.append(-1)
                data_binary_positions.append(bnorm)  # Store data section position
                continue
            
            # Check for symbol/string/address with address info
            operand_match = re.match(r'(symbol|string|address)\((0x[0-9a-fA-F]+):([\d.]+):([\d.]+):([\d.]+)\)', part)
            if operand_match:
                operand_type = operand_match.group(1)
                # addr = operand_match.group(2)  # Not used
                bnorm = float(operand_match.group(3))
                fnorm = float(operand_match.group(4))
                bbnorm = float(operand_match.group(5))
                
                tokens.append(operand_type)
                binary_positions.append(bnorm)
                function_positions.append(fnorm)
                bb_positions.append(bbnorm)
                var_offsets.append(-1)
                data_binary_positions.append(-1.0)
                continue
            
            # Check for var(0xXX) tokens
            var_match = re.match(r'var\((0x[0-9a-fA-F]+)\)', part)
            if var_match:
                offset_hex = var_match.group(1)
                offset_val = int(offset_hex, 16)
                
                tokens.append('var')
                # For var tokens, use -1 for address positions (not applicable)
                binary_positions.append(-1.0)
                function_positions.append(-1.0)
                bb_positions.append(-1.0)
                var_offsets.append(offset_val)
                data_binary_positions.append(-1.0)
                continue
            
            # Regular token (register, immediate, etc.)
            tokens.append(part)
            binary_positions.append(-1.0)
            function_positions.append(-1.0)
            bb_positions.append(-1.0)
            var_offsets.append(-1)
            data_binary_positions.append(-1.0)
        
        return tokens, binary_positions, function_positions, bb_positions, var_offsets, data_binary_positions


    def __getitem__(self, item):
        c1, c2, c_label, d1, d2, d_label = self.random_sent(item)

        # Parse DFG sequences with address info
        d1_tokens, d1_bpos, d1_fpos, d1_bbpos, d1_voffsets, d1_dbpos = self.parse_instruction_with_address(d1)
        d2_tokens, d2_bpos, d2_fpos, d2_bbpos, d2_voffsets, d2_dbpos = self.parse_instruction_with_address(d2)

        # Apply random masking to DFG
        d1_random, d1_label, d1_bpos, d1_fpos, d1_bbpos, d1_voffsets, d1_dbpos = self.random_word(
            d1_tokens, d1_bpos, d1_fpos, d1_bbpos, d1_voffsets, d1_dbpos
        )
        d2_random, d2_label, d2_bpos, d2_fpos, d2_bbpos, d2_voffsets, d2_dbpos = self.random_word(
            d2_tokens, d2_bpos, d2_fpos, d2_bbpos, d2_voffsets, d2_dbpos
        )

        # Convert to indices and add special tokens
        d1 = [self.vocab.sos_index] + d1_random + [self.vocab.eos_index]
        d2 = d2_random + [self.vocab.eos_index]
        
        d1_bpos = [-1.0] + d1_bpos + [-1.0]
        d1_fpos = [-1.0] + d1_fpos + [-1.0]
        d1_bbpos = [-1.0] + d1_bbpos + [-1.0]
        d1_voffsets = [-1] + d1_voffsets + [-1]
        d1_dbpos = [-1.0] + d1_dbpos + [-1.0]
        
        d2_bpos = d2_bpos + [-1.0]
        d2_fpos = d2_fpos + [-1.0]
        d2_bbpos = d2_bbpos + [-1.0]
        d2_voffsets = d2_voffsets + [-1]
        d2_dbpos = d2_dbpos + [-1.0]

        # Parse CFG sequences with address info
        c1_tokens, c1_bpos, c1_fpos, c1_bbpos, c1_voffsets, c1_dbpos = self.parse_instruction_with_address(c1)
        c2_tokens, c2_bpos, c2_fpos, c2_bbpos, c2_voffsets, c2_dbpos = self.parse_instruction_with_address(c2)

        # Convert CFG to indices
        c1 = [self.vocab.sos_index] + [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c1_tokens] + [self.vocab.eos_index]
        c2 = [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c2_tokens] + [self.vocab.eos_index]
        
        c1_bpos = [-1.0] + c1_bpos + [-1.0]
        c1_fpos = [-1.0] + c1_fpos + [-1.0]
        c1_bbpos = [-1.0] + c1_bbpos + [-1.0]
        c1_voffsets = [-1] + c1_voffsets + [-1]
        c1_dbpos = [-1.0] + c1_dbpos + [-1.0]
        
        c2_bpos = c2_bpos + [-1.0]
        c2_fpos = c2_fpos + [-1.0]
        c2_bbpos = c2_bbpos + [-1.0]
        c2_voffsets = c2_voffsets + [-1]
        c2_dbpos = c2_dbpos + [-1.0]

        # Handle labels
        d1_label = [self.vocab.pad_index] + d1_label + [self.vocab.pad_index]
        d2_label = d2_label + [self.vocab.pad_index]

        # Create segment labels
        dfg_segment_label = ([1 for _ in range(len(d1))] + [2 for _ in range(len(d2))])[:self.seq_len]
        cfg_segment_label = ([1 for _ in range(len(c1))] + [2 for _ in range(len(c2))])[:self.seq_len]
        
        # Concatenate sequences
        dfg_bert_input = (d1 + d2)[:self.seq_len]
        dfg_bert_label = (d1_label + d2_label)[:self.seq_len]
        dfg_binary_pos = (d1_bpos + d2_bpos)[:self.seq_len]
        dfg_function_pos = (d1_fpos + d2_fpos)[:self.seq_len]
        dfg_bb_pos = (d1_bbpos + d2_bbpos)[:self.seq_len]
        dfg_var_offsets = (d1_voffsets + d2_voffsets)[:self.seq_len]
        dfg_data_binary_pos = (d1_dbpos + d2_dbpos)[:self.seq_len]

        cfg_bert_input = (c1 + c2)[:self.seq_len]
        cfg_binary_pos = (c1_bpos + c2_bpos)[:self.seq_len]
        cfg_function_pos = (c1_fpos + c2_fpos)[:self.seq_len]
        cfg_bb_pos = (c1_bbpos + c2_bbpos)[:self.seq_len]
        cfg_var_offsets = (c1_voffsets + c2_voffsets)[:self.seq_len]
        cfg_data_binary_pos = (c1_dbpos + c2_dbpos)[:self.seq_len]

        # Padding
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(dfg_bert_input))]
        neg_one_padding = [-1.0 for _ in range(self.seq_len - len(dfg_bert_input))]
        int_neg_one_padding = [-1 for _ in range(self.seq_len - len(dfg_bert_input))]
        
        dfg_bert_input.extend(padding)
        dfg_bert_label.extend(padding)
        dfg_segment_label.extend(padding)
        dfg_binary_pos.extend(neg_one_padding)
        dfg_function_pos.extend(neg_one_padding)
        dfg_bb_pos.extend(neg_one_padding)
        dfg_var_offsets.extend(int_neg_one_padding)
        dfg_data_binary_pos.extend(neg_one_padding)
        
        cfg_padding = [self.vocab.pad_index for _ in range(self.seq_len - len(cfg_bert_input))]
        cfg_neg_one_padding = [-1.0 for _ in range(self.seq_len - len(cfg_bert_input))]
        cfg_int_neg_one_padding = [-1 for _ in range(self.seq_len - len(cfg_bert_input))]
        
        cfg_bert_input.extend(cfg_padding)
        cfg_segment_label.extend(cfg_padding)
        cfg_binary_pos.extend(cfg_neg_one_padding)
        cfg_function_pos.extend(cfg_neg_one_padding)
        cfg_bb_pos.extend(cfg_neg_one_padding)
        cfg_var_offsets.extend(cfg_int_neg_one_padding)
        cfg_data_binary_pos.extend(cfg_neg_one_padding)

        output = {
            "dfg_bert_input": dfg_bert_input,
            "dfg_bert_label": dfg_bert_label,
            "dfg_segment_label": dfg_segment_label,
            "dfg_is_next": d_label,
            "dfg_binary_pos": dfg_binary_pos,
            "dfg_function_pos": dfg_function_pos,
            "dfg_bb_pos": dfg_bb_pos,
            "dfg_var_offsets": dfg_var_offsets,
            "dfg_data_binary_pos": dfg_data_binary_pos,
            "cfg_bert_input": cfg_bert_input,
            "cfg_segment_label": cfg_segment_label,
            "cfg_is_next": c_label,
            "cfg_binary_pos": cfg_binary_pos,
            "cfg_function_pos": cfg_function_pos,
            "cfg_bb_pos": cfg_bb_pos,
            "cfg_var_offsets": cfg_var_offsets,
            "cfg_data_binary_pos": cfg_data_binary_pos
        }

        return {key: torch.tensor(value) for key, value in output.items()}


    def random_word(self, tokens, binary_pos, function_pos, bb_pos, var_offsets, data_binary_pos):
        """
        Apply random masking to tokens. When masking, also mask corresponding address info.
        """
        output_label = []
        masked_tokens = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    masked_tokens.append(self.vocab.mask_index)
                    # When masking, set address info to -1 (masked, since 0 has semantic meaning)
                    binary_pos[i] = -1.0
                    function_pos[i] = -1.0
                    bb_pos[i] = -1.0
                    var_offsets[i] = -1
                    data_binary_pos[i] = -1.0

                # 10% randomly change token to random token
                elif prob < 0.9:
                    masked_tokens.append(random.randrange(len(self.vocab)))
                    # Keep address info for random replacement

                # 10% keep current token
                else:
                    masked_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                masked_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_label.append(0)
        
        return masked_tokens, output_label, binary_pos, function_pos, bb_pos, var_offsets, data_binary_pos


    def random_sent(self, index):
        """Randomly pair sequences for NSP task."""
        c1, c2, d1, d2 = self.get_corpus_line(index)
        dice = random.random()
        if dice > 0.25:
            return c1, c2, 1, d1, d2, 1
        elif 0.25 <= dice < 0.5:
            return c1, self.get_random_line(), 0, d1, d2, 1
        elif 0.5 <= dice < 0.75:
            return c1, c2, 1, d2, d1, 0
        else:
            return c1, self.get_random_line(), 0, d2, d1, 0


    def get_corpus_line(self, item):
        """Get a pair of CFG and DFG sequences."""
        if self.on_memory:
            return self.cfg_lines[item][0], self.cfg_lines[item][1], self.dfg_lines[item][0], self.dfg_lines[item][1]


    def get_random_line(self):
        """Get a random CFG sequence for negative sampling."""
        if self.on_memory:
            l = self.cfg_lines[random.randrange(len(self.cfg_lines))]
            return l[1]
