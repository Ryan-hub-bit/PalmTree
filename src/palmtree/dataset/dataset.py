from torch.utils.data import Dataset
import tqdm
import torch
import random
import pickle as pkl


# class BERTDataset(Dataset):
#     def __init__(self, dfg_corpus_path, cfg_corpus_path, dfg_src_path, cfg_src_path, dfg_tgt_path, cfg_tgt_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
#         self.vocab = vocab
#         self.seq_len = seq_len

#         self.bb_len = 50

#         self.on_memory = on_memory
#         self.corpus_lines = corpus_lines
#         self.dfg_corpus_path = dfg_corpus_path
#         self.cfg_corpus_path = cfg_corpus_path
#         self.dfg_src_path = dfg_src_path
#         self.cfg_src_path = cfg_src_path
#         self.dfg_tgt_path = dfg_tgt_path
#         self.cfg_tgt_path = cfg_tgt_path
#         self.encoding = encoding

#         # load DFG sequences 
#         with open(dfg_corpus_path, "r", encoding=encoding) as f:
#             if self.corpus_lines is None and not on_memory:
#                 for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
#                     self.corpus_lines += 1

#             if on_memory:
#                 self.dfg_lines = [line[:-1].split("\t")
#                               for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
#                 self.corpus_lines = len(self.dfg_lines)
       
#        # load CFG sequences 
#         with open(cfg_corpus_path, "r", encoding=encoding) as f:
#             if self.corpus_lines is None and not on_memory:
#                 for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
#                     self.corpus_lines += 1

#             if on_memory:
#                 self.cfg_lines = [line[:-1].split("\t")
#                               for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
#                 if self.corpus_lines > len(self.cfg_lines):    
#                     self.corpus_lines = len(self.cfg_lines)
        
#         with  open(cfg_src_path, "r", encoding=encoding) as f:
#             if self.cfg_src_path is None and not on_memory:
#                 for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
#                     self.corpus_lines += 1

#             if on_memory:
#                 self.cfg_src_lines = [line[:-1].split("\t")
#                               for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
#                 if self.corpus_lines > len(self.cfg_lines):    
#                     self.corpus_lines = len(self.cfg_lines)
            
#         with  open(dfg_src_path, "r", encoding=encoding) as f:
#             if self.dfg_src_path is None and not on_memory:
#                 for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
#                     self.corpus_lines += 1

#             if on_memory:
#                 self.dfg_src_lines = [line[:-1].split("\t")
#                               for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
#                 if self.corpus_lines > len(self.cfg_lines):    
#                     self.corpus_lines = len(self.cfg_lines)
    
#         with  open(dfg_tgt_path, "r", encoding=encoding) as f:
#             if self.dfg_tgt_path is None and not on_memory:
#                 for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
#                     self.corpus_lines += 1

#             if on_memory:
#                 self.dfg_tgt_lines = [line[:-1].split("\t")
#                               for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
#                 if self.corpus_lines > len(self.dfg_lines):    
#                     self.corpus_lines = len(self.dfg_lines)
            
#         with  open(cfg_tgt_path, "r", encoding=encoding) as f:
#             if self.cfg_tgt_path is None and not on_memory:
#                 for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
#                     self.corpus_lines += 1

#             if on_memory:
#                 self.cfg_tgt_lines = [line[:-1].split("\t")
#                               for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
#                 if self.corpus_lines > len(self.cfg_lines):    
#                     self.corpus_lines = len(self.cfg_lines)
        


#         # if not on_memory:
#         #     self.file = open(corpus_path, "r", encoding=encoding)
#         #     self.random_file = open(corpus_path, "r", encoding=encoding)

#         #     for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
#         #         self.random_file.__next__()

#     #? question corpus_lines are the smaller one between CFG_line and DFG_line
#     def __len__(self):
#         return self.corpus_lines


#     def __getitem__(self, item):
#         c1, c2, cs1, cs2, cd1, cd2, c_label, d1, d2, ds1, ds2, dd1, dd2, d_label = self.random_sent(item)

#         d1_random, d1_label = self.random_word(d1)
#         d2_random, d2_label = self.random_word(d2)

#         d1 = [self.vocab.sos_index] + d1_random + [self.vocab.eos_index]
#         d2 = d2_random + [self.vocab.eos_index]

#         c1 = [self.vocab.sos_index] + [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c1.split()] + [self.vocab.eos_index]
#         c2 = [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c2.split()] + [self.vocab.eos_index]
        


#         d1_label = [self.vocab.pad_index] + d1_label + [self.vocab.pad_index]
#         d2_label = d2_label + [self.vocab.pad_index]

#         # dfg_segment_label = ([1 for _ in range(len(d1))] + [2 for _ in range(len(d2))])[:self.seq_len]
#         # cfg_segment_label = ([1 for _ in range(len(c1))] + [2 for _ in range(len(c2))])[:self.seq_len]
#         dfg_segment_label = (ds1 + ds2)[:self.seq_len]
#         cfg_segment_label = (cs1 + cs2)[:self.seq_len]

        
#         cfg_tgt_label = (dd1 + dd2)[:self.seq_len]
#         dfg_tgt_label = (cd1 + cd2)[:self.seq_len]
        
#         dfg_bert_input = (d1 + d2)[:self.seq_len]
#         dfg_bert_label = (d1_label + d2_label)[:self.seq_len]

#         cfg_bert_input = (c1 + c2)[:self.seq_len]

#         padding = [self.vocab.pad_index for _ in range(self.seq_len - len(dfg_bert_input))]
#         dfg_bert_input.extend(padding), dfg_bert_label.extend(padding), dfg_segment_label.extend(padding) #, comp_label.extend(padding)
#         cfg_padding = [self.vocab.pad_index for _ in range(self.seq_len - len(cfg_bert_input))]
#         cfg_bert_input.extend(cfg_padding), cfg_segment_label.extend(cfg_padding)

#         output = {"dfg_bert_input": dfg_bert_input,
#                   "dfg_bert_label": dfg_bert_label,
#                   "dfg_segment_label": dfg_segment_label,
#                   "dfg_tgt_label": dfg_tgt_label,
#                   "dfg_is_next": d_label,
#                   "cfg_bert_input": cfg_bert_input,
#                   "cfg_segment_label": cfg_segment_label,
#                   "cfg_tgt_label": cfg_tgt_label,
#                   "cfg_is_next": c_label
#                   }

#         return {key: torch.tensor(value) for key, value in output.items()}


#     def random_bb(self):
#         prob = random.random()
#         if prob > 0.5:
#             bb_pair = self.bb_pairs[random.choice(list(self.bb_pairs.keys()))]
#             return bb_pair, 1
#         else:
#             neg_keys = random.choices(list(self.bb_pairs.keys()), k=2)
#             bb_pair = (self.bb_pairs[neg_keys[0]][0], self.bb_pairs[neg_keys[1]][1])
#             return bb_pair, 0 


#     def get_index_bb(self, bb_pair):
#         tokens1 = [self.vocab.sos_index]
#         segment1 = [1]
#         i = 1
#         for ins in bb_pair[0].split(";")[-5:]:
#             if ins:
#                 for token in ins.split():
#                     tokens1.append(self.vocab.stoi.get(token, self.vocab.unk_index))
#                     segment1.append(i)
#                 tokens1.append(self.vocab.eos_index)
#                 segment1.append(i)
#                 i += 1
         
#         tokens2 = [self.vocab.sos_index]
#         segment2 = [1]
#         j = 1
#         for ins in bb_pair[0].split(";")[-5:]:
#             if ins:
#                 for token in ins.split():
#                     tokens2.append(self.vocab.stoi.get(token, self.vocab.unk_index))
#                     segment2.append(j)
#                 tokens2.append(self.vocab.eos_index)
#                 segment2.append(j)
#                 j += 1

#         tokens1 = tokens1[:self.bb_len]
#         tokens2 = tokens2[:self.bb_len]

#         segment1 = segment1[:self.bb_len]
#         segment2 = segment2[:self.bb_len]

#         padding1 = [self.vocab.pad_index for _ in range(self.bb_len - len(tokens1))]
#         padding2 = [self.vocab.pad_index for _ in range(self.bb_len - len(tokens2))]

#         tokens1.extend(padding1)
#         tokens2.extend(padding2)

#         segment1.extend(padding1)
#         segment2.extend(padding2)

#         return tokens1, tokens2, segment1, segment2
     

#     def random_word(self, sentence):
#         tokens = sentence.split()
#         output_label = []

#         for i, token in enumerate(tokens):
#             prob = random.random()
#             if prob < 0.15:
#                 prob /= 0.15

#                 # 80% randomly change token to mask token
#                 if prob < 0.8:
#                     tokens[i] = self.vocab.mask_index

#                 # 10% randomly change token to random token
#                 elif prob < 0.9:
#                     tokens[i] = random.randrange(len(self.vocab))

#                 # 10% randomly change token to current token
#                 else:
#                     tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

#                 output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

#             else:
#                 tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
#                 output_label.append(0)
        
#         return tokens, output_label


#     def random_sent(self, index):
#         c1, c2, d1, d2, cs1, cs2, cd1, cd2, ds1, ds2, dd1, dd2 = self.get_corpus_line(index)
#         #? Question here......  should make sure the get_random_line's result is not the same as the correct answer 
#         dice = random.random() # TODO: should throw the dice twice here. 
#         if dice < 0.25:
#             return c1, c2, cs1, cs2, cd1, cd2, 1, d1, d2, ds1, ds2, dd1, dd2, 1
#         elif 0.25 <= dice < 0.5:
#             cc2, ccs2, ccd2 = self.get_random_line() 
#             return c1, cc2,cs1, ccs2, cd1, ccd2, 0, d1, d2, ds1, ds2, dd1, dd2, 1
#         elif 0.5 <= dice < 0.75:
#             return c1, c2,cs1, cs2, cd1, cd2, 1, d2, d1, ds2, ds1,dd2, dd1, 0
#         else:
#             cc2, ccs2, ccd2 = self.get_random_line()
#             return c1, cc2, cs1, ccs2, cd1, ccd2,  0, d2, d1, ds2, ds1, dd2, dd1,0


#     def get_corpus_line(self, item):
#         if self.on_memory:
#             #line dfg > line cfg
#             return self.cfg_lines[item][0], self.cfg_lines[item][1], self.dfg_lines[item][0], self.dfg_lines[item][1], self.cfg_src_lines[item][0], self.cfg_src_lines[item][1], self.cfg_tgt_lines[item][0], self.cfg_tgt_lines[item][1],self.dfg_src_lines[item][0], self.dfg_src_lines[item][1], self.dfg_tgt_lines[item][0], self.dfg_tgt_lines[item][1]

#         # now only on_memory copurs are supported
#         # else:
#         #     line = self.file.__next__()
#         #     if line is None:
#         #         self.file.close()
#         #         self.file = open(self.corpus_path, "r", encoding=self.encoding)
#         #         line = self.file.__next__()

#         #     t1, t2 = line[:-1].split("\t")
#         #     return t1, t2 


#     # def get_random_line(self):
#     #     if self.on_memory:
#     #         l = self.cfg_lines[random.randrange(len(self.cfg_lines))]
#     #         return l[1]
#     def get_random_line(self, idx):
#         """Return the line (and optionally the index) at given idx."""
#         if self.on_memory:
#             idx = random.randrange(len(self.cfg_lines))
#             l = self.cfg_lines[idx]
#             s = self.cfg_src_lines[idx]
#             d = self.cfg_tgt_lines[idx]
#             return l[1], s[1], d[1]

#         # now only on_memory copurs are supported
#         # line = self.file.__next__()
#         # if line is None:
#         #     self.file.close()
#         #     self.file = open(self.corpus_path, "r", encoding=self.encoding)
#         #     for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
#         #         self.random_file.__next__()
#         #     line = self.random_file.__next__()
#         # return line[:-1].split("\t")[1] 

# ===========================================
# Address conversion helpers (global scope)
# ===========================================
MASK64   = (1 << 64) - 1
INT64_MAX = (1 << 63) - 1

def uint64_to_signed_int64(u: int) -> int:
    u &= MASK64
    return u - (1 << 64) if u > INT64_MAX else u

def signed_int64_to_uint64(s: int) -> int:
    return s & MASK64


class BERTDataset(Dataset):
    """
    Paired CFG/DFG pretraining dataset.

    Expected files (each line is "LEFT<TAB>RIGHT"):
      - dfg_corpus_path:      DFG sentences
      - cfg_corpus_path:      CFG sentences
      - dfg_src_path:         DFG segment/source labels   (per-token ints, space-separated)
      - cfg_src_path:         CFG segment/source labels   (per-token ints, space-separated)
      - dfg_tgt_path:         DFG target labels           (per-token ints, space-separated)
      - cfg_tgt_path:         CFG target labels           (per-token ints, space-separated)

    Notes:
      - By default, DFG "drives" the dataset length (drive_mode="dfg").
      - When DFG drives, CFG index is chosen by cycling or randomly (dfg_indexing).
      - `vocab` must provide: stoi (dict), pad_index, mask_index, unk_index, sos_index, eos_index, and __len__.
    """

    def __init__(
        self,
        dfg_corpus_path,
        cfg_corpus_path,
        dfg_src_path,
        cfg_src_path,
        dfg_tgt_path,
        cfg_tgt_path,
        vocab,
        seq_len,
        encoding="utf-8",
        corpus_lines=None,
        on_memory=True,
        drive_mode="dfg",          # "dfg", "cfg", "min", "max"
        dfg_indexing="cycle"       # "cycle" or "random" (when DFG drives)
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.bb_len = 50  # (kept from your original, used by get_index_bb/random_bb if you need them)
        self.on_memory = on_memory
        self.encoding = encoding

        self.drive_mode = drive_mode
        self.dfg_indexing = dfg_indexing

        # paths
        self.corpus_lines = corpus_lines
        self.dfg_corpus_path = dfg_corpus_path
        self.cfg_corpus_path = cfg_corpus_path
        self.dfg_src_path = dfg_src_path
        self.cfg_src_path = cfg_src_path
        self.dfg_tgt_path = dfg_tgt_path
        self.cfg_tgt_path = cfg_tgt_path

        # ---------- loaders ----------
        self.dfg_lines = self._load_pairs(dfg_corpus_path, desc="Loading DFG")
        self.cfg_lines = self._load_pairs(cfg_corpus_path, desc="Loading CFG")

        self.cfg_src_lines = self._load_pairs(cfg_src_path, desc="Loading CFG SRC")
        self.cfg_tgt_lines = self._load_pairs(cfg_tgt_path, desc="Loading CFG TGT")

        self.dfg_src_lines = self._load_pairs(dfg_src_path, desc="Loading DFG SRC")
        self.dfg_tgt_lines = self._load_pairs(dfg_tgt_path, desc="Loading DFG TGT")

        # sizes (make sure aux files aren't shorter than corpus)
        self.n_cfg = min(len(self.cfg_lines), len(self.cfg_src_lines), len(self.cfg_tgt_lines))
        self.n_dfg = min(len(self.dfg_lines), len(self.dfg_src_lines), len(self.dfg_tgt_lines))

        # ---------- decide dataset length ----------
        if drive_mode == "cfg":
            self._N = self.n_cfg
        elif drive_mode == "dfg":
            self._N = self.n_dfg
        elif drive_mode == "min":
            self._N = min(self.n_cfg, self.n_dfg)
        elif drive_mode == "max":
            self._N = max(self.n_cfg, self.n_dfg)
        else:
            raise ValueError(f"Unknown drive_mode: {drive_mode}")

        if self._N == 0:
            raise RuntimeError("Empty dataset after size checks; please verify your input files.")

    # -------- utilities --------
    def _load_pairs(self, path, desc="Loading"):
        """Load a tab-separated 'left<TAB>right' file into a list of [left, right]."""
        lines = []
        with open(path, "r", encoding=self.encoding) as f:
            for line in tqdm.tqdm(f, desc=desc):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    # pad if malformed
                    parts = [parts[0], ""]
                lines.append(parts[:2])
        return lines

    def __len__(self):
        return self._N

    def _map_indices(self, item: int):
        """
        Map the global dataset index to (i_cfg, i_dfg) based on drive_mode and dfg_indexing.
        """
        if self.drive_mode == "cfg":
            if self.n_cfg == 0 or self.n_dfg == 0:
                raise RuntimeError("CFG/DFG empty but required.")
            i_cfg = item % self.n_cfg
            i_dfg = item % self.n_dfg  # symmetric choice
        elif self.drive_mode == "dfg":
            if self.n_cfg == 0 or self.n_dfg == 0:
                raise RuntimeError("CFG/DFG empty but required.")
            i_dfg = item % self.n_dfg
            if self.dfg_indexing == "cycle":
                i_cfg = item % self.n_cfg
            elif self.dfg_indexing == "random":
                i_cfg = random.randrange(self.n_cfg)
            else:
                raise ValueError(f"Unknown dfg_indexing: {self.dfg_indexing}")
        elif self.drive_mode == "min":
            i_cfg = item % self.n_cfg
            i_dfg = item % self.n_dfg
        elif self.drive_mode == "max":
            i_cfg = item % self.n_cfg if self.n_cfg else 0
            i_dfg = item % self.n_dfg if self.n_dfg else 0
        else:
            raise ValueError(f"Unknown drive_mode: {self.drive_mode}")
        return i_cfg, i_dfg

    # ---------- parsing helpers ----------
    def _to_ids(self, sentence: str):
        """Map whitespace tokens to vocab ids (no SOS/EOS)."""
        stoi = self.vocab.stoi
        unk = self.vocab.unk_index
        return [stoi.get(t, unk) for t in sentence.split()]

    # def _to_int_list(self, s: str):
    #     """Parse a space-separated string of ints into a list of ints."""
    #     if not s:
    #         return []
    #     return [int(x, 16) for x in s.split()]

    # _to_int_list: parse tokens as unsigned then wrap to signed for tensors
    def _to_int_list(self, s: str):
        if not s:
            return []
        out = []
        for tok in s.split():
            try:
                u = int(tok, 0) & MASK64         # parse (0x... or decimal) → unsigned range
                out.append(uint64_to_signed_int64(u))  # store as signed int64 value
            except ValueError:
                # handle malformed token (skip or raise)
                continue
        return out

    # When building tensors (line ~660)



    # ---------- main fetchers ----------
    def get_corpus_line(self, item):
        """
        Return:
          c1, c2,
          d1, d2,
          cs1, cs2,  (CFG src/segment labels as space-separated ints)
          cd1, cd2,  (CFG target labels as space-separated ints)
          ds1, ds2,  (DFG src/segment labels)
          dd1, dd2   (DFG target labels)
        """
        i_cfg, i_dfg = self._map_indices(item)

        c1, c2 = self.cfg_lines[i_cfg]
        d1, d2 = self.dfg_lines[i_dfg]

        cs1, cs2 = self.cfg_src_lines[i_cfg]
        cd1, cd2 = self.cfg_tgt_lines[i_cfg]

        ds1, ds2 = self.dfg_src_lines[i_dfg]
        dd1, dd2 = self.dfg_tgt_lines[i_dfg]

        return c1, c2, d1, d2, cs1, cs2, cd1, cd2, ds1, ds2, dd1, dd2

    def get_random_line_not_idx(self, forbid_idx):
        """Pick a random CFG line whose index != forbid_idx; return right side + src/tgt right labels."""
        if self.n_cfg <= 1:
            # fallback: just return some line
            idx = 0
        else:
            while True:
                idx = random.randrange(self.n_cfg)
                if idx != forbid_idx:
                    break
        l = self.cfg_lines[idx]
        s = self.cfg_src_lines[idx]
        d = self.cfg_tgt_lines[idx]
        return l[1], s[1], d[1]

    def random_word(self, sentence):
        """
        BERT-style masking on a whitespace tokenized sentence.
        Returns (masked_token_ids, label_ids) where label_ids are 0 for unmasked positions.
        """
        tokens = sentence.split()
        output_label = []
        stoi = self.vocab.stoi
        unk = self.vocab.unk_index

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% -> [MASK]
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10% -> random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                # 10% -> keep original id
                else:
                    tokens[i] = stoi.get(token, unk)
                output_label.append(stoi.get(token, unk))
            else:
                tokens[i] = stoi.get(token, unk)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        """
        Create NSP-style variants by mixing CFG/DFG orders or negative pairs.
        Returns:
          c1, c2, cs1, cs2, cd1, cd2, c_label,
          d1, d2, ds1, ds2, dd1, dd2, d_label
        where labels are 1 for positive, 0 for negative.
        """
        c1, c2, d1, d2, cs1, cs2, cd1, cd2, ds1, ds2, dd1, dd2 = self.get_corpus_line(index)

        dice = random.random()
        if dice < 0.25:
            # both positives
            return c1, c2, cs1, cs2, cd1, cd2, 1, d1, d2, ds1, ds2, dd1, dd2, 1
        elif dice < 0.5:
            # negative CFG right
            cc2, ccs2, ccd2 = self.get_random_line_not_idx(index)
            return c1, cc2, cs1, ccs2, cd1, ccd2, 0, d1, d2, ds1, ds2, dd1, dd2, 1
        elif dice < 0.75:
            # negative DFG order (swap)
            return c1, c2, cs1, cs2, cd1, cd2, 1, d2, d1, ds2, ds1, dd2, dd1, 0
        else:
            # negative CFG and negative DFG order
            cc2, ccs2, ccd2 = self.get_random_line_not_idx(index)
            return c1, cc2, cs1, ccs2, cd1, ccd2, 0, d2, d1, ds2, ds1, dd2, dd1, 0

    # ---------- optional BB helpers from your original (kept intact) ----------
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

        padding1 = [self.vocab.pad_index] * (self.bb_len - len(tokens1))
        padding2 = [self.vocab.pad_index] * (self.bb_len - len(tokens2))

        tokens1.extend(padding1)
        tokens2.extend(padding2)
        segment1.extend(padding1)
        segment2.extend(padding2)

        return tokens1, tokens2, segment1, segment2
    
    def __getitem__(self, item):
        # -------- local helper --------
        def _pad_to(xs, n, pad):
            if len(xs) < n:
                xs = xs + [pad] * (n - len(xs))
            else:
                xs = xs[:n]
            return xs

        # Compose positive/negative sample
        c1, c2, cs1, cs2, cd1, cd2, c_label, d1, d2, ds1, ds2, dd1, dd2, d_label = self.random_sent(item)

        # Masking for DFG sentence pair
        d1_masked, d1_label = self.random_word(d1)
        d2_masked, d2_label = self.random_word(d2)

        # Add SOS/EOS to inputs
        dfg_left  = [self.vocab.sos_index] + d1_masked + [self.vocab.eos_index]
        dfg_right = d2_masked + [self.vocab.eos_index]

        cfg_left  = [self.vocab.sos_index] + self._to_ids(c1) + [self.vocab.eos_index]
        cfg_right = self._to_ids(c2) + [self.vocab.eos_index]

        # Mask labels (0 for non-masked positions); pad at ends to match SOS/EOS
        d1_label = [self.vocab.pad_index] + d1_label + [self.vocab.pad_index]
        d2_label = d2_label + [self.vocab.pad_index]

        # Segment/target labels -> lists of ints (your _to_int_list already does unsigned→signed64 wrap)
        cs1 = self._to_int_list(cs1); cs2 = self._to_int_list(cs2)
        cd1 = self._to_int_list(cd1); cd2 = self._to_int_list(cd2)

        ds1 = self._to_int_list(ds1); ds2 = self._to_int_list(ds2)
        dd1 = self._to_int_list(dd1); dd2 = self._to_int_list(dd2)

        # Concatenate then truncate to seq_len (we'll pad to exact length below)
        dfg_segment_label = (ds1 + ds2)[:self.seq_len]
        cfg_segment_label = (cs1 + cs2)[:self.seq_len]

        cfg_tgt_label = (cd1 + cd2)[:self.seq_len]
        dfg_tgt_label = (dd1 + dd2)[:self.seq_len]

        # Truncate BERT inputs (we'll pad to exact length below)
        dfg_bert_input = (dfg_left + dfg_right)[:self.seq_len]
        dfg_bert_label = (d1_label + d2_label)[:self.seq_len]
        cfg_bert_input = (cfg_left + cfg_right)[:self.seq_len]

        # ---- Pad EVERYTHING to self.seq_len so collate can stack ----
        pad = self.vocab.pad_index
        L = self.seq_len

        dfg_bert_input    = _pad_to(dfg_bert_input,    L, pad)
        dfg_bert_label    = _pad_to(dfg_bert_label,    L, pad)
        dfg_segment_label = _pad_to(dfg_segment_label, L, pad)
        dfg_tgt_label     = _pad_to(dfg_tgt_label,     L, pad)

        cfg_bert_input    = _pad_to(cfg_bert_input,    L, pad)
        cfg_segment_label = _pad_to(cfg_segment_label, L, pad)
        cfg_tgt_label     = _pad_to(cfg_tgt_label,     L, pad)

        # (Optional) strict checks during debugging
        # for key, arr in {
        #     "dfg_bert_input": dfg_bert_input,
        #     "dfg_bert_label": dfg_bert_label,
        #     "dfg_segment_label": dfg_segment_label,
        #     "dfg_tgt_label": dfg_tgt_label,
        #     "cfg_bert_input": cfg_bert_input,
        #     "cfg_segment_label": cfg_segment_label,
        #     "cfg_tgt_label": cfg_tgt_label,
        # }.items():
        #     assert len(arr) == L, f"{key} len={len(arr)} != {L}"

        output = {
            "dfg_bert_input": dfg_bert_input,
            "dfg_bert_label": dfg_bert_label,
            "dfg_segment_label": dfg_segment_label,
            "dfg_tgt_label": dfg_tgt_label,
            "dfg_is_next": d_label,  # scalar (0/1)

            "cfg_bert_input": cfg_bert_input,
            "cfg_segment_label": cfg_segment_label,
            "cfg_tgt_label": cfg_tgt_label,
            "cfg_is_next": c_label,  # scalar (0/1)
        }

        # Tensorize (keep long for indices/labels)
        try:
            return {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        except Exception as e:
            print("\n[DEBUG] Tensorization failed:", e)
            INT64_MAX = (1 << 63) - 1
            INT64_MIN = -(1 << 63)
            for k, v in output.items():
                if isinstance(v, (list, tuple)):
                    bad = [x for x in v if isinstance(x, int) and (x > INT64_MAX or x < INT64_MIN)]
                    if bad:
                        print(f"{k} overflow values:", bad[:8], ("...(+more)" if len(bad) > 8 else ""))
                    if len(v) != L:
                        print(f"{k} length {len(v)} != {L}")
                elif isinstance(v, int):
                    if v > INT64_MAX or v < INT64_MIN:
                        print(f"{k} scalar overflow: {v}")
            print("[END DEBUG]\n")
            raise
