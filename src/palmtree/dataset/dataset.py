from torch.utils.data import Dataset
import tqdm
import torch
import random
import pickle as pkl
import os

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
      - cfg_corpus_path:      CFG sentences (FIXED: CFG first)
      - dfg_corpus_path:      DFG sentences
      - cfg_src_path:         CFG segment/source labels   (per-token ints, space-separated)
      - dfg_src_path:         DFG segment/source labels   (per-token ints, space-separated)
      - cfg_tgt_path:         CFG target labels           (per-token ints, space-separated)
      - dfg_tgt_path:         DFG target labels           (per-token ints, space-separated)

    Notes:
      - By default, DFG "drives" the dataset length (drive_mode="dfg").
      - When DFG drives, CFG index is chosen by cycling or randomly (dfg_indexing).
      - `vocab` must provide: stoi (dict), pad_index, mask_index, unk_index, sos_index, eos_index, and __len__.
    """

    def __init__(
        self,
        cfg_corpus_path,
        dfg_corpus_path,
        cfg_src_path,
        dfg_src_path,
        cfg_tgt_path,
        dfg_tgt_path,
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

        # paths (FIXED: now matches parameter order CFG then DFG)
        self.corpus_lines = corpus_lines
        self.cfg_corpus_path = cfg_corpus_path
        self.dfg_corpus_path = dfg_corpus_path
        self.cfg_src_path = cfg_src_path
        self.dfg_src_path = dfg_src_path
        self.cfg_tgt_path = cfg_tgt_path
        self.dfg_tgt_path = dfg_tgt_path

        # ---------- loaders ----------
        self.dfg_lines = self._load_instruction_pairs(dfg_corpus_path, desc="Loading DFG")
        self.cfg_lines = self._load_instruction_pairs(cfg_corpus_path, desc="Loading CFG")

        # Track if we're using addresses or not
        self.use_addresses = (cfg_src_path and os.path.exists(cfg_src_path)) or \
                             (cfg_tgt_path and os.path.exists(cfg_tgt_path)) or \
                             (dfg_src_path and os.path.exists(dfg_src_path)) or \
                             (dfg_tgt_path and os.path.exists(dfg_tgt_path))
        
        # Load address files only if they exist
        if cfg_src_path and os.path.exists(cfg_src_path):
            self.cfg_src_lines = self._load_address_pairs(cfg_src_path, self.cfg_lines, desc="Loading CFG SRC")
        else:
            if cfg_src_path:
                print(f"CFG SRC file not found: {cfg_src_path}")
            self.cfg_src_lines = None
        
        if cfg_tgt_path and os.path.exists(cfg_tgt_path):
            self.cfg_tgt_lines = self._load_address_pairs(cfg_tgt_path, self.cfg_lines, desc="Loading CFG TGT")
        else:
            if cfg_tgt_path:
                print(f"CFG TGT file not found: {cfg_tgt_path}")
            self.cfg_tgt_lines = None

        if dfg_src_path and os.path.exists(dfg_src_path):
            self.dfg_src_lines = self._load_address_pairs(dfg_src_path, self.dfg_lines, desc="Loading DFG SRC")
        else:
            if dfg_src_path:
                print(f"DFG SRC file not found: {dfg_src_path}")
            self.dfg_src_lines = None
        
        if dfg_tgt_path and os.path.exists(dfg_tgt_path):
            self.dfg_tgt_lines = self._load_address_pairs(dfg_tgt_path, self.dfg_lines, desc="Loading DFG TGT")
        else:
            if dfg_tgt_path:
                print(f"DFG TGT file not found: {dfg_tgt_path}")
            self.dfg_tgt_lines = None

        # sizes (only consider address files if they're loaded)
        if self.cfg_src_lines and self.cfg_tgt_lines:
            self.n_cfg = min(len(self.cfg_lines), len(self.cfg_src_lines), len(self.cfg_tgt_lines))
        else:
            self.n_cfg = len(self.cfg_lines)
            
        if self.dfg_src_lines and self.dfg_tgt_lines:
            self.n_dfg = min(len(self.dfg_lines), len(self.dfg_src_lines), len(self.dfg_tgt_lines))
        else:
            self.n_dfg = len(self.dfg_lines)

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
    def _load_instruction_pairs(self, path, desc="Loading"):
        """Load a tab-separated file with 8 instructions per line into a list of 8 separate instructions."""
        lines = []
        with open(path, "r", encoding=self.encoding) as f:
            for line in tqdm.tqdm(f, desc=desc):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                
                # Expect 8 instructions per line
                if len(parts) < 8:
                    # Pad with empty strings if malformed
                    parts = parts + [""] * (8 - len(parts))
                
                # Keep all 8 instructions separate
                lines.append(parts[:8])
        return lines

    def _load_address_pairs(self, path, instruction_lines, desc="Loading"):
        """
        Load address file and split addresses to match instruction token counts.
        Each line has space-separated addresses for all tokens across 8 instructions.
        We split them to match each of the 8 instructions.
        """
        lines = []
        with open(path, "r", encoding=self.encoding) as f:
            for idx, line in enumerate(tqdm.tqdm(f, desc=desc)):
                line = line.rstrip("\n")
                if not line:
                    lines.append([""] * 8)
                    continue
                
                # All addresses are space-separated
                addresses = line.split()
                
                if idx < len(instruction_lines):
                    # Split addresses for each of the 8 instructions
                    addr_groups = []
                    addr_idx = 0
                    for ins in instruction_lines[idx]:
                        token_count = len(ins.split())
                        ins_addrs = " ".join(addresses[addr_idx:addr_idx + token_count])
                        addr_groups.append(ins_addrs)
                        addr_idx += token_count
                    lines.append(addr_groups)
                else:
                    # Fallback if instruction line doesn't exist - split evenly
                    chunk_size = len(addresses) // 8
                    addr_groups = []
                    for i in range(8):
                        start = i * chunk_size
                        end = start + chunk_size if i < 7 else len(addresses)
                        addr_groups.append(" ".join(addresses[start:end]))
                    lines.append(addr_groups)
        
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
        Return 8 instructions and their corresponding addresses:
          cfg_insts: list of 8 CFG instructions
          dfg_insts: list of 8 DFG instructions
          cfg_src: list of 8 CFG src address strings (or None if not available)
          cfg_tgt: list of 8 CFG tgt address strings (or None if not available)
          dfg_src: list of 8 DFG src address strings (or None if not available)
          dfg_tgt: list of 8 DFG tgt address strings (or None if not available)
        """
        i_cfg, i_dfg = self._map_indices(item)

        cfg_insts = self.cfg_lines[i_cfg]  # list of 8 instructions
        dfg_insts = self.dfg_lines[i_dfg]  # list of 8 instructions

        cfg_src = self.cfg_src_lines[i_cfg] if self.cfg_src_lines else None
        cfg_tgt = self.cfg_tgt_lines[i_cfg] if self.cfg_tgt_lines else None

        dfg_src = self.dfg_src_lines[i_dfg] if self.dfg_src_lines else None
        dfg_tgt = self.dfg_tgt_lines[i_dfg] if self.dfg_tgt_lines else None

        return cfg_insts, dfg_insts, cfg_src, cfg_tgt, dfg_src, dfg_tgt

    def get_random_line_not_idx(self, forbid_idx):
        """Pick a random CFG line whose index != forbid_idx; return all 8 instructions + addresses (or None)."""
        if self.n_cfg <= 1:
            # fallback: just return some line
            idx = 0
        else:
            while True:
                idx = random.randrange(self.n_cfg)
                if idx != forbid_idx:
                    break
        cfg_insts = self.cfg_lines[idx]
        cfg_src = self.cfg_src_lines[idx] if self.cfg_src_lines else None
        cfg_tgt = self.cfg_tgt_lines[idx] if self.cfg_tgt_lines else None
        return cfg_insts, cfg_src, cfg_tgt

    def get_random_line_dfg(self, forbid_idx):
        """Pick a random DFG line whose index != forbid_idx and content is different; return all 8 instructions + addresses (or None)."""
        if self.n_dfg <= 1:
            # fallback: just return some line
            idx = 0
        else:
            # Get the original content at forbid_idx to compare
            original_content = self.dfg_lines[forbid_idx] if forbid_idx < self.n_dfg else None
            
            max_attempts = 100  # prevent infinite loop
            attempts = 0
            while attempts < max_attempts:
                idx = random.randrange(self.n_dfg)
                # Ensure different index AND different content
                if idx != forbid_idx and self.dfg_lines[idx] != original_content:
                    break
                attempts += 1
            
            # If we couldn't find different content after max_attempts, just use different index
            if attempts >= max_attempts:
                while True:
                    idx = random.randrange(self.n_dfg)
                    if idx != forbid_idx:
                        break
        
        dfg_insts = self.dfg_lines[idx]
        dfg_src = self.dfg_src_lines[idx] if self.dfg_src_lines else None
        dfg_tgt = self.dfg_tgt_lines[idx] if self.dfg_tgt_lines else None
        return dfg_insts, dfg_src, dfg_tgt

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
        Returns 8 instructions for each stream with their addresses and labels.
        
        CFG: 4/4 split - tests if second half follows first half in correct order
        DFG: 7/1 split - tests if last instruction belongs to same trace as first 7
        """
        cfg_insts, dfg_insts, cfg_src, cfg_tgt, dfg_src, dfg_tgt = self.get_corpus_line(index)

        dice = random.random()
        if dice < 0.25:
            # both positives
            # CFG: correct order (1-4, then 5-8)
            # DFG: same trace (1-7, then 8)
            return cfg_insts, cfg_src, cfg_tgt, 1, dfg_insts, dfg_src, dfg_tgt, 1
        elif dice < 0.5:
            # negative CFG (swap order: 5-8 first, then 1-4), positive DFG
            cfg_swapped = cfg_insts[4:] + cfg_insts[:4]
            cfg_src_swapped = cfg_src[4:] + cfg_src[:4] if cfg_src else None
            cfg_tgt_swapped = cfg_tgt[4:] + cfg_tgt[:4] if cfg_tgt else None
            return cfg_swapped, cfg_src_swapped, cfg_tgt_swapped, 0, \
                   dfg_insts, dfg_src, dfg_tgt, 1
        elif dice < 0.75:
            # positive CFG, negative DFG (replace last instruction with random)
            dfg_rand, dfg_src_rand, dfg_tgt_rand = self.get_random_line_dfg(index)
            dfg_mixed = dfg_insts[:7] + [dfg_rand[7]]  # keep first 7, replace 8th
            dfg_src_mixed = dfg_src[:7] + [dfg_src_rand[7]] if dfg_src and dfg_src_rand else None
            dfg_tgt_mixed = dfg_tgt[:7] + [dfg_tgt_rand[7]] if dfg_tgt and dfg_tgt_rand else None
            return cfg_insts, cfg_src, cfg_tgt, 1, \
                   dfg_mixed, dfg_src_mixed, dfg_tgt_mixed, 0
        else:
            # both negative
            cfg_swapped = cfg_insts[4:] + cfg_insts[:4]
            cfg_src_swapped = cfg_src[4:] + cfg_src[:4] if cfg_src else None
            cfg_tgt_swapped = cfg_tgt[4:] + cfg_tgt[:4] if cfg_tgt else None
            
            dfg_rand, dfg_src_rand, dfg_tgt_rand = self.get_random_line_dfg(index)
            dfg_mixed = dfg_insts[:7] + [dfg_rand[7]]
            dfg_src_mixed = dfg_src[:7] + [dfg_src_rand[7]] if dfg_src and dfg_src_rand else None
            dfg_tgt_mixed = dfg_tgt[:7] + [dfg_tgt_rand[7]] if dfg_tgt and dfg_tgt_rand else None
            
            return cfg_swapped, cfg_src_swapped, cfg_tgt_swapped, 0, \
                   dfg_mixed, dfg_src_mixed, dfg_tgt_mixed, 0

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

        # Compose positive/negative sample - now returns 8 instructions each
        cfg_insts, cfg_src, cfg_tgt, c_label, dfg_insts, dfg_src, dfg_tgt, d_label = self.random_sent(item)

        # Build CFG sequence: <cls> ins1 <sep> ins2 <sep> ... ins8 <sep>
        # Build DFG sequence: <cls> ins1 <sep> ins2 <sep> ... ins8 <sep>
        cfg_tokens = [self.vocab.sos_index]  # <cls>
        cfg_addrs_src = [0] if cfg_src else []  # dummy for <cls> or empty if no addresses
        cfg_addrs_tgt = [0] if cfg_tgt else []  # dummy for <cls> or empty if no addresses
        cfg_mask_labels = [self.vocab.pad_index]  # no mask for <cls>
        
        for i, ins in enumerate(cfg_insts):
            # Apply masking to each instruction
            ins_masked, ins_label = self.random_word(ins)
            cfg_tokens.extend(ins_masked)
            cfg_mask_labels.extend(ins_label)
            
            # Add addresses for this instruction (only if available)
            if cfg_src:
                ins_addrs_src = self._to_int_list(cfg_src[i])
                cfg_addrs_src.extend(ins_addrs_src)
            if cfg_tgt:
                ins_addrs_tgt = self._to_int_list(cfg_tgt[i])
                cfg_addrs_tgt.extend(ins_addrs_tgt)
            
            # Add <sep> token
            cfg_tokens.append(self.vocab.eos_index)
            cfg_mask_labels.append(self.vocab.pad_index)
            if cfg_src:
                cfg_addrs_src.append(0)
            if cfg_tgt:
                cfg_addrs_tgt.append(0)
        
        # Build DFG sequence (no masking)
        dfg_tokens = [self.vocab.sos_index]  # <cls>
        dfg_addrs_src = [0] if dfg_src else []  # dummy for <cls> or empty if no addresses
        dfg_addrs_tgt = [0] if dfg_tgt else []  # dummy for <cls> or empty if no addresses
        
        for i, ins in enumerate(dfg_insts):
            # No masking for DFG
            ins_tokens = self._to_ids(ins)
            dfg_tokens.extend(ins_tokens)
            
            # Add addresses for this instruction (only if available)
            if dfg_src:
                ins_addrs_src = self._to_int_list(dfg_src[i])
                dfg_addrs_src.extend(ins_addrs_src)
            if dfg_tgt:
                ins_addrs_tgt = self._to_int_list(dfg_tgt[i])
                dfg_addrs_tgt.extend(ins_addrs_tgt)
            
            # Add <sep> token
            dfg_tokens.append(self.vocab.eos_index)
            if dfg_src:
                dfg_addrs_src.append(0)
            if dfg_tgt:
                dfg_addrs_tgt.append(0)

        # Truncate to seq_len
        cfg_bert_input = cfg_tokens[:self.seq_len]
        cfg_bert_label = cfg_mask_labels[:self.seq_len]
        cfg_segment_label = cfg_addrs_src[:self.seq_len] if cfg_src else []
        cfg_tgt_label = cfg_addrs_tgt[:self.seq_len] if cfg_tgt else []
        
        dfg_bert_input = dfg_tokens[:self.seq_len]
        dfg_segment_label = dfg_addrs_src[:self.seq_len] if dfg_src else []
        dfg_tgt_label = dfg_addrs_tgt[:self.seq_len] if dfg_tgt else []

        # ---- Pad EVERYTHING to self.seq_len so collate can stack ----
        pad = self.vocab.pad_index
        L = self.seq_len

        cfg_bert_input    = _pad_to(cfg_bert_input,    L, pad)
        cfg_bert_label    = _pad_to(cfg_bert_label,    L, pad)
        # If no addresses, fill with pad
        cfg_segment_label = _pad_to(cfg_segment_label, L, pad) if cfg_src else [pad] * L
        cfg_tgt_label     = _pad_to(cfg_tgt_label,     L, pad) if cfg_tgt else [pad] * L

        dfg_bert_input    = _pad_to(dfg_bert_input,    L, pad)
        # If no addresses, fill with pad
        dfg_segment_label = _pad_to(dfg_segment_label, L, pad) if dfg_src else [pad] * L
        dfg_tgt_label     = _pad_to(dfg_tgt_label,     L, pad) if dfg_tgt else [pad] * L


        output = {
            "cfg_bert_input": cfg_bert_input,
            "cfg_bert_label": cfg_bert_label,
            "cfg_segment_label": cfg_segment_label,
            "cfg_tgt_label": cfg_tgt_label,
            "cfg_is_next": c_label,  # scalar (0/1)
            "dfg_bert_input": dfg_bert_input,
            "dfg_segment_label": dfg_segment_label, # src addr
            "dfg_tgt_label": dfg_tgt_label,
            "dfg_is_next": d_label,  # scalar (0/1)
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