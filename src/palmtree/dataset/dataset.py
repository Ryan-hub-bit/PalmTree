from torch.utils.data import Dataset
import tqdm
import torch
import random
import pickle as pkl

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

    def get_random_line_dfg(self, forbid_idx):
        """Pick a random DFG line whose index != forbid_idx and content is different; return right side + src/tgt right labels."""
        if self.n_dfg <= 1:
            # fallback: just return some line
            idx = 0
        else:
            # Get the original content at forbid_idx to compare
            original_content = self.dfg_lines[forbid_idx][1] if forbid_idx < self.n_dfg else None
            
            max_attempts = 100  # prevent infinite loop
            attempts = 0
            while attempts < max_attempts:
                idx = random.randrange(self.n_dfg)
                # Ensure different index AND different content
                if idx != forbid_idx and self.dfg_lines[idx][1] != original_content:
                    break
                attempts += 1
            
            # If we couldn't find different content after max_attempts, just use different index
            if attempts >= max_attempts:
                while True:
                    idx = random.randrange(self.n_dfg)
                    if idx != forbid_idx:
                        break
        
        l = self.dfg_lines[idx]
        s = self.dfg_src_lines[idx]
        d = self.dfg_tgt_lines[idx]
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
        
        CFG negatives: swap order (c2, c1)
        DFG negatives: random line replacement
        """
        c1, c2, d1, d2, cs1, cs2, cd1, cd2, ds1, ds2, dd1, dd2 = self.get_corpus_line(index)

        dice = random.random()
        if dice < 0.25:
            # both positives
            return c1, c2, cs1, cs2, cd1, cd2, 1, d1, d2, ds1, ds2, dd1, dd2, 1
        elif dice < 0.5:
            # negative CFG (swap order)
            return c2, c1, cs2, cs1, cd2, cd1, 0, d1, d2, ds1, ds2, dd1, dd2, 1
        elif dice < 0.75:
            # negative DFG (random line)
            dd2_rand, dds2_rand, ddd2_rand = self.get_random_line_dfg(index)
            return c1, c2, cs1, cs2, cd1, cd2, 1, d1, dd2_rand, ds1, dds2_rand, dd1, ddd2_rand, 0
        else:
            # negative CFG (swap) and negative DFG (random)
            dd2_rand, dds2_rand, ddd2_rand = self.get_random_line_dfg(index)
            return c2, c1, cs2, cs1, cd2, cd1, 0, d1, dd2_rand, ds1, dds2_rand, dd1, ddd2_rand, 0

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

        # Masking for CFG sentence pair
        c1_masked, c1_label = self.random_word(c1)
        c2_masked, c2_label = self.random_word(c2)

        # Add SOS/EOS to inputs
        cfg_left  = [self.vocab.sos_index] + c1_masked + [self.vocab.eos_index]
        cfg_right = c2_masked + [self.vocab.eos_index]

        dfg_left  = [self.vocab.sos_index] + self._to_ids(d1) + [self.vocab.eos_index]
        dfg_right = self._to_ids(d2) + [self.vocab.eos_index]

        # Mask labels (0 for non-masked positions); pad at ends to match SOS/EOS
        c1_label = [self.vocab.pad_index] + c1_label + [self.vocab.pad_index]
        c2_label = c2_label + [self.vocab.pad_index]

        # Segment/target labels -> lists of ints (your _to_int_list already does unsigned→signed64 wrap)
        cs1 = self._to_int_list(cs1); cs2 = self._to_int_list(cs2)
        cd1 = [0] + self._to_int_list(cd1) + [0]; cd2 = self._to_int_list(cd2)  + [0]

        ds1 = self._to_int_list(ds1); ds2 = self._to_int_list(ds2)
        dd1 = [0] + self._to_int_list(dd1) + [0]; dd2 = self._to_int_list(dd2) + [0]

        # Concatenate then truncate to seq_len (we'll pad to exact length below)
        # dfg_segment_label = (ds1 + ds2)[:self.seq_len]
        dfg_segment_label = ([ds1[0] for _ in range(len(dfg_left))] + [ds2[0] for _ in range(len(dfg_right))])[:self.seq_len]
        cfg_segment_label = ([cs1[0] for _ in range(len(cfg_left))] + [cs2[0] for _ in range(len(cfg_right))])[:self.seq_len]

        cfg_tgt_label = (cd1 + cd2)[:self.seq_len]
        dfg_tgt_label = (dd1 + dd2)[:self.seq_len]

        # Truncate BERT inputs (we'll pad to exact length below)
        cfg_bert_input = (cfg_left + cfg_right)[:self.seq_len]
        cfg_bert_label = (c1_label + c2_label)[:self.seq_len]
        dfg_bert_input = (dfg_left + dfg_right)[:self.seq_len]

        # ---- Pad EVERYTHING to self.seq_len so collate can stack ----
        pad = self.vocab.pad_index
        L = self.seq_len

        cfg_bert_input    = _pad_to(cfg_bert_input,    L, pad)
        cfg_bert_label    = _pad_to(cfg_bert_label,    L, pad)
        cfg_segment_label = _pad_to(cfg_segment_label, L, pad)
        cfg_tgt_label     = _pad_to(cfg_tgt_label,     L, pad)

        dfg_bert_input    = _pad_to(dfg_bert_input,    L, pad)
        dfg_segment_label = _pad_to(dfg_segment_label, L, pad)
        dfg_tgt_label     = _pad_to(dfg_tgt_label,     L, pad)


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
