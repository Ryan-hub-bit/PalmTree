from torch.utils.data import Dataset
import tqdm
import torch
import random
import pickle as pkl


class BERTDataset2(Dataset):
    def __init__(
        self,
        cfg_corpus_path,
        dfg_corpus_path,
        vocab,
        seq_len,
        encoding="utf-8",
        corpus_lines=None,
        on_memory=True,
        # Optional semantic augmentation
        cfg_semantic_path=None,
        dfg_semantic_path=None,
        use_semantic: str = "none",  # one of {"none", "semantic-only", "augment"}
        augment_ratio: float = 0.5,
    ):
        self.vocab = vocab
        self.seq_len = seq_len

        self.bb_len = 50

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.dfg_corpus_path = cfg_corpus_path
        self.cfg_corpus_path = dfg_corpus_path
        self.encoding = encoding

        # Semantic options
        self.use_semantic = (use_semantic or "none").lower()
        if self.use_semantic not in {"none", "semantic-only", "augment"}:
            self.use_semantic = "none"
        # clamp ratio
        try:
            self.augment_ratio = max(0.0, min(1.0, float(augment_ratio)))
        except Exception:
            self.augment_ratio = 0.5
        self.cfg_semantic_path = cfg_semantic_path
        self.dfg_semantic_path = dfg_semantic_path

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

        # Optionally load semantic variants (must be aligned line-by-line)
        self.cfg_sem_lines = None
        self.dfg_sem_lines = None
        if on_memory and self.use_semantic in {"semantic-only", "augment"}:
            if self.cfg_semantic_path:
                try:
                    with open(self.cfg_semantic_path, "r", encoding=encoding) as f:
                        self.cfg_sem_lines = [line[:-1].split("\t") for line in f]
                except Exception:
                    self.cfg_sem_lines = None
            if self.dfg_semantic_path:
                try:
                    with open(self.dfg_semantic_path, "r", encoding=encoding) as f:
                        self.dfg_sem_lines = [line[:-1].split("\t") for line in f]
                except Exception:
                    self.dfg_sem_lines = None
        


        # if not on_memory:
        #     self.file = open(corpus_path, "r", encoding=encoding)
        #     self.random_file = open(corpus_path, "r", encoding=encoding)

        #     for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
        #         self.random_file.__next__()


    def __len__(self):
        return self.corpus_lines


    def __getitem__(self, item):
        c1, c2, c_label, d1, d2, d_label = self.random_sent(item)

        d1_random, d1_label = self.random_word(d1)
        d2_random, d2_label = self.random_word(d2)

        d1 = [self.vocab.sos_index] + d1_random + [self.vocab.eos_index]
        d2 = d2_random + [self.vocab.eos_index]

        c1 = [self.vocab.sos_index] + [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c1.split()] + [self.vocab.eos_index]
        c2 = [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c2.split()] + [self.vocab.eos_index]
        


        d1_label = [self.vocab.pad_index] + d1_label + [self.vocab.pad_index]
        d2_label = d2_label + [self.vocab.pad_index]

        dfg_segment_label = ([1 for _ in range(len(d1))] + [2 for _ in range(len(d2))])[:self.seq_len]
        cfg_segment_label = ([1 for _ in range(len(c1))] + [2 for _ in range(len(c2))])[:self.seq_len]
        dfg_bert_input = (d1 + d2)[:self.seq_len]
        dfg_bert_label = (d1_label + d2_label)[:self.seq_len]

        cfg_bert_input = (c1 + c2)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(dfg_bert_input))]
        dfg_bert_input.extend(padding), dfg_bert_label.extend(padding), dfg_segment_label.extend(padding) #, comp_label.extend(padding)
        cfg_padding = [self.vocab.pad_index for _ in range(self.seq_len - len(cfg_bert_input))]
        cfg_bert_input.extend(cfg_padding), cfg_segment_label.extend(cfg_padding)

        output = {"dfg_bert_input": dfg_bert_input,
                  "dfg_bert_label": dfg_bert_label,
                  "dfg_segment_label": dfg_segment_label,
                  "dfg_is_next": d_label,
                  "cfg_bert_input": cfg_bert_input,
                  "cfg_segment_label": cfg_segment_label,
                  "cfg_is_next": c_label
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
            if prob < 0.15:
                prob /= 0.15

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
        try:
            # Choose original or semantic based on mode
            use_sem_cfg = False
            use_sem_dfg = False

            if self.use_semantic == "semantic-only":
                use_sem_cfg = self.cfg_sem_lines is not None
                use_sem_dfg = self.dfg_sem_lines is not None
            elif self.use_semantic == "augment":
                use_sem_cfg = (self.cfg_sem_lines is not None) and (random.random() < self.augment_ratio)
                use_sem_dfg = (self.dfg_sem_lines is not None) and (random.random() < self.augment_ratio)

            c_src = self.cfg_sem_lines if use_sem_cfg else self.cfg_lines
            d_src = self.dfg_sem_lines if use_sem_dfg else self.dfg_lines

            c_line = c_src[item]
            d_line = d_src[item]

            # sanity check: both should have at least 2 items
            if len(c_line) < 2 or len(d_line) < 2:
                print(f"[Bad line @ index {item}]")
                print(f"  CFG: {c_line}")
                print(f"  DFG: {d_line}")
                # skip to next valid line
                return self.get_corpus_line((item + 1) % len(self))
            
            return c_line[0], c_line[1], d_line[0], d_line[1]

        except IndexError:
            print(f"[IndexError @ item={item}]")
            print(f"  len(cfg_lines)={len(self.cfg_lines)} len(dfg_lines)={len(self.dfg_lines)}")
            # wrap around instead of crashing
            total_len = min(len(self.cfg_lines), len(self.dfg_lines))
            return self.get_corpus_line(item % max(1, total_len))

    # def get_corpus_line(self, item):
    #     if self.on_memory:
    #         return self.cfg_lines[item][0], self.cfg_lines[item][1], self.dfg_lines[item][0], self.dfg_lines[item][1]

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