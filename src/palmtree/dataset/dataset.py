from torch.utils.data import Dataset
import tqdm
import torch
import random
import pickle as pkl


class BERTDataset(Dataset):
    def __init__(self, dfg_corpus_path, cfg_corpus_path, dfg_src_path, cfg_src_path, dfg_tgt_path, cfg_tgt_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.bb_len = 50

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.dfg_corpus_path = dfg_corpus_path
        self.cfg_corpus_path = cfg_corpus_path
        self.dfg_src_path = dfg_src_path
        self.cfg_src_path = cfg_src_path
        self.dfg_tgt_path = dfg_tgt_path
        self.cfg_tgt_path = cfg_tgt_path
        self.encoding = encoding

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
        
        with  open(cfg_src_path, "r", encoding=encoding) as f:
            if self.cfg_src_path is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.cfg_src_lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
                if self.corpus_lines > len(self.cfg_lines):    
                    self.corpus_lines = len(self.cfg_lines)
            
        with  open(dfg_src_path, "r", encoding=encoding) as f:
            if self.dfg_src_path is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.dfg_src_lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
                if self.corpus_lines > len(self.cfg_lines):    
                    self.corpus_lines = len(self.cfg_lines)
    
        with  open(dfg_tgt_path, "r", encoding=encoding) as f:
            if self.dfg_tgt_path is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.dfg_tgt_lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
                if self.corpus_lines > len(self.dfg_lines):    
                    self.corpus_lines = len(self.dfg_lines)
            
        with  open(cfg_tgt_path, "r", encoding=encoding) as f:
            if self.cfg_tgt_path is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.cfg_tgt_lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                
                if self.corpus_lines > len(self.cfg_lines):    
                    self.corpus_lines = len(self.cfg_lines)
        


        # if not on_memory:
        #     self.file = open(corpus_path, "r", encoding=encoding)
        #     self.random_file = open(corpus_path, "r", encoding=encoding)

        #     for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
        #         self.random_file.__next__()

    #? question corpus_lines are the smaller one between CFG_line and DFG_line
    def __len__(self):
        return self.corpus_lines


    def __getitem__(self, item):
        c1, c2, cs1, cs2, cd1, cd2, c_label, d1, d2, ds1, ds2, dd1, dd2, d_label = self.random_sent(item)

        d1_random, d1_label = self.random_word(d1)
        d2_random, d2_label = self.random_word(d2)

        d1 = [self.vocab.sos_index] + d1_random + [self.vocab.eos_index]
        d2 = d2_random + [self.vocab.eos_index]

        c1 = [self.vocab.sos_index] + [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c1.split()] + [self.vocab.eos_index]
        c2 = [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c2.split()] + [self.vocab.eos_index]
        


        d1_label = [self.vocab.pad_index] + d1_label + [self.vocab.pad_index]
        d2_label = d2_label + [self.vocab.pad_index]

        # dfg_segment_label = ([1 for _ in range(len(d1))] + [2 for _ in range(len(d2))])[:self.seq_len]
        # cfg_segment_label = ([1 for _ in range(len(c1))] + [2 for _ in range(len(c2))])[:self.seq_len]
        dfg_segment_label = (ds1 + ds2)[:self.seq_len]
        cfg_segment_label = (cs1 + cs2)[:self.seq_len]

        
        cfg_tgt_label = (dd1 + dd2)[:self.seq_len]
        dfg_tgt_label = (cd1 + cd2)[:self.seq_len]
        
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
                  "dfg_tgt_label": dfg_tgt_label,
                  "dfg_is_next": d_label,
                  "cfg_bert_input": cfg_bert_input,
                  "cfg_segment_label": cfg_segment_label,
                  "cfg_tgt_label": cfg_tgt_label,
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
        c1, c2, d1, d2, cs1, cs2, cd1, cd2, ds1, ds2, dd1, dd2 = self.get_corpus_line(index)
        #? Question here......  should make sure the get_random_line's result is not the same as the correct answer 
        dice = random.random() # TODO: should throw the dice twice here. 
        if dice < 0.25:
            return c1, c2, cs1, cs2, cd1, cd2, 1, d1, d2, ds1, ds2, dd1, dd2, 1
        elif 0.25 <= dice < 0.5:
            cc2, ccs2, ccd2 = self.get_random_line() 
            return c1, cc2,cs1, ccs2, cd1, ccd2, 0, d1, d2, ds1, ds2, dd1, dd2, 1
        elif 0.5 <= dice < 0.75:
            return c1, c2,cs1, cs2, cd1, cd2, 1, d2, d1, ds2, ds1,dd2, dd1, 0
        else:
            cc2, ccs2, ccd2 = self.get_random_line()
            return c1, cc2, cs1, ccs2, cd1, ccd2,  0, d2, d1, ds2, ds1, dd2, dd1,0


    def get_corpus_line(self, item):
        if self.on_memory:
            return self.cfg_lines[item][0], self.cfg_lines[item][1], self.dfg_lines[item][0], self.dfg_lines[item][1], self.cfg_src_lines[item][0], self.cfg_src_lines[item][1], self.cfg_tgt_lines[item][0], self.cfg_tgt_lines[item][1],self.dfg_src_lines[item][0], self.dfg_src_lines[item][1], self.dfg_tgt_lines[item][0], self.dfg_tgt_lines[item][1]

        # now only on_memory copurs are supported
        # else:
        #     line = self.file.__next__()
        #     if line is None:
        #         self.file.close()
        #         self.file = open(self.corpus_path, "r", encoding=self.encoding)
        #         line = self.file.__next__()

        #     t1, t2 = line[:-1].split("\t")
        #     return t1, t2 


    # def get_random_line(self):
    #     if self.on_memory:
    #         l = self.cfg_lines[random.randrange(len(self.cfg_lines))]
    #         return l[1]
    def get_random_line(self, idx):
        """Return the line (and optionally the index) at given idx."""
        if self.on_memory:
            idx = random.randrange(len(self.cfg_lines))
            l = self.cfg_lines[idx]
            s = self.cfg_src_lines[idx]
            d = self.cfg_tgt_lines[idx]
            return l[1], s[1], d[1]

        # now only on_memory copurs are supported
        # line = self.file.__next__()
        # if line is None:
        #     self.file.close()
        #     self.file = open(self.corpus_path, "r", encoding=self.encoding)
        #     for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
        #         self.random_file.__next__()
        #     line = self.random_file.__next__()
        # return line[:-1].split("\t")[1] 