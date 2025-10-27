import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import *
import numpy as np
import palmtree
from palmtree import dataset
from palmtree import trainer
import pickle as pkl
import bert_pytorch
import os


print(palmtree.__file__)
vocab_path = "/home/louie/PalmTree/result/windows8noaddr/vocab"
train_cfg_dataset = "/home/louie/PalmTree/datalong/cfg_8.txt"
train_dfg_dataset = "/home/louie/PalmTree/datalong/dfg_8.txt"
output_path = "/home/louie/PalmTree/result/windows8noaddr/transformer"

# Create directories if they don't exist
os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
    with open(train_dfg_dataset, "r", encoding="utf-8") as f2:
        vocab = dataset.WordVocab([f1, f2], max_size=13000, min_freq=1)

print("VOCAB SIZE:", len(vocab))
vocab.save_vocab(vocab_path)


print("Loading Vocab", vocab_path)
vocab = dataset.WordVocab.load_vocab(vocab_path)
print("Vocab Size: ", len(vocab))
# print(vocab.itos)


print("Loading Train Dataset")
train_dataset = dataset.BERTDataset(
    cfg_corpus_path=train_cfg_dataset,
    dfg_corpus_path=train_dfg_dataset,
    cfg_src_path=None,  # Address files not available by default
    dfg_src_path=None,
    cfg_tgt_path=None,
    dfg_tgt_path=None,
    vocab=vocab,
    seq_len=100,  # Increased from 20 to 100 for 8 instructions (was 20 for 2 instructions)
    corpus_lines=None,
    on_memory=True
)

# print("Loading Test Dataset", test_dataset)
# test_dataset = bert_pytorch.dataset.BERTDataset(test_dataset, test_dataset, vocab, seq_len=20, on_memory=True) \
#     if test_dataset is not None else None

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)

# test_data_loader = DataLoader(test_dataset, batch_size=256, num_workers=10) \
    # if test_dataset is not None else None
test_data_loader = None

print("Building BERT model") 
bert = bert_pytorch.BERT2(len(vocab), hidden=128, n_layers=6, attn_heads=8, dropout=0.1)

print("Creating BERT Trainer")
trainer = trainer.BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        lr=1e-5, betas=(0.9, 0.999), weight_decay=0.0,
                        with_cuda=True, cuda_devices=[0], log_freq=100)


print("Training Start")
for epoch in range(20):
    trainer.train(epoch)
    trainer.save(epoch, output_path)
#    if test_data_loader is not None:
#        trainer.test(epoch)     
