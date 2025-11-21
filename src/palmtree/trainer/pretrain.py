import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import os
import json

from ..model import BERTLM, BERT
from .optim_schedule import ScheduledOptim

import tqdm


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        # self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.masked_criterion = nn.NLLLoss(ignore_index=0)
        self.dfg_next_criterion = nn.NLLLoss()
        self.cfg_next_criterion = nn.NLLLoss()
        self.comp_criterion = nn.NLLLoss(ignore_index=0)
        self.sentence_bert = nn.NLLLoss()
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        # Initialize metrics tracking
        total_mask_loss = 0
        total_dfg_next_loss = 0
        total_cfg_next_loss = 0
        total_dfg_correct = 0
        total_cfg_correct = 0
        total_samples = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            dfg_next_sent_output, cfg_next_sent_output, mask_lm_output = self.model(
                data["cfg_bert_input"], 
                data["cfg_segment_label"], 
                data["cfg_tgt_label"], 
                data["dfg_bert_input"], 
                data["dfg_segment_label"], 
                data["dfg_tgt_label"]
            )
            
            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            dfg_next_loss = self.dfg_next_criterion(dfg_next_sent_output, data["dfg_is_next"])
            cfg_next_loss = self.cfg_next_criterion(cfg_next_sent_output, data["cfg_is_next"])

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.masked_criterion(mask_lm_output.transpose(1, 2), data["cfg_bert_label"])

            # Calculate metrics for both train and test
            dfg_pred = torch.argmax(dfg_next_sent_output, dim=1)
            cfg_pred = torch.argmax(cfg_next_sent_output, dim=1)
            dfg_correct = (dfg_pred == data["dfg_is_next"]).sum().item()
            cfg_correct = (cfg_pred == data["cfg_is_next"]).sum().item()
            
            # Accumulate batch metrics
            batch_size = data["dfg_bert_input"].size(0)
            total_mask_loss += mask_loss.item() * batch_size
            total_dfg_next_loss += dfg_next_loss.item() * batch_size
            total_cfg_next_loss += cfg_next_loss.item() * batch_size
            total_dfg_correct += dfg_correct
            total_cfg_correct += cfg_correct
            total_samples += batch_size

            # 2-5. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = dfg_next_loss + cfg_next_loss + mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            
            # Show current metrics
            current_mlm_loss = total_mask_loss / total_samples
            current_dfg_acc = total_dfg_correct / total_samples
            current_cfg_acc = total_cfg_correct / total_samples
            
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "MLM Loss": f"{current_mlm_loss:.4f}",
                "DFG Acc": f"{current_dfg_acc:.4f}",
                "CFG Acc": f"{current_cfg_acc:.4f}",
                "CWP Loss": f"{cfg_next_loss.item():.4f}",
                "DUP Loss": f"{dfg_next_loss.item():.4f}"
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                
        # Calculate and save metrics at the end of epoch
        avg_mlm_loss = total_mask_loss / total_samples
        avg_dfg_nsp_loss = total_dfg_next_loss / total_samples
        avg_cfg_nsp_loss = total_cfg_next_loss / total_samples
        dfg_nsp_acc = total_dfg_correct / total_samples
        cfg_nsp_acc = total_cfg_correct / total_samples
        perplexity = torch.exp(torch.tensor(avg_mlm_loss)).item()

        metrics = {
            'epoch': epoch,
            'mode': str_code,
            'total_loss': avg_mlm_loss + avg_dfg_nsp_loss + avg_cfg_nsp_loss,
            'mlm_loss': avg_mlm_loss,
            'perplexity': perplexity,
            'dfg_nsp_loss': avg_dfg_nsp_loss,
            'cfg_nsp_loss': avg_cfg_nsp_loss,
            'dfg_nsp_acc': dfg_nsp_acc,
            'cfg_nsp_acc': cfg_nsp_acc
        }
        
        # Save metrics to file
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'epoch_{epoch:02d}_{str_code}.json')
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nEpoch {epoch} {str_code.capitalize()} Metrics:")
        print(f"MLM Loss: {avg_mlm_loss:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
        print(f"DFG NSP Loss: {avg_dfg_nsp_loss:.4f} (Accuracy: {dfg_nsp_acc:.4f})")
        print(f"CFG NSP Loss: {avg_cfg_nsp_loss:.4f} (Accuracy: {cfg_nsp_acc:.4f})")
        print(f"Metrics saved to {output_file}")


    def save(self, epoch, file_path="output/bert_trained.model", save_best_only=False, is_best=False, best_dir="output_addressaware_new"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+".ep%d" % epoch
        :param save_best_only: If True, only save when is_best is True
        :param is_best: If True, this is the best model so far
        :param best_dir: Directory to save best model and best BERT
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)

        # If this is the best model, save both best_model.pt and best_bert.pt
        if is_best:
            os.makedirs(best_dir, exist_ok=True)
            best_model_path = os.path.join(best_dir, "best_model.pt")
            best_bert_path = os.path.join(best_dir, "best_bert.pt")
            # Save the full model (for resuming training)
            torch.save(self.model.state_dict(), best_model_path)
            # Save only the BERT encoder (for embedding extraction)
            torch.save(self.bert.cpu(), best_bert_path)
            self.bert.to(self.device)
            print(f"Best model saved: {best_model_path}")
            print(f"Best BERT encoder saved: {best_bert_path}")

        return output_path
