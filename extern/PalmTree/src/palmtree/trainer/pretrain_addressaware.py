import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from ..model import BERTLM, BERT
from ..model.language_model_addressaware import AddressAwareBERTLM
from .optim_schedule import ScheduledOptim

import tqdm


class BERTTrainer:
    """
    BERTTrainer with support for both original and address-aware models.
    
    Supports two training modes:
        1. original: Standard BERT (tokens only)
        2. addressaware: AddressAwareBERT (tokens + address positions + var offsets)
    
    Training tasks:
        1. Masked Language Model (MLM)
        2. Next Sentence Prediction (NSP) for CFG
        3. Data-flow Utilization Prediction (DUP) for DFG
    """

    def __init__(self, bert, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, mode='original'):
        """
        :param bert: BERT or AddressAwareBERT model
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: training with cuda
        :param cuda_devices: GPU device IDs
        :param log_freq: logging frequency of the batch iteration
        :param mode: 'original' or 'addressaware'
        """
        
        self.mode = mode
        
        # Setup cuda device for BERT training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        
        # Initialize the BERT Language Model with appropriate wrapper
        if mode == 'addressaware':
            print("Using AddressAwareBERTLM wrapper")
            self.model = AddressAwareBERTLM(bert, vocab_size).to(self.device)
        else:
            print("Using standard BERTLM wrapper")
            self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the AdamW optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Loss functions
        # Use ignore_index=-100 to ignore non-masked tokens in loss calculation
        self.masked_criterion = nn.NLLLoss(ignore_index=-100)
        self.dfg_next_criterion = nn.NLLLoss()
        self.cfg_next_criterion = nn.NLLLoss()
        
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        Loop over the data_loader for training or testing.
        Handles both original and address-aware modes.
        
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

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. Forward pass - different for original vs addressaware
            if self.mode == 'addressaware':
                # Address-aware forward: pass address positions and var offsets
                dfg_next_sent_output, cfg_next_sent_output, mask_lm_output = self.model.forward(
                    dfg_input=data["dfg_bert_input"],
                    dfg_segment_label=data["dfg_segment_label"],
                    dfg_binary_pos=data["dfg_binary_pos"],
                    dfg_function_pos=data["dfg_function_pos"],
                    dfg_bb_pos=data["dfg_bb_pos"],
                    dfg_var_offsets=data["dfg_var_offsets"],
                    cfg_input=data["cfg_bert_input"],
                    cfg_segment_label=data["cfg_segment_label"],
                    cfg_binary_pos=data["cfg_binary_pos"],
                    cfg_function_pos=data["cfg_function_pos"],
                    cfg_bb_pos=data["cfg_bb_pos"],
                    cfg_var_offsets=data["cfg_var_offsets"]
                )
            else:
                # Original forward: only tokens and segment labels
                dfg_next_sent_output, cfg_next_sent_output, mask_lm_output = self.model.forward(
                    data["dfg_bert_input"], 
                    data["dfg_segment_label"], 
                    data["cfg_bert_input"], 
                    data["cfg_segment_label"]
                )

            # 2. Calculate losses
            # 2-1. Next Sentence Prediction losses
            dfg_next_loss = self.dfg_next_criterion(dfg_next_sent_output, data["dfg_is_next"])
            cfg_next_loss = self.cfg_next_criterion(cfg_next_sent_output, data["cfg_is_next"])

            # 2-2. Masked Language Model loss
            mask_loss = self.masked_criterion(mask_lm_output.transpose(1, 2), data["dfg_bert_label"])

            # 2-3. Total loss
            loss = dfg_next_loss + cfg_next_loss + mask_loss

            # 3. Backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # 4. Logging
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "mode": self.mode,
                "CWP": cfg_next_loss.item(),
                "DUP": dfg_next_loss.item(),
                "MLM": mask_loss.item(),
                "total": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

            avg_loss += loss.item()

        print(f"EP{epoch}, {str_code}: avg_loss={avg_loss / len(data_iter):.6f}")


    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
