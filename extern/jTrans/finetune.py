from unicodedata import name
from transformers import BertTokenizer, BertForMaskedLM, BertModel, BertConfig
import torch.multiprocessing
from torch.utils.data import DataLoader
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data import load_paired_data, FunctionDataset_CL, FunctionDataset_CL_Load
from data_json import FunctionDataset_CL_JSON, FunctionDataset_CL_Load_JSON, FunctionDataset_CL_AddressAware_JSON
# from finetune_eval_with_pool import finetune_eval_cached, generate_embeddings, evaluate_with_pool  # Module not available
from transformers import AdamW
import torch.nn.functional as F
import argparse
import wandb
import logging
import time
import data
import pickle
from pathlib import Path
import json

# Add pretrain/address_aware to path for importing address embedding
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pretrain', 'address_aware'))
WANDB = False  # Set to True and run `wandb login` if you want to enable logging

def get_logger(name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=name)
    logger = logging.getLogger(__name__)
    s_handle = logging.StreamHandler(sys.stdout)
    s_handle.setLevel(logging.INFO)
    s_handle.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    logger.addHandler(s_handle)
    return logger

def train_dp(model, args, train_set, valid_set, logger):

    class Triplet_COS_Loss(nn.Module):
        def __init__(self,margin):
            super(Triplet_COS_Loss, self).__init__()
            self.margin=margin

        def forward(self, repr, good_code_repr, bad_code_repr):
            good_sim=F.cosine_similarity(repr, good_code_repr)
            bad_sim=F.cosine_similarity(repr, bad_code_repr)
            #print("simm ",good_sim.shape)
            loss=(self.margin-(good_sim-bad_sim)).clamp(min=1e-6).mean()
            return loss

    if WANDB:
        wandb.init(project=f'jTrans-finetune', name="jTrans_Freeze_10_Train_Test")
        wandb.config.update(args)

    logger.info("Initializing Model...")
    device = torch.device("cuda")
    model.to(device)
    logger.info("Finished Initialization...")
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=2, shuffle=True, prefetch_factor=1)
    valid_dataloader = DataLoader(valid_set, batch_size=args.eval_batch_size, num_workers=2, shuffle=True, prefetch_factor=1)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    optimizer_grouped_parameters.extend(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Only use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU")
    
    global_steps = 0
    etc=0
    for epoch in range(args.epoch):
        model.train()
        triplet_loss=Triplet_COS_Loss(margin=args.triplet_margin)
        train_iterator = tqdm(train_dataloader)
        
        for i, batch_data in enumerate(train_iterator):
            t1=time.time()
            
            # Check if this is address-aware (21 items) or baseline (9 items)
            if args.model_type == 'addressaware':
                # Unpack address-aware batch: 7 fields × 3 (anchor, positive, negative) = 21
                (seq1, seq2, seq3, mask1, mask2, mask3, seg1, seg2, seg3,
                 binary_pos1, binary_pos2, binary_pos3,
                 function_pos1, function_pos2, function_pos3,
                 bb_pos1, bb_pos2, bb_pos3,
                 var_offsets1, var_offsets2, var_offsets3) = batch_data
                
                # Move to GPU
                input_ids1, attention_mask1, token_type_ids1 = seq1.cuda(), mask1.cuda(), seg1.cuda()
                input_ids2, attention_mask2, token_type_ids2 = seq2.cuda(), mask2.cuda(), seg2.cuda()
                input_ids3, attention_mask3, token_type_ids3 = seq3.cuda(), mask3.cuda(), seg3.cuda()
                
                binary_pos1, binary_pos2, binary_pos3 = binary_pos1.cuda(), binary_pos2.cuda(), binary_pos3.cuda()
                function_pos1, function_pos2, function_pos3 = function_pos1.cuda(), function_pos2.cuda(), function_pos3.cuda()
                bb_pos1, bb_pos2, bb_pos3 = bb_pos1.cuda(), bb_pos2.cuda(), bb_pos3.cuda()
                var_offsets1, var_offsets2, var_offsets3 = var_offsets1.cuda(), var_offsets2.cuda(), var_offsets3.cuda()
                
                optimizer.zero_grad()
                
                # Address-aware model forward (wrapped to return pooler_output)
                output1 = model(
                    token_ids=input_ids1, attention_mask=attention_mask1, 
                    token_type_ids=token_type_ids1,
                    binary_pos=binary_pos1, function_pos=function_pos1, 
                    bb_pos=bb_pos1, var_offsets=var_offsets1
                )
                anchor = output1['pooler_output']
                
                output2 = model(
                    token_ids=input_ids2, attention_mask=attention_mask2, 
                    token_type_ids=token_type_ids2,
                    binary_pos=binary_pos2, function_pos=function_pos2, 
                    bb_pos=bb_pos2, var_offsets=var_offsets2
                )
                pos = output2['pooler_output']
                
                output3 = model(
                    token_ids=input_ids3, attention_mask=attention_mask3, 
                    token_type_ids=token_type_ids3,
                    binary_pos=binary_pos3, function_pos=function_pos3, 
                    bb_pos=bb_pos3, var_offsets=var_offsets3
                )
                neg = output3['pooler_output']
                
            else:
                # Baseline: 3 fields × 3 (anchor, positive, negative) = 9
                seq1, seq2, seq3, mask1, mask2, mask3, seg1, seg2, seg3 = batch_data
                
                input_ids1, attention_mask1, token_type_ids1 = seq1.cuda(), mask1.cuda(), seg1.cuda()
                input_ids2, attention_mask2, token_type_ids2 = seq2.cuda(), mask2.cuda(), seg2.cuda()
                input_ids3, attention_mask3, token_type_ids3 = seq3.cuda(), mask3.cuda(), seg3.cuda()

                optimizer.zero_grad()

                output1 = model(input_ids=input_ids1, attention_mask=attention_mask1, token_type_ids=token_type_ids1)
                anchor = output1.pooler_output

                output2 = model(input_ids=input_ids2, attention_mask=attention_mask2, token_type_ids=token_type_ids2)
                pos = output2.pooler_output

                output3 = model(input_ids=input_ids3, attention_mask=attention_mask3, token_type_ids=token_type_ids3)
                neg = output3.pooler_output

            loss = triplet_loss(anchor, pos, neg)

            loss.backward()
            
            # Gradient clipping for stability
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if (i+1) % args.log_every == 0:
                global_steps += 1
                tmp_lr = optimizer.param_groups[0]["lr"]
                # logger.info(f"[*] epoch: [{epoch}/{args.epoch+1}], steps: [{i}/{len(train_iterator)}], lr={tmp_lr}, loss={loss}")
                train_iterator.set_description(f"[*] epoch: [{epoch}/{args.epoch+1}], steps: [{i}/{len(train_iterator)}], lr={tmp_lr}, loss={loss}")
                if WANDB:
                    wandb.log({
                        'triplet loss' : loss,
                        'lr' : tmp_lr,
                        'global_step' : global_steps,
                    })

        if (epoch+1) % args.eval_every == 0:
            logger.info(f"Skipping evaluation (training only mode)")
            # logger.info(f"Doing Evaluation ...")
            
            # if args.use_cached_eval:
            #     logger.info("WARNING: Cached evaluation not available (finetune_eval_with_pool module missing)")
            #     logger.info("Skipping cached evaluation. Use original evaluation or run separate eval script after training.")
            #     mrr = 0.0
            # else:
            #     # Original batch-based evaluation
            #     mrr = finetune_eval(model, valid_dataloader, model_type=args.model_type)
            #     logger.info(f"[*] epoch: [{epoch}/{args.epoch+1}], mrr={mrr}")
            #     if WANDB:
            #         wandb.log({
            #             'mrr': mrr
            #         })
        if (epoch+1) % args.save_every == 0:
            logger.info(f"Saving Model ...")
            save_dir = os.path.join(args.output_path, f"finetune_epoch_{epoch+1}")
            # Handle both DataParallel and single GPU models
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(save_dir)
            logger.info(f"Model saved to {save_dir}")


def finetune_eval(net, data_loader, model_type='baseline'):
    net.eval()
    with torch.no_grad():
        avg=[]
        gt=[]
        cons=[]
        eval_iterator = tqdm(data_loader)
        
        for i, batch_data in enumerate(eval_iterator):
            if model_type == 'addressaware':
                # Unpack address-aware batch (only need anchor and positive for eval)
                (seq1, seq2, _, mask1, mask2, _, seg1, seg2, _,
                 binary_pos1, binary_pos2, _,
                 function_pos1, function_pos2, _,
                 bb_pos1, bb_pos2, _,
                 var_offsets1, var_offsets2, _) = batch_data
                
                input_ids1, attention_mask1, token_type_ids1 = seq1.cuda(), mask1.cuda(), seg1.cuda()
                input_ids2, attention_mask2, token_type_ids2 = seq2.cuda(), mask2.cuda(), seg2.cuda()
                
                binary_pos1, binary_pos2 = binary_pos1.cuda(), binary_pos2.cuda()
                function_pos1, function_pos2 = function_pos1.cuda(), function_pos2.cuda()
                bb_pos1, bb_pos2 = bb_pos1.cuda(), bb_pos2.cuda()
                var_offsets1, var_offsets2 = var_offsets1.cuda(), var_offsets2.cuda()
                
                output1 = model(
                    token_ids=input_ids1, attention_mask=attention_mask1, 
                    token_type_ids=token_type_ids1,
                    binary_pos=binary_pos1, function_pos=function_pos1, 
                    bb_pos=bb_pos1, var_offsets=var_offsets1
                )
                anchor = output1['pooler_output']
                
                output2 = model(
                    token_ids=input_ids2, attention_mask=attention_mask2, 
                    token_type_ids=token_type_ids2,
                    binary_pos=binary_pos2, function_pos=function_pos2, 
                    bb_pos=bb_pos2, var_offsets=var_offsets2
                )
                pos = output2['pooler_output']
                
            else:
                # Baseline
                seq1, seq2, _, mask1, mask2, _, seg1, seg2, _ = batch_data
                
                input_ids1, attention_mask1, token_type_ids1 = seq1.cuda(), mask1.cuda(), seg1.cuda()
                input_ids2, attention_mask2, token_type_ids2 = seq2.cuda(), mask2.cuda(), seg2.cuda()

                output1 = model(input_ids=input_ids1, attention_mask=attention_mask1, token_type_ids=token_type_ids1)
                anchor = output1.pooler_output

                output2 = model(input_ids=input_ids2, attention_mask=attention_mask2, token_type_ids=token_type_ids2)
                pos = output2.pooler_output

            ans=0
            for i in range(len(anchor)):    # check every vector of (vA,vB)
                vA=anchor[i:i+1].cpu()  #pos[i]
                sim=[]
                for j in range(len(pos)):
                    vB=pos[j:j+1].cpu()   # pos[j]
                    AB_sim=F.cosine_similarity(vA, vB).item()
                    sim.append(AB_sim)
                    if j!=i:
                        cons.append(AB_sim)
                sim=np.array(sim)
                y=np.argsort(-sim)
                posi=0
                for j in range(len(pos)):
                    if y[j]==i:
                        posi=j+1

                gt.append(sim[i])

                ans+=1/posi

            ans=ans/len(anchor)
            avg.append(ans)
        return np.mean(np.array(avg))

class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        
        # Baseline model: position_embeddings = word_embeddings (share weights)
        # This replicates the pretraining behavior
        self.embeddings.position_embeddings.weight = self.embeddings.word_embeddings.weight
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, **kwargs):
        # Use input_ids as position_ids to index into the shared embedding matrix
        if position_ids is None:
            position_ids = input_ids.clone()
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            **kwargs
        )


class AddressAwareBertWrapper(nn.Module):
    """
    Wrapper for address-aware BERT encoder to match BERT interface for finetuning.
    Extracts [CLS] token as pooled output.
    """
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model  # The BERT model with AddressAwareBERTEmbedding
        
    def forward(self, token_ids, attention_mask, token_type_ids,
                binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Forward pass returning pooled output (CLS token).
        
        Returns:
            Dict with:
                - pooler_output: [batch_size, hidden_size]
                - last_hidden_state: [batch_size, seq_len, hidden_size]
        """
        # Get embeddings
        embeddings = self.bert.embeddings(
            token_ids,
            token_type_ids,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets
        )
        
        # Pass through transformer encoder
        outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden]
        pooler_output = sequence_output[:, 0, :]  # CLS token
        
        # Return as dict (compatible with DataParallel)
        return {
            'pooler_output': pooler_output,
            'last_hidden_state': sequence_output
        }
    
    def save_pretrained(self, save_directory):
        """
        Save the model to a directory (compatible with HuggingFace interface).
        """
        import os
        import json
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the wrapped BERT model's state dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.bert.state_dict(), model_path)
        
        # Save config (extract from the BERT model's config if available)
        config_path = os.path.join(save_directory, 'config.json')
        
        # Try to get config from the model if it exists
        if hasattr(self.bert, 'config'):
            config = self.bert.config.to_dict()
        else:
            # Otherwise, construct config from model structure
            config = {
                'vocab_size': self.bert.embeddings.token_embedding.num_embeddings,
                'hidden_size': self.bert.embeddings.embed_size,
                'num_hidden_layers': len(self.bert.encoder.layer),
                'num_attention_heads': self.bert.encoder.layer[0].attention.self.num_attention_heads,
                'max_position_embeddings': 512,
                'type_vocab_size': self.bert.embeddings.segment_embedding.num_embeddings,
            }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="jTrans-Finetune")
    parser.add_argument("--model_path", type=str, default='./models/jTrans-pretrain',  help='the path of pretrain model')
    parser.add_argument("--model_type", type=str, default='baseline', choices=['baseline', 'addressaware'],
                        help='model type: baseline (BinBertModel) or addressaware (AddressAwareJTrans)')
    parser.add_argument("--output_path", type=str, default='./models/jTrans-finetune', help='the path where the finetune model be saved')
    parser.add_argument("--tokenizer", type=str, default='./jtrans_tokenizer', help='the path of tokenizer')
    parser.add_argument("--epoch", type=int, default=10, help='number of training epochs')
    parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
    parser.add_argument("--warmup", type=int, default=1000, help='warmup steps')
    parser.add_argument("--step_size", type=int, default=40000, help='scheduler step size')
    parser.add_argument("--gamma", type=float, default=0.99, help='scheduler gamma')
    parser.add_argument("--batch_size", type=int, default = 64, help='training batch size')
    parser.add_argument("--eval_batch_size", type=int, default = 256, help='evaluation batch size')
    parser.add_argument("--log_every", type=int, default =1, help='logging frequency')
    parser.add_argument("--local_rank", type=int, default = 0, help='local rank used for ddp')
    parser.add_argument("--freeze_cnt", type=int, default=10, help='number of layers to freeze')
    parser.add_argument("--weight_decay", type=float, default = 1e-4, help='regularization weight decay')
    parser.add_argument("--eval_every", type=int, default=1, help="evaluate the model every x epochs")
    parser.add_argument("--eval_every_step", type=int, default=1000, help="evaluate the model every x epochs")
    parser.add_argument("--save_every", type=int, default=1, help="save the model every x epochs")
    parser.add_argument("--train_path", type=str, default='./BinaryCorp/small_train', help='the path of training data')
    parser.add_argument("--eval_path", type=str, default='./BinaryCorp/small_test', help='the path of evaluation data')
    parser.add_argument("--load_path", type=str, default='./experiments/BinaryCorp-3M/', help='load path')
    parser.add_argument("--data_type", type=str, default='pickle', choices=['pickle', 'json'], 
                        help='data format: pickle (original) or json (baseline/address-aware)')
    parser.add_argument("--func_blocks", type=str, help='path to func_blocks.json (for json data_type)')
    parser.add_argument("--ground_truth", type=str, help='path to ground_truth.json (for json data_type)')
    parser.add_argument("--data_ratio", type=float, default=1.0,
                        help='ratio of data to use (0.0 to 1.0), default 1.0 for all data')
    parser.add_argument("--eval_pool_size", type=int, default=100, 
                        help='pool size for evaluation (positive + negatives)')
    parser.add_argument("--use_cached_eval", action='store_true', 
                        help='use cached embeddings for evaluation')
    parser.add_argument("--embedding_cache_dir", type=str, default=None, 
                        help='directory to cache embeddings')
    parser.add_argument("--triplet_margin", type=float, default=0.5,
                        help='margin for triplet loss (higher = stricter separation)')
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help='max gradient norm for clipping (0 = no clipping)')
    parser.add_argument("--target_opt", type=str, default=None,
                        help='target optimization level for positive samples (e.g., O3). '
                             'If set, anchor will be randomly selected from other opts (O0/O1/O2), '
                             'and positive will always be this opt from the same function. '
                             'Useful for training specifically for Ox->O3 retrieval tasks.')

    args = parser.parse_args()

    from datetime import datetime
    now = datetime.now() # current date and time
    TIMESTAMP="%Y%m%d%H%M"
    tim = now.strftime(TIMESTAMP)
    logger = get_logger(f"jTrans_{args.lr}_batchsize_{args.batch_size}_weight_decay_{args.weight_decay}_{tim}")

    logger.info(f"Loading Pretrained Model from {args.model_path} ...")
    
    if args.model_type == 'addressaware':
        # Load address-aware BERT encoder (pretrain only saves bert.state_dict())
        from pretrain.address_aware.address_embedding import AddressAwareBERTEmbedding
        import pickle
        
        # Load vocab for address vs daddr distinction
        vocab_path = os.path.join(args.tokenizer, 'vocab_addr.pkl')
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        vocab_stoi = vocab.stoi  # string to index mapping
        
        # Load config
        import json
        config_path = os.path.join(args.model_path, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create BERT model with config
        config = BertConfig(
            vocab_size=config_dict['vocab_size'],
            hidden_size=config_dict['hidden_size'],
            num_hidden_layers=config_dict['num_hidden_layers'],
            num_attention_heads=config_dict['num_attention_heads'],
            intermediate_size=config_dict['hidden_size'] * 4,
            max_position_embeddings=config_dict['max_position_embeddings'],
            type_vocab_size=config_dict.get('type_vocab_size', 2),
        )
        
        bert_model = BertModel(config, add_pooling_layer=False)
        
        # Replace embeddings with address-aware version
        bert_model.embeddings = AddressAwareBERTEmbedding(
            vocab_size=config_dict['vocab_size'],
            embed_size=config_dict['hidden_size'],
            dropout=0.1,
            max_len=config_dict['max_position_embeddings'],
            use_address_embedding=True,
            use_var_embedding=True,
            segment_types=256,
            vocab_stoi=vocab_stoi  # Pass vocab mapping for address vs daddr distinction
        )
        
        # Load pretrained weights
        weights_path = os.path.join(args.model_path, 'pytorch_model.bin')
        state_dict = torch.load(weights_path, map_location='cpu')
        bert_model.load_state_dict(state_dict)
        
        # Wrap for finetuning
        model = AddressAwareBertWrapper(bert_model)
        logger.info("Loaded address-aware BERT encoder")
        
    else:
        # Load baseline model 
        # The pretrained model has position_embeddings = word_embeddings (size 2902)
        # We need to load with the correct size
        from transformers import BertConfig as BaseConfig
        config_path = os.path.join(args.model_path, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Override max_position_embeddings to match vocab_size
        config_dict['max_position_embeddings'] = config_dict['vocab_size']
        
        config = BaseConfig(**config_dict)
        model = BinBertModel(config)
        
        # Load weights
        weights_path = os.path.join(args.model_path, 'pytorch_model.bin')
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)  # strict=False for pooler
        logger.info(f"Loaded baseline model with position_embeddings size {config.max_position_embeddings}")

    freeze_layer_count = args.freeze_cnt
    # Handle wrapper for address-aware
    if args.model_type == 'addressaware':
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
    else:
        for param in model.embeddings.parameters():
            param.requires_grad = False

    if freeze_layer_count != -1:
        # Handle wrapper for address-aware
        encoder = model.bert.encoder if args.model_type == 'addressaware' else model.encoder
        for layer in encoder.layer[:freeze_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False
    print(model)

    logger.info("Done ...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    logger.info("Tokenizer Done ...")

    load_train, load_test = False, False
    # load_train = f"{args.load_path}/jTrans-{args.train_path.split('/')[-1]}.pkl"
    # load_test = f"{args.load_path}/jTrans-{args.eval_path.split('/')[-1]}.pkl"
    
    if args.data_type == 'json':
        # Use JSON-based datasets (baseline or address-aware)
        logger.info(f"Loading JSON datasets from {args.func_blocks} and {args.ground_truth}")
        
        if args.model_type == 'addressaware':
            # Address-aware uses hierarchical position embeddings
            ft_train_dataset = FunctionDataset_CL_AddressAware_JSON(
                tokenizer, args.func_blocks, args.ground_truth,
                opt=['O0','O1','O2','O3'], add_ebd=True, data_ratio=args.data_ratio,
                target_opt=args.target_opt
            )
            ft_valid_dataset = FunctionDataset_CL_AddressAware_JSON(
                tokenizer, args.func_blocks, args.ground_truth,
                opt=['O0','O1','O2','O3'], add_ebd=True, data_ratio=args.data_ratio,
                target_opt=args.target_opt
            )
        else:
            # Baseline uses standard token sequences
            ft_train_dataset = FunctionDataset_CL_Load_JSON(
                tokenizer, args.func_blocks, args.ground_truth,
                opt=['O0','O1','O2','O3'], add_ebd=True, data_ratio=args.data_ratio
            )
            ft_valid_dataset = FunctionDataset_CL_Load_JSON(
                tokenizer, args.func_blocks, args.ground_truth,
                opt=['O0','O1','O2','O3'], add_ebd=True, data_ratio=args.data_ratio
            )
    else:
        # Use original pickle-based datasets
        ft_train_dataset = FunctionDataset_CL_Load(
            tokenizer, args.train_path, convert_jump_addr=True, 
            load=load_train, opt=['O0','O1','O2','O3','Os']
        )
        ft_valid_dataset = FunctionDataset_CL_Load(
            tokenizer, args.eval_path, convert_jump_addr=True, 
            load=load_test, opt=['O0','O1','O2','O3','Os']
        )
        if not load_train:
            pickle.dump(ft_train_dataset.datas, open(f"{args.load_path}/jTrans-{args.train_path.split('/')[-1]}.pkl", 'wb'))
            pickle.dump(ft_valid_dataset.datas, open(f"{args.load_path}/jTrans-{args.eval_path.split('/')[-1]}.pkl", 'wb'))
    
    logger.info("Done ...")
    train_dp(model, args, ft_train_dataset, ft_valid_dataset, logger)
    logger.info("Finished Training")

