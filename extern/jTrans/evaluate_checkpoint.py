"""
Standalone evaluation script for testing saved checkpoints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'pretrain', 'address_aware'))

from data_json import load_paired_data_json, FunctionDataset_CL_Load_JSON, FunctionDataset_CL_AddressAware_JSON
from address_embedding import AddressAwareBERTEmbedding


class AddressAwareBertWrapper(nn.Module):
    """Wrapper to use address-aware BERT with DataParallel"""
    def __init__(self, embedding_module, bert_model):
        super().__init__()
        self.embedding = embedding_module
        self.bert = bert_model
    
    def forward(self, token_ids, attention_mask, token_type_ids, 
                binary_pos, function_pos, bb_pos, var_offsets):
        embeddings = self.embedding(
            token_ids, binary_pos, function_pos, bb_pos, var_offsets
        )
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return {
            'pooler_output': outputs.pooler_output,
            'last_hidden_state': outputs.last_hidden_state
        }


def evaluate_checkpoint(args):
    """Evaluate a saved checkpoint"""
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_dir)
    
    # Load data
    print("Loading validation data...")
    val_func_emb_data = load_paired_data_json(
        args.data_path, 
        args.ground_truth_path, 
        data_ratio=args.data_ratio
    )
    
    if args.model_type == 'addressaware':
        val_set = FunctionDataset_CL_AddressAware_JSON(
            val_func_emb_data, tokenizer, args.max_len
        )
    else:
        val_set = FunctionDataset_CL_Load_JSON(
            val_func_emb_data, tokenizer, args.max_len
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, 
        shuffle=False, num_workers=4
    )
    
    # Load model
    print("Loading model...")
    if args.model_type == 'addressaware':
        bert_config = BertConfig.from_pretrained(args.vocab_dir)
        bert_model = BertModel(bert_config)
        
        embedding_module = AddressAwareBERTEmbedding(
            d_model=bert_config.hidden_size,
            vocab_stoi=tokenizer.get_vocab(),
            max_len=args.max_len
        )
        
        model = AddressAwareBertWrapper(embedding_module, bert_model)
        
        # Load state dict
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint)
    else:
        model = BertModel.from_pretrained(args.checkpoint_path)
    
    # Move to GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.cuda()
    
    # Evaluate
    print("\n" + "="*80)
    print("Starting Evaluation")
    print("="*80)
    
    results = finetune_eval(model, val_loader, model_type=args.model_type)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"MRR:        {results['mrr']:.4f}")
    print(f"Recall@1:   {results['recall@1']:.4f}")
    print(f"Recall@5:   {results['recall@5']:.4f}")
    print(f"Recall@10:  {results['recall@10']:.4f}")
    print("="*80)
    
    return results


def finetune_eval(net, data_loader, model_type='baseline'):
    """
    Evaluate using pooling strategy with proper ground truth tracking.
    Uses dataset metadata to identify which samples belong to same function.
    """
    net.eval()
    with torch.no_grad():
        # Step 1: Collect all embeddings with their function IDs
        all_embeddings = []
        function_ids = []  # Track which function each embedding belongs to
        
        print("Collecting embeddings from validation set...")
        batch_idx = 0
        for batch_data in tqdm(data_loader, desc="Encoding"):
            if model_type == 'addressaware':
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
                
                output1 = net(
                    token_ids=input_ids1, attention_mask=attention_mask1, 
                    token_type_ids=token_type_ids1,
                    binary_pos=binary_pos1, function_pos=function_pos1, 
                    bb_pos=bb_pos1, var_offsets=var_offsets1
                )
                emb1 = output1['pooler_output'].cpu()
                
                output2 = net(
                    token_ids=input_ids2, attention_mask=attention_mask2, 
                    token_type_ids=token_type_ids2,
                    binary_pos=binary_pos2, function_pos=function_pos2, 
                    bb_pos=bb_pos2, var_offsets=var_offsets2
                )
                emb2 = output2['pooler_output'].cpu()
            else:
                seq1, seq2, _, mask1, mask2, _, seg1, seg2, _ = batch_data
                
                input_ids1, attention_mask1, token_type_ids1 = seq1.cuda(), mask1.cuda(), seg1.cuda()
                input_ids2, attention_mask2, token_type_ids2 = seq2.cuda(), mask2.cuda(), seg2.cuda()

                output1 = net(input_ids=input_ids1, attention_mask=attention_mask1, token_type_ids=token_type_ids1)
                emb1 = output1.pooler_output.cpu()

                output2 = net(input_ids=input_ids2, attention_mask=attention_mask2, token_type_ids=token_type_ids2)
                emb2 = output2.pooler_output.cpu()
            
            # Both embeddings from same batch come from SAME functions
            # This is the key: anchor and positive in same batch are from same function
            for i in range(len(emb1)):
                all_embeddings.append(emb1[i])
                function_ids.append(batch_idx * len(emb1) + i)  # Unique function ID per batch item
                
                all_embeddings.append(emb2[i])
                function_ids.append(batch_idx * len(emb1) + i)  # Same function ID as emb1[i]
            
            batch_idx += 1
        
        all_embeddings = torch.stack(all_embeddings)  # [N, hidden_size]
        function_ids = np.array(function_ids)  # [N]
        
        print(f"Collected {len(all_embeddings)} embeddings from {len(np.unique(function_ids))} unique functions")
        print(f"Evaluating function similarity...")
        
        # Step 2: For each embedding, find its ground truth pairs (same function_id) and evaluate
        mrr_list = []
        recall_at_1 = []
        recall_at_5 = []
        recall_at_10 = []
        
        for i in tqdm(range(len(all_embeddings)), desc="Evaluating"):
            query_emb = all_embeddings[i:i+1]
            query_func_id = function_ids[i]
            
            # Compute similarity against ALL embeddings
            similarities = F.cosine_similarity(query_emb, all_embeddings, dim=1).numpy()
            
            # Exclude self-similarity
            similarities[i] = -np.inf
            
            # Rank by similarity (descending)
            ranked_indices = np.argsort(-similarities)
            
            # Find where ANY ground truth appears (same function_id, excluding self)
            ground_truth_mask = (function_ids == query_func_id) & (np.arange(len(function_ids)) != i)
            ground_truth_indices = np.where(ground_truth_mask)[0]
            
            if len(ground_truth_indices) == 0:
                continue  # Skip if no ground truth pairs
            
            # Find best rank among all ground truth pairs
            ranks = []
            for gt_idx in ground_truth_indices:
                rank = np.where(ranked_indices == gt_idx)[0][0] + 1
                ranks.append(rank)
            
            best_rank = min(ranks)  # Best rank among all ground truth pairs
            
            # Compute metrics using best rank
            mrr_list.append(1.0 / best_rank)
            recall_at_1.append(1.0 if best_rank <= 1 else 0.0)
            recall_at_5.append(1.0 if best_rank <= 5 else 0.0)
            recall_at_10.append(1.0 if best_rank <= 10 else 0.0)
        
        # Compute final metrics
        mrr = np.mean(mrr_list)
        r1 = np.mean(recall_at_1)
        r5 = np.mean(recall_at_5)
        r10 = np.mean(recall_at_10)
        
        print(f"\nEvaluated {len(mrr_list)} queries")
        
        return {
            'mrr': mrr,
            'recall@1': r1,
            'recall@5': r5,
            'recall@10': r10
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to validation data JSON')
    parser.add_argument('--ground_truth_path', type=str, required=True,
                        help='Path to ground truth JSON')
    parser.add_argument('--vocab_dir', type=str, required=True,
                        help='Path to vocabulary directory')
    
    # Model config
    parser.add_argument('--model_type', type=str, default='addressaware',
                        choices=['baseline', 'addressaware'])
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_ratio', type=float, default=0.1,
                        help='Ratio of data to use (for faster testing)')
    
    args = parser.parse_args()
    
    evaluate_checkpoint(args)
