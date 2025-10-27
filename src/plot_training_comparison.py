import torch
import numpy as np
import matplotlib.pyplot as plt
from palmtree.trainer.pretrain import BERTTrainer
import bert_pytorch
from bert_pytorch import BERT
import json
import os


def evaluate_epoch(trainer, test_dataloader):
    """Evaluate model on test dataset and return metrics."""
    trainer.model.eval()
    total_mlm_loss = 0
    total_dfg_nsp_loss = 0
    total_cfg_nsp_loss = 0
    total_dfg_correct = 0
    total_cfg_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in test_dataloader:
            # Move data to device
            data = {key: value.to(trainer.device) for key, value in data.items()}
            
            # Forward pass
            dfg_next_sent_output, cfg_next_sent_output, mask_lm_output = trainer.model(
                data["cfg_bert_input"],
                data["cfg_segment_label"],
                data["cfg_tgt_label"],
                data["dfg_bert_input"],
                data["dfg_segment_label"],
                data["dfg_tgt_label"]
            )
            
            # Calculate losses
            mlm_loss = trainer.masked_criterion(mask_lm_output.transpose(1, 2), 
                                             data["cfg_bert_label"])
            dfg_nsp_loss = trainer.dfg_next_criterion(dfg_next_sent_output, 
                                                    data["dfg_is_next"])
            cfg_nsp_loss = trainer.cfg_next_criterion(cfg_next_sent_output, 
                                                    data["cfg_is_next"])
            
            # Calculate NSP accuracy
            dfg_pred = torch.argmax(dfg_next_sent_output, dim=1)
            cfg_pred = torch.argmax(cfg_next_sent_output, dim=1)
            dfg_correct = (dfg_pred == data["dfg_is_next"]).sum().item()
            cfg_correct = (cfg_pred == data["cfg_is_next"]).sum().item()
            
            # Accumulate metrics
            batch_size = data["dfg_bert_input"].size(0)
            total_mlm_loss += mlm_loss.item() * batch_size
            total_dfg_nsp_loss += dfg_nsp_loss.item() * batch_size
            total_cfg_nsp_loss += cfg_nsp_loss.item() * batch_size
            total_dfg_correct += dfg_correct
            total_cfg_correct += cfg_correct
            total_samples += batch_size
    
    # Calculate averages
    avg_mlm_loss = total_mlm_loss / total_samples
    avg_dfg_nsp_loss = total_dfg_nsp_loss / total_samples
    avg_cfg_nsp_loss = total_cfg_nsp_loss / total_samples
    dfg_nsp_acc = total_dfg_correct / total_samples
    cfg_nsp_acc = total_cfg_correct / total_samples
    perplexity = np.exp(avg_mlm_loss)
    
    # Calculate total loss the same way as in training
    total_loss = avg_dfg_nsp_loss + avg_cfg_nsp_loss + avg_mlm_loss
    
    return {
        'total_loss': total_loss,
        'mlm_loss': avg_mlm_loss,
        'perplexity': perplexity,
        'dfg_nsp_loss': avg_dfg_nsp_loss,
        'cfg_nsp_loss': avg_cfg_nsp_loss,
        'dfg_nsp_acc': dfg_nsp_acc,
        'cfg_nsp_acc': cfg_nsp_acc
    }

# Plotting functionality moved to plot_metrics.py

def create_test_dataloader(data_path, vocab_path, batch_size=32, model_type="baseline"):
    """Create appropriate test dataloader based on model type."""
    from palmtree.dataset import BERTDataset, WordVocab
    
    # Load vocabulary
    vocab = WordVocab.load_vocab(vocab_path)
    
    # Use the test directory directly
    base_dir = data_path  # This should be 'data/test'
    
    # Construct paths for all required files
    cfg_dataset = os.path.join(base_dir, f"cfg_test.txt")
    dfg_dataset = os.path.join(base_dir, f"dfg_test.txt")
    cfg_srcaddr = os.path.join(base_dir, f"cfg_test_src.txt")
    dfg_srcaddr = os.path.join(base_dir, f"dfg_test_src.txt")
    cfg_tgtaddr = os.path.join(base_dir, f"cfg_test_tgt.txt")
    dfg_tgtaddr = os.path.join(base_dir, f"dfg_test_tgt.txt")
    
    if model_type == "address_aware":
        # Use BERTDataset for address-aware model
        test_dataset = BERTDataset(
            cfg_dataset, dfg_dataset,
            cfg_srcaddr, dfg_srcaddr,
            cfg_tgtaddr, dfg_tgtaddr,
            vocab, seq_len=20,
            corpus_lines=None, 
            on_memory=True,
            drive_mode="min"
        )
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

def load_and_evaluate_model(model_path, test_data_path, vocab_path, model_type="baseline"):
    """Load model and evaluate it with appropriate dataloader."""
    from palmtree.dataset import WordVocab  # Import WordVocab class
    from palmtree.model import BERTLM  # Import BERTLM wrapper
    
    # Load vocabulary and create a fresh BERT model
    vocab = WordVocab.load_vocab(vocab_path)
    vocab_size = len(vocab)
    
    # Create a new BERT model with the same architecture as used in training
    bert = bert_pytorch.BERT(vocab_size, hidden=128, n_layers=12, attn_heads=8, dropout=0.0)
    
    # Load the state dict from checkpoint with weights_only=False for trusted model
    checkpoint = torch.load(model_path, weights_only=False)
    if isinstance(checkpoint, dict):
        # If checkpoint is a state dict
        bert.load_state_dict(checkpoint)
    else:
        # If checkpoint is the full model
        bert = checkpoint
        
    # Wrap the BERT model with BERTLM as done in training
    model = BERTLM(bert, vocab_size)
    
    test_dataloader = create_test_dataloader(test_data_path, vocab_path, 
                                           model_type=model_type)
    
    trainer = BERTTrainer(
        bert=bert,
        vocab_size=vocab_size,
        train_dataloader=None,
        test_dataloader=test_dataloader,
        with_cuda=torch.cuda.is_available()
    )
    
    # Set the loaded model as trainer's model
    trainer.model = model.to(trainer.device)
    
    return evaluate_epoch(trainer, test_dataloader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Directory containing model checkpoints')
    parser.add_argument('--test_data', required=True, help='Base name for test data (e.g., "train" for cfg_train.txt)')
    parser.add_argument('--vocab_path', required=True, help='Path to vocabulary file')
    parser.add_argument('--model_type', required=True, choices=['baseline', 'address_aware'], 
                      help='Type of model to evaluate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', default='evaluation_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory for this model type
    model_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize metrics collection
    metrics = {
        'total_loss': [], 'mlm_loss': [], 'perplexity': [], 
        'dfg_nsp_loss': [], 'cfg_nsp_loss': [],
        'dfg_nsp_acc': [], 'cfg_nsp_acc': []
    }
    
    # Find all epoch checkpoints
    checkpoint_pattern = 'transformer_mlp.ep' if args.model_type == 'address_aware' else 'transformer.ep'
    checkpoints = sorted([f for f in os.listdir(args.model_dir) if f.startswith(checkpoint_pattern)])
    
    # Evaluate each epoch
    for checkpoint in checkpoints:
        # Extract the epoch number by removing 'ep' from the end
        epoch = int(checkpoint.split('ep')[-1])
        print(f"Evaluating {args.model_type} model - epoch {epoch}")
        
        # Evaluate model
        epoch_metrics = load_and_evaluate_model(
            os.path.join(args.model_dir, checkpoint),
            args.test_data,
            args.vocab_path,
            model_type=args.model_type
        )
        
        # Save per-epoch results
        epoch_data = {
            'epoch': epoch,
            'checkpoint': checkpoint,
            **epoch_metrics
        }
        with open(os.path.join(model_dir, f'epoch_{epoch:02d}.json'), 'w') as f:
            json.dump(epoch_data, f, indent=2)
        
        # Collect metrics
        for k, v in epoch_metrics.items():
            metrics[k].append(v)
    
    # Save all metrics
    results = {
        'metrics': metrics,
        'settings': {
            'model_dir': args.model_dir,
            'test_data': args.test_data,
            'vocab_path': args.vocab_path,
            'model_type': args.model_type,
            'batch_size': args.batch_size
        }
    }
    with open(os.path.join(model_dir, 'all_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved in {model_dir}")
    print(f"Per-epoch results are saved in epoch_XX.json files")
    print(f"All metrics saved in {os.path.join(model_dir, 'all_metrics.json')}")