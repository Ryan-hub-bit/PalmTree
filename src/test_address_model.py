import torch
import numpy as np
from palmtree.trainer.pretrain import BERTTrainer
import bert_pytorch
from bert_pytorch import BERT
from palmtree.dataset import BERTDataset, WordVocab
from palmtree.model import BERTLM
import json
import os

def evaluate_epoch(trainer, test_dataloader):
    """Evaluate model on test dataset and return metrics."""
    print("Starting evaluation...")
    trainer.model.eval()
    total_mlm_loss = 0
    total_dfg_nsp_loss = 0
    total_cfg_nsp_loss = 0
    total_dfg_correct = 0
    total_cfg_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        print(f"Processing {len(test_dataloader)} batches...")
        for batch_idx, data in enumerate(test_dataloader):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}/{len(test_dataloader)}")
            # Move data to device
            data = {key: value.to(trainer.device) for key, value in data.items()}
            
            # Forward pass
            dfg_next_sent_output, cfg_next_sent_output, mask_lm_output = trainer.model(
                data["dfg_bert_input"],
                data["dfg_segment_label"],
                data["dfg_tgt_label"],
                data["cfg_bert_input"],
                data["cfg_segment_label"],
                data["cfg_tgt_label"]
            )
            
            # Calculate losses
            mlm_loss = trainer.masked_criterion(mask_lm_output.transpose(1, 2), 
                                             data["dfg_bert_label"])
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
            
            if batch_idx % 100 == 0:
                # Print intermediate metrics
                current_mlm_loss = total_mlm_loss / total_samples
                current_dfg_acc = total_dfg_correct / total_samples
                current_cfg_acc = total_cfg_correct / total_samples
                print(f"Current metrics - MLM Loss: {current_mlm_loss:.4f}, "
                      f"DFG Acc: {current_dfg_acc:.4f}, CFG Acc: {current_cfg_acc:.4f}")
    
    print("Calculating final metrics...")
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

def verify_files_exist(*file_paths):
    """Verify that all required files exist."""
    missing_files = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

def create_test_dataloader(data_path, vocab_path, batch_size=32):
    """Create test dataloader for address-aware model."""
    # Load vocabulary
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    vocab = WordVocab.load_vocab(vocab_path)
    print(f"Loaded vocabulary with size: {len(vocab)}")
    
    # Construct paths for all required files
    cfg_dataset = os.path.join(data_path, "cfg_test.txt")
    dfg_dataset = os.path.join(data_path, "dfg_test.txt")
    cfg_srcaddr = os.path.join(data_path, "cfg_test_src.txt")
    dfg_srcaddr = os.path.join(data_path, "dfg_test_src.txt")
    cfg_tgtaddr = os.path.join(data_path, "cfg_test_tgt.txt")
    dfg_tgtaddr = os.path.join(data_path, "dfg_test_tgt.txt")
    
    # Verify all required files exist
    verify_files_exist(
        cfg_dataset, dfg_dataset,
        cfg_srcaddr, dfg_srcaddr,
        cfg_tgtaddr, dfg_tgtaddr
    )
    
    # Create test dataloader
    test_dataset = BERTDataset(
        cfg_dataset, dfg_dataset,
        cfg_srcaddr, dfg_srcaddr,
        cfg_tgtaddr, dfg_tgtaddr,
        vocab, seq_len=20,
        corpus_lines=None, 
        on_memory=True,
        drive_mode="min"
    )
    
    # Print dataset info
    print(f"Dataset size: {len(test_dataset)} samples")
    print(f"Using batch size: {batch_size}")
    print(f"Will process approximately {len(test_dataset) // batch_size} batches")
    
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

def load_and_evaluate_model(model_path, test_data_path, vocab_path):
    """Load model and evaluate it."""
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load vocabulary
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    print(f"Loading model from {model_path}")
    print(f"Using vocabulary from {vocab_path}")
    
    try:
        vocab = WordVocab.load_vocab(vocab_path)
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")
        
        # Create a new BERT model with the same architecture as used in training
        print("Initializing BERT model...")
        bert = bert_pytorch.BERT(vocab_size, hidden=128, n_layers=12, attn_heads=8, dropout=0.0)
        
        # Load the state dict from checkpoint
        print("Loading checkpoint...")
        checkpoint = torch.load(model_path, weights_only=False)
        if isinstance(checkpoint, dict):
            bert.load_state_dict(checkpoint)
            print("Loaded model state dictionary")
        else:
            bert = checkpoint
            print("Loaded complete model")
    except Exception as e:
        raise RuntimeError(f"Error loading model and vocabulary: {str(e)}") from e
        
    # Wrap the BERT model with BERTLM as done in training
    model = BERTLM(bert, vocab_size)
    
    # Create test dataloader
    test_dataloader = create_test_dataloader(test_data_path, vocab_path)
    
    # Create trainer
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
    from tqdm import tqdm
    import sys
    import signal
    
    def signal_handler(signum, frame):
        print("\nReceived interrupt signal. Cleaning up...")
        sys.exit(0)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Evaluate address-aware BERT model on test data")
    parser.add_argument('--model_dir', required=True, help='Directory containing model checkpoints')
    parser.add_argument('--test_data', required=True, help='Directory containing test data files')
    parser.add_argument('--vocab_path', required=True, help='Path to vocabulary file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', default='evaluation_results', help='Directory to save results')
    args = parser.parse_args()
    try:
        # Verify input paths
        if not os.path.exists(args.model_dir):
            raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
        if not os.path.exists(args.test_data):
            raise FileNotFoundError(f"Test data directory not found: {args.test_data}")
        if not os.path.exists(args.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {args.vocab_path}")
            
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nStarting evaluation...")
        print(f"Model directory: {args.model_dir}")
        print(f"Test data: {args.test_data}")
        print(f"Output directory: {args.output_dir}\n")
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        exit(1)
        print(f"Output directory: {args.output_dir}\n")
        
        # Initialize metrics collection
        metrics = {
            'total_loss': [], 'mlm_loss': [], 'perplexity': [], 
            'dfg_nsp_loss': [], 'cfg_nsp_loss': [],
            'dfg_nsp_acc': [], 'cfg_nsp_acc': []
        }
        
        # Find and verify checkpoints
        checkpoints = sorted([f for f in os.listdir(args.model_dir) 
                            if f.startswith('transformer_mlp.ep')])
        if not checkpoints:
            raise FileNotFoundError(f"No transformer_mlp.ep* checkpoints found in {args.model_dir}")
        
        print(f"Found {len(checkpoints)} checkpoints to evaluate")
        
        # Evaluate each epoch
        progress_bar = tqdm(checkpoints, desc="Evaluating epochs", unit="epoch")
        for checkpoint in progress_bar:
            epoch = int(checkpoint.split('ep')[-1])
            progress_bar.set_description(f"Evaluating epoch {epoch}")
            
            try:
                # Evaluate model
                epoch_metrics = load_and_evaluate_model(
                    os.path.join(args.model_dir, checkpoint),
                    args.test_data,
                    args.vocab_path
                )
                
                # Save per-epoch results
                epoch_data = {
                    'epoch': epoch,
                    'checkpoint': checkpoint,
                    **epoch_metrics
                }
                with open(os.path.join(args.output_dir, f'epoch_{epoch:02d}.json'), 'w') as f:
                    json.dump(epoch_data, f, indent=2)
                
                # Collect metrics
                for k, v in epoch_metrics.items():
                    metrics[k].append(v)
                    
                # Update progress bar with current metrics
                desc = f"Epoch {epoch:02d} - Loss: {epoch_metrics['total_loss']:.4f}"
                progress_bar.set_description(desc)
                
            except Exception as e:
                print(f"\nError evaluating epoch {epoch}: {str(e)}")
                continue
        
        # Save all metrics
        results = {
            'metrics': metrics,
            'settings': {
                'model_dir': args.model_dir,
                'test_data': args.test_data,
                'vocab_path': args.vocab_path,
                'batch_size': args.batch_size
            }
        }
        metrics_file = os.path.join(args.output_dir, 'all_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved in: {args.output_dir}")
        print(f"Per-epoch results are in: epoch_XX.json files")
        print(f"Complete metrics saved in: {metrics_file}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        sys.exit(1)

    # Initialize metrics collection
    metrics = {
        'total_loss': [], 'mlm_loss': [], 'perplexity': [], 
        'dfg_nsp_loss': [], 'cfg_nsp_loss': [],
        'dfg_nsp_acc': [], 'cfg_nsp_acc': []
    }
    
    # Find all epoch checkpoints
    checkpoints = sorted([f for f in os.listdir(args.model_dir) 
                         if f.startswith('transformer_mlp.ep')])
    
    # Evaluate each epoch
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split('ep')[-1])
        print(f"Evaluating address-aware model - epoch {epoch}")
        
        # Evaluate model
        epoch_metrics = load_and_evaluate_model(
            os.path.join(args.model_dir, checkpoint),
            args.test_data,
            args.vocab_path
        )
        
        # Save per-epoch results
        epoch_data = {
            'epoch': epoch,
            'checkpoint': checkpoint,
            **epoch_metrics
        }
        with open(os.path.join(args.output_dir, f'epoch_{epoch:02d}.json'), 'w') as f:
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
            'batch_size': args.batch_size
        }
    }
    with open(os.path.join(args.output_dir, 'all_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved in {args.output_dir}")
    print(f"Per-epoch results are saved in epoch_XX.json files")
    print(f"All metrics saved in {os.path.join(args.output_dir, 'all_metrics.json')}")