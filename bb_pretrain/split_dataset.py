"""
Split large BB pairs file into train/val/test sets
Does NOT load entire file into memory - processes line by line
"""
import argparse
import random
import os
from tqdm import tqdm


def count_lines(filepath):
    """Count total lines in file"""
    print(f"Counting lines in {filepath}...")
    count = 0
    with open(filepath, 'r') as f:
        for _ in tqdm(f, desc="Counting"):
            count += 1
    return count


def split_file_streaming(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split large file into train/val/test without loading into memory
    
    Args:
        input_file: Path to input BB pairs file (e.g., all_bb_pairs.txt)
        output_dir: Directory to save split files
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for test (default 0.1)
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total lines
    total_lines = count_lines(input_file)
    print(f"\nTotal lines: {total_lines:,}")
    
    # Calculate split sizes
    train_size = int(total_lines * train_ratio)
    val_size = int(total_lines * val_ratio)
    test_size = total_lines - train_size - val_size
    
    print(f"\nSplit sizes:")
    print(f"  Train: {train_size:,} ({train_ratio*100:.1f}%)")
    print(f"  Val:   {val_size:,} ({val_ratio*100:.1f}%)")
    print(f"  Test:  {test_size:,} ({test_ratio*100:.1f}%)")
    
    # Output files
    train_file = os.path.join(output_dir, 'train.txt')
    val_file = os.path.join(output_dir, 'val.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    
    print(f"\nOutput files:")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    print(f"  Test:  {test_file}")
    
    # Open all output files
    with open(train_file, 'w') as f_train, \
         open(val_file, 'w') as f_val, \
         open(test_file, 'w') as f_test, \
         open(input_file, 'r') as f_in:
        
        print(f"\nSplitting file...")
        
        # Generate random assignments for each line
        # Use reservoir sampling approach for memory efficiency
        for line_idx, line in enumerate(tqdm(f_in, total=total_lines, desc="Splitting")):
            # Deterministic split based on line index and seed
            random.seed(seed + line_idx)
            rand = random.random()
            
            if rand < train_ratio:
                f_train.write(line)
            elif rand < train_ratio + val_ratio:
                f_val.write(line)
            else:
                f_test.write(line)
    
    print(f"\nDone! Files saved to {output_dir}")
    
    # Verify split sizes
    print(f"\nVerifying split sizes...")
    train_lines = count_lines(train_file)
    val_lines = count_lines(val_file)
    test_lines = count_lines(test_file)
    
    print(f"\nActual sizes:")
    print(f"  Train: {train_lines:,}")
    print(f"  Val:   {val_lines:,}")
    print(f"  Test:  {test_lines:,}")
    print(f"  Total: {train_lines + val_lines + test_lines:,}")
    
    return train_file, val_file, test_file


def split_file_with_reservoir(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                                seed=42, shuffle_buffer_size=100000):
    """
    Split with better shuffling using reservoir sampling
    More memory intensive but better randomization
    
    Args:
        shuffle_buffer_size: Size of shuffle buffer (larger = more random, more memory)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_lines = count_lines(input_file)
    print(f"\nTotal lines: {total_lines:,}")
    
    # Calculate split sizes
    train_size = int(total_lines * train_ratio)
    val_size = int(total_lines * val_ratio)
    test_size = total_lines - train_size - val_size
    
    print(f"\nSplit sizes:")
    print(f"  Train: {train_size:,} ({train_ratio*100:.1f}%)")
    print(f"  Val:   {val_size:,} ({val_ratio*100:.1f}%)")
    print(f"  Test:  {test_size:,} ({test_ratio*100:.1f}%)")
    
    # Output files
    train_file = os.path.join(output_dir, 'train.txt')
    val_file = os.path.join(output_dir, 'val.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    
    print(f"\nOutput files:")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    print(f"  Test:  {test_file}")
    
    print(f"\nUsing shuffle buffer size: {shuffle_buffer_size:,}")
    
    # Streaming split with shuffle buffer
    with open(train_file, 'w') as f_train, \
         open(val_file, 'w') as f_val, \
         open(test_file, 'w') as f_test, \
         open(input_file, 'r') as f_in:
        
        buffer = []
        
        for line in tqdm(f_in, total=total_lines, desc="Splitting"):
            buffer.append(line)
            
            # When buffer is full, shuffle and write
            if len(buffer) >= shuffle_buffer_size:
                random.shuffle(buffer)
                
                # Split buffer according to ratios
                n_train = int(len(buffer) * train_ratio)
                n_val = int(len(buffer) * val_ratio)
                
                for i, line in enumerate(buffer):
                    if i < n_train:
                        f_train.write(line)
                    elif i < n_train + n_val:
                        f_val.write(line)
                    else:
                        f_test.write(line)
                
                buffer = []
        
        # Write remaining buffer
        if buffer:
            random.shuffle(buffer)
            n_train = int(len(buffer) * train_ratio)
            n_val = int(len(buffer) * val_ratio)
            
            for i, line in enumerate(buffer):
                if i < n_train:
                    f_train.write(line)
                elif i < n_train + n_val:
                    f_val.write(line)
                else:
                    f_test.write(line)
    
    print(f"\nDone! Files saved to {output_dir}")
    
    # Verify
    print(f"\nVerifying split sizes...")
    train_lines = count_lines(train_file)
    val_lines = count_lines(val_file)
    test_lines = count_lines(test_file)
    
    print(f"\nActual sizes:")
    print(f"  Train: {train_lines:,}")
    print(f"  Val:   {val_lines:,}")
    print(f"  Test:  {test_lines:,}")
    print(f"  Total: {train_lines + val_lines + test_lines:,}")
    
    return train_file, val_file, test_file


def main():
    parser = argparse.ArgumentParser(description='Split large BB pairs file into train/val/test')
    parser.add_argument('--input', type=str, required=True,
                        help='Input BB pairs file (e.g., all_bb_pairs.txt)')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                        help='Output directory for split files')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_shuffle_buffer', action='store_true',
                        help='Use shuffle buffer for better randomization (uses more memory)')
    parser.add_argument('--shuffle_buffer_size', type=int, default=100000,
                        help='Shuffle buffer size (default: 100000)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BB Pairs File Splitter")
    print("=" * 80)
    print(f"\nInput file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    
    if not os.path.exists(args.input):
        print(f"\nError: Input file '{args.input}' not found!")
        return
    
    file_size_gb = os.path.getsize(args.input) / (1024**3)
    print(f"\nInput file size: {file_size_gb:.2f} GB")
    
    if args.use_shuffle_buffer:
        print(f"\nUsing shuffle buffer method (buffer size: {args.shuffle_buffer_size:,})")
        split_file_with_reservoir(
            input_file=args.input,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            shuffle_buffer_size=args.shuffle_buffer_size
        )
    else:
        print(f"\nUsing deterministic split method (memory efficient)")
        split_file_streaming(
            input_file=args.input,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    
    print("\n" + "=" * 80)
    print("Split complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
