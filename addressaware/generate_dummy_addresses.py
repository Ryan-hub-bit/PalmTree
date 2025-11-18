"""
Generate dummy address files for PalmTree test data.
Creates src and tgt address files with zeros to match the instruction corpus.
"""

import sys
from tqdm import tqdm

def generate_dummy_addresses(corpus_path, output_path):
    """
    For each line in corpus (8 tab-separated instructions),
    create a line with space-separated zeros matching token counts.
    """
    with open(corpus_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc=f"Processing {corpus_path.split('/')[-1]}"):
            line = line.strip()
            if not line:
                f_out.write('\n')
                continue
            
            # Split by tab to get 8 instructions
            instructions = line.split('\t')
            
            # Count tokens in each instruction and generate zeros
            all_zeros = []
            for ins in instructions:
                tokens = ins.strip().split()
                # Add a zero for each token
                all_zeros.extend(['0'] * len(tokens))
            
            # Write all zeros space-separated
            f_out.write(' '.join(all_zeros) + '\n')
    
    print(f"Created: {output_path}")


if __name__ == "__main__":
    # Test data paths
    base_path = "/home/kun/Document/PalmTree/data/test"
    
    # CFG
    cfg_corpus = f"{base_path}/cfg/all_cfg_palmtree.txt"
    cfg_src = f"{base_path}/cfg/all_cfg_src_addr.txt"
    cfg_tgt = f"{base_path}/cfg/all_cfg_tgt_addr.txt"
    
    # DFG
    dfg_corpus = f"{base_path}/dfg/all_dfg_palmtree.txt"
    dfg_src = f"{base_path}/dfg/all_dfg_src_addr.txt"
    dfg_tgt = f"{base_path}/dfg/all_dfg_tgt_addr.txt"
    
    print("Generating dummy address files for test data...")
    print("="*70)
    
    generate_dummy_addresses(cfg_corpus, cfg_src)
    generate_dummy_addresses(cfg_corpus, cfg_tgt)
    generate_dummy_addresses(dfg_corpus, dfg_src)
    generate_dummy_addresses(dfg_corpus, dfg_tgt)
    
    print("="*70)
    print("Done! Address files created with dummy zeros.")
    print("\nNote: These are dummy addresses for testing MLM/NSP only.")
    print("Address prediction tasks will not be meaningful with these files.")
