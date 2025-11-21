"""
Generate Basic Block Bucket Probe Data

This script generates instruction-level data for BB bucket prediction:
- One instruction per line (no pairing like CFG)
- Each instruction has position embeddings (binary_pos, func_pos, bb_pos)
- Follows all CFG generation rules (normalization, formatting)
- Bucket assignment based on bb_norm position within basic block

Output format:
    opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2 ...

Each line represents one instruction with its address and normalized positions.
The bucket is determined by bb_norm (position within BB).
"""

import os
import sys
import argparse
import json
import re
from collections import defaultdict

# Try to import Binary Ninja
try:
    import binaryninja as binja
    from binaryninja import load
    HAS_BINARYNINJA = True
except ImportError:
    HAS_BINARYNINJA = False
    print("WARNING: Binary Ninja not available. Some features may be limited.")

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def normalize_and_mask(ins_raw: str):
    """
    Normalize instruction and extract mask (addresses).
    Similar to cfg_address_icfg.py
    """
    ins = re.sub(r"\s+", ", ", ins_raw, 1)
    parts = ins.split(", ")
    opcode = parts[0]
    operands = parts[1:] if len(parts) > 1 else []

    out_tokens = [opcode]
    mask_tokens = ["0"]

    for op in operands:
        pieces = re.split(r"(0x[0-9A-Fa-f]+|[A-Za-z0-9_]+|\[|\]|,|:|\(|\))", op)
        for tok in pieces:
            if not tok or tok.isspace():
                continue
            if tok.startswith("0x") and bool(HEX_RE.fullmatch(tok)):
                out_tokens.append(tok)
                mask_tokens.append(tok)
            else:
                out_tokens.append(tok)
                mask_tokens.append("0")

    return " ".join(out_tokens), " ".join(mask_tokens)


def get_bb_bucket(bb_norm, num_buckets=10):
    """
    Assign instruction to a bucket based on its position within the basic block.
    
    Args:
        bb_norm: Normalized position within BB [0.0, 1.0]
        num_buckets: Number of buckets (default 10)
    
    Returns:
        bucket_id: 0 to num_buckets-1
    
    Bucketing strategy:
        Divide [0.0, 1.0] into equal buckets:
        - Bucket 0: [0.00, 0.10)
        - Bucket 1: [0.10, 0.20)
        - Bucket 2: [0.20, 0.30)
        - ...
        - Bucket 9: [0.90, 1.00]
    
    Example:
        bb_norm = 0.12 -> bucket 1
        bb_norm = 0.00 -> bucket 0
        bb_norm = 0.95 -> bucket 9
    """
    # Convert normalized position to bucket
    bucket = int(bb_norm * num_buckets)
    
    # Handle edge case where bb_norm == 1.0
    if bucket >= num_buckets:
        bucket = num_buckets - 1
    
    return bucket


def generate_bb_bucket_data(binary_path, output_file, num_buckets=10):
    """
    Generate BB bucket probe data from a binary.
    
    Args:
        binary_path: Path to binary file
        output_file: Output file path for instructions
        num_buckets: Number of buckets for position classification
    
    Returns:
        metadata: Dictionary with statistics and bucket information
    """
    if not HAS_BINARYNINJA:
        print("ERROR: Binary Ninja is required for this script")
        return None
    
    print(f"Loading binary: {binary_path}")
    bv = load(binary_path)
    if bv is None:
        print(f"ERROR: Could not load binary {binary_path}")
        return None
    
    # Get global address range
    min_addr = bv.start
    max_addr = bv.end
    print(f"Address range: 0x{min_addr:x} - 0x{max_addr:x}")
    
    # Build addr_positions map: addr -> (idx, addr, func_name, bb_start, func_start, func_end)
    # This is used to detect code vs data addresses
    addr_positions = {}
    bb_range_map = {}  # bb_start -> (bb_start, bb_end)
    bin_counter = 0
    
    print("Building address maps...")
    for func in bv.functions:
        func_start = func.start
        func_end = func.start + func.total_bytes
        
        # Build BB range map
        for bb in func.basic_blocks:
            bb_start = bb.start
            bb_end = bb.start
            curr = bb.start
            for inst in bb:
                bb_end = curr
                curr += inst[1]
            bb_range_map[bb_start] = (bb_start, bb_end)
        
        # Build instruction address map
        for bb in func.basic_blocks:
            curr = bb.start
            for inst in bb:
                addr_positions[curr] = (bin_counter, curr, func.name, bb.start, func_start, func_end)
                bin_counter += 1
                curr += inst[1]
    
    total_bin = bin_counter
    
    # Get sections for data address normalization
    sections = []
    try:
        for sec in bv.sections.values():
            sections.append((sec.start, sec.end, sec.name))
    except Exception as e:
        print(f"[WARNING] Could not read sections: {e}")
    
    def section_norm(a):
        """Normalize address within its section"""
        for sstart, send, _ in sections:
            if sstart <= a < send:
                if send - sstart > 1:
                    return (a - sstart) / float(send - sstart - 1)
                return 0.0
        return 0.0
    
    def format_positions(entry):
        """Format position string for an instruction address"""
        bin_counter, addr, func_name, bb_start, func_start, func_end = entry
        
        # Binary-level normalization
        if max_addr > min_addr:
            bnorm = (addr - min_addr) / float(max_addr - min_addr)
            bnorm = max(0.0, min(1.0, bnorm))
        else:
            bnorm = 0.0
        
        # Function-level normalization
        if func_end > func_start:
            fnorm = (addr - func_start) / float(func_end - func_start)
            fnorm = max(0.0, min(1.0, fnorm))
        else:
            fnorm = 0.0
        
        # BB-level normalization
        if bb_start in bb_range_map:
            bb_start_addr, bb_end_addr = bb_range_map[bb_start]
            if bb_end_addr > bb_start_addr:
                bbnorm = (addr - bb_start_addr) / float(bb_end_addr - bb_start_addr)
                bbnorm = max(0.0, min(1.0, bbnorm))
            else:
                bbnorm = 0.0
        else:
            bbnorm = 0.0
        
        return f"{bnorm:.8f}:{fnorm:.6f}:{bbnorm:.4f}"
    
    # Data structures for output
    instructions = []  # List of (instruction_line, bb_bucket, bb_norm, func_name)
    bb_bucket_counts = defaultdict(int)
    total_bbs = 0
    total_instructions = 0
    
    # Process each function to generate instructions
    print(f"Processing {len(bv.functions)} functions...")
    for func_idx, func in enumerate(bv.functions):
        if func_idx % 100 == 0:
            print(f"  Progress: {func_idx}/{len(bv.functions)} functions")
        
        func_start = func.start
        func_end = func.start + func.total_bytes
        
        # Process each basic block in the function
        for bb in func.basic_blocks:
            bb_start = bb.start
            bb_instructions = []
            
            # Process each instruction in the basic block
            inst_addr = bb.start
            for inst in bb:
                try:
                    # Get disassembly and normalize
                    disasm_raw = bv.get_disassembly(inst_addr)
                    if not disasm_raw or disasm_raw.strip() == "":
                        inst_addr += inst[1]
                        continue
                    
                    norm_text, mask_line = normalize_and_mask(disasm_raw)
                    
                    tokens = norm_text.strip().split()
                    masks = mask_line.strip().split()
                    if not tokens:
                        inst_addr += inst[1]
                        continue
                    
                    opcode = tokens[0]
                    operands = tokens[1:]
                    operand_masks = masks[1:]
                    
                    # Format operands with address handling
                    formatted_ops = []
                    for i, tok in enumerate(operands):
                        mk = operand_masks[i] if i < len(operand_masks) else "0"
                        mk_hex = None
                        if isinstance(mk, str) and mk.startswith('0x'):
                            mk_hex = mk
                        elif isinstance(tok, str) and tok.startswith('0x') and bool(HEX_RE.fullmatch(tok)):
                            mk_hex = tok
                        
                        if mk_hex is not None:
                            try:
                                tgt = int(mk_hex, 16)
                            except Exception:
                                tgt = None
                            
                            # Filter out immediate values (like cfg_address_icfg.py)
                            is_immediate = False
                            if tgt is not None:
                                if tgt < min_addr:  # below binary base
                                    is_immediate = True
                                elif tgt > 0xffffffffffff0000:  # large bit patterns/masks
                                    is_immediate = True
                            
                            if tgt is not None and tgt >= min_addr and not is_immediate:
                                if tgt in addr_positions:
                                    # Code address: full position info
                                    entry = addr_positions[tgt]
                                    pos = format_positions(entry)
                                    formatted_ops.append(f"address({mk_hex}:{pos})")
                                else:
                                    # Data address: binary + section norm, bb field = 0
                                    if max_addr > min_addr:
                                        bnorm = (tgt - min_addr) / float(max_addr - min_addr)
                                        bnorm = max(0.0, min(1.0, bnorm))
                                    else:
                                        bnorm = 0.0
                                    sn = section_norm(tgt)
                                    formatted_ops.append(f"address({mk_hex}:{bnorm:.8f}:{sn:.6f}:0)")
                            else:
                                # Immediate value, keep as-is
                                formatted_ops.append(mk_hex)
                        else:
                            # Not an address, keep token
                            formatted_ops.append(tok)
                    
                    # Format instruction header with position info
                    if inst_addr in addr_positions:
                        addr_hdr = f"{hex(inst_addr)}:{format_positions(addr_positions[inst_addr])}"
                    else:
                        addr_hdr = hex(inst_addr)
                    
                    # Assemble final instruction string
                    ops_join = ' '.join(formatted_ops)
                    # Fix bracket imbalance (like cfg_address_icfg.py)
                    if ops_join.count('[') > ops_join.count(']'):
                        ops_join = ops_join + ' ]'
                    
                    if ops_join.strip():
                        inst_str = f"{opcode}({addr_hdr}) {ops_join}"
                    else:
                        inst_str = f"{opcode}({addr_hdr})"
                    
                    # Get bb_norm for bucket assignment
                    entry = addr_positions[inst_addr]
                    _, addr, func_name, bb_start, func_start, func_end = entry
                    bb_start_addr, bb_end_addr = bb_range_map[bb_start]
                    if bb_end_addr > bb_start_addr:
                        bb_norm = (addr - bb_start_addr) / float(bb_end_addr - bb_start_addr)
                        bb_norm = max(0.0, min(1.0, bb_norm))
                    else:
                        bb_norm = 0.0
                    
                    # Calculate function norm for function bucket
                    if func_end > func_start:
                        func_norm = (addr - func_start) / float(func_end - func_start)
                        func_norm = max(0.0, min(1.0, func_norm))
                    else:
                        func_norm = 0.0
                    
                    # Calculate binary norm for binary bucket
                    if max_addr > min_addr:
                        binary_norm = (addr - min_addr) / float(max_addr - min_addr)
                        binary_norm = max(0.0, min(1.0, binary_norm))
                    else:
                        binary_norm = 0.0
                    
                    # Calculate BB bucket based on position within BB
                    bb_bucket = get_bb_bucket(bb_norm, num_buckets)
                    
                    # Calculate function bucket based on position within function
                    func_bucket = get_bb_bucket(func_norm, num_buckets)
                    
                    # Calculate binary bucket based on position within binary
                    binary_bucket = get_bb_bucket(binary_norm, num_buckets)
                    
                    # Store instruction with its buckets
                    bb_instructions.append((inst_str, bb_bucket, bb_norm, func_bucket, func_norm, binary_bucket, binary_norm))
                    total_instructions += 1
                    
                except Exception as e:
                    # Skip problematic instructions
                    pass
                
                # Advance to next instruction
                inst_addr += inst[1]
            
            # Add all instructions from this BB
            if len(bb_instructions) > 0:
                total_bbs += 1
                
                # Add all instructions from this BB with their respective buckets
                for inst_str, inst_bucket, inst_bb_norm, inst_func_bucket, inst_func_norm, inst_bin_bucket, inst_bin_norm in bb_instructions:
                    bb_bucket_counts[inst_bucket] += 1
                    instructions.append((inst_str, inst_bucket, inst_bb_norm, inst_func_bucket, inst_func_norm, inst_bin_bucket, inst_bin_norm, func.name))
    
    print(f"\nGenerated {total_instructions} instructions from {total_bbs} basic blocks")
    
    # Write instructions to file
    print(f"Writing instructions to: {output_file}")
    with open(output_file, 'w') as f:
        for inst_str, bb_bucket, bb_norm, func_bucket, func_norm, bin_bucket, bin_norm, func_name in instructions:
            f.write(f"{inst_str}\n")
    
    # Create metadata
    metadata = {
        'binary': os.path.basename(binary_path),
        'total_instructions': total_instructions,
        'total_bbs': total_bbs,
        'num_buckets': num_buckets,
        'bb_bucket_distribution': dict(bb_bucket_counts),
        'bucket_descriptions': {
            i: f"bb_norm [{i/num_buckets:.2f}, {(i+1)/num_buckets:.2f})" 
            for i in range(num_buckets)
        }
    }
    
    # Save labels separately
    labels_file = output_file.replace('.txt', '_labels.json')
    print(f"Writing labels to: {labels_file}")
    labels_data = {
        'instructions': [
            {
                'instruction': inst_str,
                'bb_bucket': bb_bucket,
                'bb_norm': bb_norm,
                'function_bucket': func_bucket,
                'function_norm': func_norm,
                'binary_bucket': bin_bucket,
                'binary_norm': bin_norm,
                'function': func_name
            }
            for inst_str, bb_bucket, bb_norm, func_bucket, func_norm, bin_bucket, bin_norm, func_name in instructions
        ]
    }
    with open(labels_file, 'w') as f:
        json.dump(labels_data, f, indent=2)
    
    # Print statistics
    print("\n" + "="*70)
    print("BB Bucket Distribution (by position within BB):")
    print("="*70)
    for bucket_id in range(num_buckets):
        count = bb_bucket_counts.get(bucket_id, 0)
        desc = metadata['bucket_descriptions'][bucket_id]
        pct = (count / total_instructions * 100) if total_instructions > 0 else 0
        print(f"  Bucket {bucket_id} ({desc:20s}): {count:6d} instructions ({pct:5.2f}%)")
    print("="*70)
    
    return metadata


def process_all_binaries(binary_dir, output_dir, num_buckets=10):
    """
    Process all binaries in a directory.
    
    Args:
        binary_dir: Directory containing binaries (default: /home/kun/onebinary/)
        output_dir: Output directory for instruction files
        num_buckets: Number of buckets
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in binary directory
    binary_files = [f for f in os.listdir(binary_dir) if os.path.isfile(os.path.join(binary_dir, f))]
    
    print(f"\nFound {len(binary_files)} files in {binary_dir}")
    print("="*70)
    
    all_metadata = {}
    
    for idx, binary_name in enumerate(binary_files):
        print(f"\n[{idx+1}/{len(binary_files)}] Processing: {binary_name}")
        print("-"*70)
        
        binary_path = os.path.join(binary_dir, binary_name)
        output_file = os.path.join(output_dir, f"{binary_name}_instructions.txt")
        
        metadata = generate_bb_bucket_data(binary_path, output_file, num_buckets)
        
        if metadata:
            all_metadata[binary_name] = metadata
    
    # Save combined metadata
    combined_metadata_file = os.path.join(output_dir, 'all_metadata.json')
    print(f"\n\nWriting combined metadata to: {combined_metadata_file}")
    with open(combined_metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✓ Processed {len(all_metadata)} binaries")
    print(f"✓ Output directory: {output_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Generate BB bucket probe data")
    parser.add_argument('--binary', help='Path to single binary file')
    parser.add_argument('--binary_dir', default='/home/kun/onebinary/', 
                        help='Directory containing binaries (default: /home/kun/onebinary/)')
    parser.add_argument('--output', help='Output file for single binary')
    parser.add_argument('--output_dir', default='data', 
                        help='Output directory for multiple binaries')
    parser.add_argument('--num_buckets', type=int, default=10, 
                        help='Number of BB position buckets')
    parser.add_argument('--metadata', help='Output file for metadata (JSON)')
    
    args = parser.parse_args()
    
    if not HAS_BINARYNINJA:
        print("ERROR: Binary Ninja is required but not available")
        sys.exit(1)
    
    if args.binary:
        # Process single binary
        if not args.output:
            print("ERROR: --output required when --binary is specified")
            sys.exit(1)
        
        metadata = generate_bb_bucket_data(args.binary, args.output, args.num_buckets)
        
        if metadata is None:
            print("ERROR: Failed to generate data")
            sys.exit(1)
        
        # Save metadata
        if args.metadata:
            print(f"\nWriting metadata to: {args.metadata}")
            with open(args.metadata, 'w') as f:
                json.dump(metadata, f, indent=2)
    else:
        # Process all binaries in directory
        process_all_binaries(args.binary_dir, args.output_dir, args.num_buckets)
    
    print("\n✓ Data generation complete!")


if __name__ == '__main__':
    main()
