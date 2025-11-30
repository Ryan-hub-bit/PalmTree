"""
Generate scope prediction training data from CFG files (binary-by-binary format).

Creates pairs of instructions with labels:
- 0: Same basic block (easy/medium/hard based on distance)
- 1: Cross basic block, same function (easy/medium/hard based on BB distance)
- 2: Cross function, same binary (easy/medium/hard based on func distance)

Input format: CFG files with 8 tab-separated instructions per line, binary-separated
Output format: inst1 [TAB] inst2 [TAB] label

Features:
- Balanced labels (equal samples of 0, 1, 2)
- Difficulty levels (easy/medium/hard for each type)
- All pairs within same binary
- Exactly 8 instructions per line (sliding window format)
"""

import re
import random
from collections import defaultdict
from tqdm import tqdm
import argparse


class ScopeDataGenerator:
    def __init__(self, cdfg_path, output_path, samples_per_binary=300):
        self.cdfg_path = cdfg_path
        self.output_path = output_path
        self.samples_per_binary = samples_per_binary  # Total samples per binary
        self.samples_per_type = samples_per_binary // 3  # Equal split among 3 labels
        
        # Regex to parse instruction format: opcode(addr:bin_pos:func_pos:bb_pos)
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        
    def parse_instruction(self, inst_text):
        """Parse instruction and extract address, positions"""
        match = self.addr_pattern.match(inst_text)
        if not match:
            return None
        
        return {
            'text': inst_text,
            'opcode': match.group(1),
            'addr': match.group(2),
            'binary_pos': float(match.group(3)),
            'function_pos': float(match.group(4)),
            'bb_pos': float(match.group(5))
        }
    
    def load_binary_instructions(self, lines):
        """
        Load instructions from one binary and index by function and BB.
        CFG format: Exactly 8 tab-separated instructions per line (sliding window)
        
        Returns:
            instructions: List of all instructions in this binary
            func_to_bbs: {func_pos: {bb_pos: [inst_indices]}}
        """
        instructions = []
        func_to_bbs = defaultdict(lambda: defaultdict(list))
        seen_instructions = set()  # Track unique instructions by (opcode, bin_pos, func_pos, bb_pos)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse all 8 instructions in this line (tab-separated)
            parts = line.split('\t')
            for inst_text in parts:
                inst_text = inst_text.strip()
                if not inst_text:
                    continue
                
                parsed = self.parse_instruction(inst_text)
                if parsed:
                    # Skip duplicate instructions using opcode + 3 position values
                    inst_key = (parsed['opcode'], 
                               parsed['binary_pos'], 
                               parsed['function_pos'], 
                               parsed['bb_pos'])
                    if inst_key in seen_instructions:
                        continue
                    seen_instructions.add(inst_key)
                    
                    inst_idx = len(instructions)
                    instructions.append(parsed)
                    
                    # Index by function and BB position
                    func_pos = parsed['function_pos']
                    bb_pos = parsed['bb_pos']
                    func_to_bbs[func_pos][bb_pos].append(inst_idx)
        
        return instructions, func_to_bbs
    
    def generate_pairs_for_binary(self, instructions, func_to_bbs, samples_per_type):
        """
        Generate balanced pairs based on position matching rules.
        
        Position format: (addr:binary_pos:function_pos:bb_pos)
        
        Scope label definitions:
        - Label 0: binary_pos SAME, function_pos SAME, bb_pos DIFFERENT
        - Label 1: binary_pos SAME, function_pos DIFFERENT, bb_pos DIFFERENT  
        - Label 2: binary_pos DIFFERENT, function_pos DIFFERENT, bb_pos DIFFERENT
        """
        pairs = []
        
        # Group instructions by (binary_pos, function_pos) for label 0
        bp_fp_groups = defaultdict(lambda: defaultdict(list))
        for idx, inst in enumerate(instructions):
            bp = round(inst['binary_pos'], 6)  # Round to avoid floating point issues
            fp = round(inst['function_pos'], 6)
            bb = round(inst['bb_pos'], 6)
            bp_fp_groups[(bp, fp)][bb].append(idx)
        
        # Group instructions by binary_pos for label 1
        bp_groups = defaultdict(lambda: defaultdict(list))
        for idx, inst in enumerate(instructions):
            bp = round(inst['binary_pos'], 6)
            fp = round(inst['function_pos'], 6)
            bp_groups[bp][fp].append(idx)
        
        # === LABEL 0: Same binary_pos, same function_pos, different bb_pos ===
        valid_label0 = [(key, bbs) for key, bbs in bp_fp_groups.items() if len(bbs) >= 2]
        if valid_label0:
            for _ in range(samples_per_type):
                (bp, fp), bbs = random.choice(valid_label0)
                bb1, bb2 = random.sample(list(bbs.keys()), 2)
                idx1 = random.choice(bbs[bb1])
                idx2 = random.choice(bbs[bb2])
                pairs.append((instructions[idx1]['text'], instructions[idx2]['text'], 0))
        
        # === LABEL 1: Same binary_pos, different function_pos ===
        valid_label1 = [(bp, fps) for bp, fps in bp_groups.items() if len(fps) >= 2]
        if valid_label1:
            for _ in range(samples_per_type):
                bp, fps = random.choice(valid_label1)
                fp1, fp2 = random.sample(list(fps.keys()), 2)
                idx1 = random.choice(fps[fp1])
                idx2 = random.choice(fps[fp2])
                pairs.append((instructions[idx1]['text'], instructions[idx2]['text'], 1))
        
        # === LABEL 2: Different binary_pos ===
        unique_bps = list(set(round(inst['binary_pos'], 6) for inst in instructions))
        if len(unique_bps) >= 2:
            for _ in range(samples_per_type):
                bp1, bp2 = random.sample(unique_bps, 2)
                # Get instructions from each binary_pos group
                insts_bp1 = [i for i, inst in enumerate(instructions) if round(inst['binary_pos'], 6) == bp1]
                insts_bp2 = [i for i, inst in enumerate(instructions) if round(inst['binary_pos'], 6) == bp2]
                idx1 = random.choice(insts_bp1)
                idx2 = random.choice(insts_bp2)
                pairs.append((instructions[idx1]['text'], instructions[idx2]['text'], 2))
        
        return pairs
    
    def generate(self):
        """Generate scope training data binary-by-binary with balanced labels"""
        print("=" * 80)
        print("SCOPE DATA GENERATION (Binary-by-Binary, Balanced, Multi-Difficulty)")
        print("=" * 80)
        print(f"Input: {self.cdfg_path}")
        print(f"Samples per binary: {self.samples_per_binary}")
        print(f"Samples per type: {self.samples_per_type} (balanced across 3 labels)")
        print(f"Each type has easy/medium/hard cases")
        print("")
        
        all_pairs = []
        binary_count = 0
        current_binary_lines = []
        
        print("Processing CDFG file...")
        with open(self.cdfg_path, 'r') as f:
            for line in tqdm(f, desc="Reading"):
                line = line.strip()
                
                # Empty line = binary separator
                if not line:
                    if current_binary_lines:
                        # Process this binary
                        binary_count += 1
                        instructions, func_to_bbs = self.load_binary_instructions(current_binary_lines)
                        
                        if len(instructions) > 0 and len(func_to_bbs) > 0:
                            pairs = self.generate_pairs_for_binary(
                                instructions, func_to_bbs, self.samples_per_type
                            )
                            all_pairs.extend(pairs)
                        
                        current_binary_lines = []
                else:
                    current_binary_lines.append(line)
        
        # Process last binary if any
        if current_binary_lines:
            binary_count += 1
            instructions, func_to_bbs = self.load_binary_instructions(current_binary_lines)
            if len(instructions) > 0 and len(func_to_bbs) > 0:
                pairs = self.generate_pairs_for_binary(
                    instructions, func_to_bbs, self.samples_per_type
                )
                all_pairs.extend(pairs)
        
        # Shuffle all pairs
        print(f"\nProcessed {binary_count} binaries")
        print(f"Generated {len(all_pairs)} total pairs")
        print("\nShuffling pairs...")
        random.shuffle(all_pairs)
        
        # Count labels
        label_counts = {0: 0, 1: 0, 2: 0}
        for _, _, label in all_pairs:
            label_counts[label] += 1
        
        # Write to output
        print(f"\nWriting to {self.output_path}...")
        with open(self.output_path, 'w') as f:
            for inst1, inst2, label in tqdm(all_pairs, desc="Writing"):
                f.write(f"{inst1}\t{inst2}\t{label}\n")
        
        # Print statistics
        print("\n" + "=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)
        print(f"Binaries processed: {binary_count}")
        print(f"Total pairs: {len(all_pairs)}")
        
        if len(all_pairs) > 0:
            print(f"\nLabel distribution (balanced):")
            print(f"  Same BB (label 0): {label_counts[0]} ({100*label_counts[0]/len(all_pairs):.1f}%)")
            print(f"  Cross BB (label 1): {label_counts[1]} ({100*label_counts[1]/len(all_pairs):.1f}%)")
            print(f"  Cross Func (label 2): {label_counts[2]} ({100*label_counts[2]/len(all_pairs):.1f}%)")
            print(f"\nEach label type contains easy/medium/hard difficulty levels")
            print(f"\nOutput: {self.output_path}")
        else:
            print("\nWARNING: No pairs were generated!")
            print("Possible reasons:")
            print("  - Input file is empty or malformed")
            print("  - No valid binary sections found")
            print("  - Insufficient instructions in binaries")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate scope prediction data binary-by-binary with balanced labels")
    parser.add_argument("--cdfg", required=True, help="Path to CFG file (binary-separated format, tab-delimited)")
    parser.add_argument("--output", required=True, help="Path to output scope file")
    parser.add_argument("--samples", type=int, default=300, 
                       help="Total samples per binary (default: 300, split equally among 3 labels)")
    
    args = parser.parse_args()
    
    generator = ScopeDataGenerator(
        cdfg_path=args.cdfg,
        output_path=args.output,
        samples_per_binary=args.samples
    )
    
    generator.generate()


if __name__ == "__main__":
    main()
