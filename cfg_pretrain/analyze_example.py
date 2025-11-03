#!/usr/bin/env python3
"""
Analyze the 3-level embeddings for a specific BB pair example
"""

import re

class SimpleAddressTokenizer:
    """Simplified tokenizer without torch dependency"""
    
    def parse_address(self, addr_text: str):
        """Parse address tag like <addr_type:hex:bin:func>"""
        pattern = r'<(addr_\w+):(0x[0-9a-fA-F]+):([0-9.\-]+):([0-9.\-]+)>'
        match = re.match(pattern, addr_text)
        
        if match:
            return {
                'type': match.group(1),
                'hex': match.group(2),
                'binary_norm': float(match.group(3)),
                'function_norm': float(match.group(4))
            }
        return None
    
    def extract_addresses(self, bb_text: str):
        """Extract all addresses from BB text"""
        pattern = r'<addr_[^>]+>'
        addresses = []
        
        for match in re.finditer(pattern, bb_text):
            parsed = self.parse_address(match.group(0))
            if parsed:
                addresses.append(parsed)
        
        return addresses

def analyze_example():
    print("=" * 80)
    print("3-LEVEL EMBEDDING ANALYSIS WITH [SEQ] SEPARATORS")
    print("=" * 80)
    print()
    
    # The example line (note: \t represents tab separator between instructions)
    source_bb = "<addr_start:0x4020f0:0.267785:0.000000> lea rdi [ rel <addr_data:0x405150:0.661077:-3.000000> ]\tlea rax [ rel <addr_data:0x405150:0.661077:-3.000000> ]\tcmp rax rdi\tje <addr_code:0x402118:0.269055:1.000000> <addr_end:0x402103:0.268388:0.558824>"
    
    target_bb = "<addr_start:0x402118:0.269055:1.000000> retn <addr_end:0x402119:0.269087:1.000000>"
    
    print("SOURCE BB:")
    print("-" * 80)
    print(source_bb)
    print()
    
    print("TARGET BB:")
    print("-" * 80)
    print(target_bb)
    print()
    
    # Create tokenizer
    tokenizer = SimpleAddressTokenizer()
    
    # Extract addresses
    source_addrs = tokenizer.extract_addresses(source_bb)
    target_addrs = tokenizer.extract_addresses(target_bb)
    
    print("=" * 80)
    print("TOKENIZATION AND 3-LEVEL EMBEDDINGS")
    print("=" * 80)
    print()
    
    # Process source BB
    print("SOURCE BB TOKENS:")
    print("-" * 80)
    
    # Find start and end positions
    source_start_pos = (0.0, 0.0)
    source_end_pos = (0.0, 0.0)
    
    for addr in source_addrs:
        if addr['type'] == 'addr_start':
            source_start_pos = (addr['binary_norm'], addr['function_norm'])
        elif addr['type'] == 'addr_end':
            source_end_pos = (addr['binary_norm'], addr['function_norm'])
    
    # Split by tabs to get instructions
    instructions = source_bb.split('\t')
    
    addr_pattern = r'<addr_[^>]+>'
    
    position = 0
    print(f"{'Pos':<4} {'Token':<20} {'Original Tag':<35} {'Level 1':<15} {'Level 2 (Addr Pos)':<25} {'Level 3'}")
    print("-" * 120)
    
    for instr_idx, instruction in enumerate(instructions):
        instruction = instruction.strip()
        if not instruction:
            continue
        
        # Split instruction by address tags
        parts = re.split(f'({addr_pattern})', instruction)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if re.match(addr_pattern, part):
                # This is an address tag - use the type as token
                parsed = tokenizer.parse_address(part)
                if parsed:
                    addr_pos = (parsed['binary_norm'], parsed['function_norm'])
                    addr_type = parsed['type']  # addr_start, addr_end, addr_code, addr_data
                    tag_display = part[:40] + "..." if len(part) > 40 else part
                    print(f"{position:<4} {addr_type:<20} {tag_display:<35} {addr_type:<15} {str(addr_pos):<25} {position}")
                    position += 1
            else:
                # Regular tokens
                tokens = part.split()
                for token in tokens:
                    print(f"{position:<4} {token:<20} {'(instruction token)':<35} {token:<15} {'(0.0, 0.0)':<25} {position}")
                    position += 1
        
        # Add [SEQ] separator between instructions (but not after the last one)
        if instr_idx < len(instructions) - 1:
            print(f"{position:<4} {'[SEQ]':<20} {'(instruction separator)':<35} {'[SEQ]':<15} {'(0.0, 0.0)':<25} {position}")
            position += 1
    
    print()
    print("TARGET BB TOKENS:")
    print("-" * 80)
    
    # Find target start and end positions
    target_start_pos = (0.0, 0.0)
    target_end_pos = (0.0, 0.0)
    
    for addr in target_addrs:
        if addr['type'] == 'addr_start':
            target_start_pos = (addr['binary_norm'], addr['function_norm'])
        elif addr['type'] == 'addr_end':
            target_end_pos = (addr['binary_norm'], addr['function_norm'])
    
    parts = re.split(f'({addr_pattern})', target_bb)
    
    print(f"{'Pos':<4} {'Token':<20} {'Original Tag':<35} {'Level 1':<15} {'Level 2 (Addr Pos)':<25} {'Level 3'}")
    print("-" * 120)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if re.match(addr_pattern, part):
            # This is an address tag - use the type as token
            parsed = tokenizer.parse_address(part)
            if parsed:
                addr_pos = (parsed['binary_norm'], parsed['function_norm'])
                addr_type = parsed['type']  # addr_start, addr_end, addr_code, addr_data
                tag_display = part[:40] + "..." if len(part) > 40 else part
                print(f"{position:<4} {addr_type:<20} {tag_display:<35} {addr_type:<15} {str(addr_pos):<25} {position}")
                position += 1
        else:
            # Regular tokens
            tokens = part.split()
            for token in tokens:
                print(f"{position:<4} {token:<20} {'(instruction token)':<35} {token:<15} {'(0.0, 0.0)':<25} {position}")
                position += 1
    
    print()
    print("=" * 120)
    print("EXPLANATION OF 3 LEVELS:")
    print("=" * 120)
    print()
    print("LEVEL 1 - SEMANTIC (Token IDs from Vocabulary):")
    print("  • Regular instruction tokens keep their identity: 'lea', 'rdi', 'cmp', 'je', etc.")
    print("  • Address tags keep their TYPE as tokens: 'addr_start', 'addr_end', 'addr_code', 'addr_data'")
    print("  • [SEQ] token separates individual instructions within a basic block")
    print()
    print("  Examples:")
    print("    <addr_data:0x405150:0.661077:-3.000000>  →  'addr_data' token")
    print("    <addr_code:0x402118:0.269055:1.000000>   →  'addr_code' token")
    print("    <addr_start:0x4020f0:0.267785:0.000000>  →  'addr_start' token (marks BB start)")
    print("    <addr_end:0x402103:0.268388:0.558824>    →  'addr_end' token (marks BB end)")
    print("    \\t (tab character)                        →  [SEQ] token (separates instructions)")
    print()
    print("LEVEL 2 - ADDRESS POSITION (Binary & Function Normalized):")
    print("  • For each address token, extract from the original tag:")
    print("    - 3rd element (after 2nd ':') → binary_norm (position in binary, 0.0-1.0)")
    print("    - 4th element (after 3rd ':') → function_norm (position relative to function)")
    print()
    print("  • Regular instruction tokens (lea, rdi, cmp, je, etc.) → (0.0, 0.0)")
    print("  • [SEQ] tokens → (0.0, 0.0)")
    print("  • Address tokens (addr_start, addr_end, addr_code, addr_data) get their positions")
    print()
    print("  Example:")
    print("    <addr_data:0x405150:0.661077:-3.000000>")
    print("    ↓")
    print("    'addr_data' token with position (0.661077, -3.000000)")
    print("                                      ^^^^^^^^  ^^^^^^^^^")
    print("                                      binary    function")
    print()
    print("LEVEL 3 - SEQUENCE POSITION (Position in Token Sequence):")
    print("  • ALL tokens get sequential position: 0, 1, 2, 3, ...")
    print("  • This is standard positional encoding")
    print("  • Tells model the order of tokens in the sequence")
    print()
    print("=" * 120)
    print("KEY INSIGHT:")
    print("=" * 120)
    print("✓ Address tags → type name becomes the semantic token (addr_start, addr_code, addr_data, addr_end)")
    print("✓ No need for separate boundary markers - addr_start and addr_end already mark boundaries!")
    print("✓ [SEQ] tokens separate instructions (inserted at tab boundaries)")
    print("✓ Type information IS in vocabulary, allowing model to learn different behaviors per type")
    print("✓ Position information (binary_norm, function_norm) is extracted as Level 2 embeddings")
    print("✓ Regular tokens and [SEQ] get semantic + sequence, but no address position (0.0, 0.0)")
    print("✓ Clean and efficient: every token has a purpose, no redundancy!")
    print()

if __name__ == "__main__":
    analyze_example()
