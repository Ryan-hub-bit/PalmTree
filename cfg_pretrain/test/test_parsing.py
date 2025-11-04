"""
Simple test of data parsing without requiring PyTorch
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class SimpleAddressTokenizer:
    """Tokenizer for address format"""
    
    def parse_address(self, addr_str: str):
        """Parse address string like <addr_start:0x402118:0.269055:0.5>"""
        addr_str = addr_str.strip('<>')
        parts = addr_str.split(':')
        
        if len(parts) != 4:
            return None
        
        addr_type, hex_addr, bin_norm, func_norm = parts
        
        try:
            return {
                'type': addr_type,
                'hex': hex_addr,
                'binary_norm': float(bin_norm),
                'function_norm': float(func_norm)
            }
        except ValueError:
            return None
    
    def extract_addresses(self, text: str):
        """Extract all addresses from text"""
        pattern = r'<addr_[^>]+>'
        matches = re.findall(pattern, text)
        addresses = []
        for match in matches:
            parsed = self.parse_address(match)
            if parsed:
                addresses.append(parsed)
        return addresses


def test_parsing():
    """Test parsing BB pairs"""
    
    print("="*70)
    print("CFG Data Loader - Simple Parsing Test")
    print("="*70)
    
    data_file = '../bb_pairs_output/atilibusb.so_bb_pairs.txt'
    
    tokenizer = SimpleAddressTokenizer()
    
    print(f"\nReading from: {data_file}")
    
    # Read first few lines
    pairs_count = 0
    total_addresses = 0
    addr_type_counts = {}
    
    with open(data_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            pairs_count += 1
            
            # Extract addresses
            addresses = tokenizer.extract_addresses(line)
            total_addresses += len(addresses)
            
            for addr in addresses:
                addr_type = addr['type']
                addr_type_counts[addr_type] = addr_type_counts.get(addr_type, 0) + 1
            
            # Show first few examples
            if line_num <= 3:
                print(f"\n--- Pair {line_num} ---")
                print(f"Length: {len(line)} chars")
                print(f"Addresses found: {len(addresses)}")
                
                # Split into source and target
                split_pattern = r'(<addr_end:[^>]+>)\s+(<addr_start:[^>]+>)'
                matches = list(re.finditer(split_pattern, line))
                
                if matches:
                    match = matches[0]
                    source_bb = line[:match.end(1)].strip()
                    target_bb = line[match.start(2):].strip()
                    
                    print(f"\nSource BB ({len(source_bb)} chars):")
                    print(f"  {source_bb[:150]}...")
                    print(f"\nTarget BB ({len(target_bb)} chars):")
                    print(f"  {target_bb[:150]}...")
                    
                    # Show address details
                    if addresses:
                        print(f"\nFirst address details:")
                        addr = addresses[0]
                        print(f"  Type: {addr['type']}")
                        print(f"  Hex: {addr['hex']}")
                        print(f"  Binary norm: {addr['binary_norm']:.6f}")
                        print(f"  Function norm: {addr['function_norm']:.6f}")
            
            if line_num >= 50:
                break
    
    print(f"\n{'='*70}")
    print(f"Summary (first {pairs_count} pairs):")
    print(f"{'='*70}")
    print(f"Total pairs: {pairs_count}")
    print(f"Total addresses: {total_addresses}")
    print(f"Avg addresses per pair: {total_addresses/pairs_count:.1f}")
    
    print(f"\nAddress type distribution:")
    for addr_type, count in sorted(addr_type_counts.items()):
        print(f"  {addr_type:20s}: {count:4d} ({count/total_addresses*100:5.1f}%)")
    
    print(f"\n{'='*70}")
    print("✓ Parsing test completed!")
    print(f"{'='*70}")

if __name__ == '__main__':
    test_parsing()
