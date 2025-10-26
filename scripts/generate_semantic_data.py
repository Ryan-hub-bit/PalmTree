#!/usr/bin/env python3
"""
Batch convert CFG and DFG data files to semantic format.
Minimal script to generate semantic training data for testing.
"""

import sys
from pathlib import Path
from instruction_to_semantic import process_file

def main():
    # Data directory
    data_dir = Path(__file__).parent.parent / "data"
    
    # Files to convert
    files_to_convert = [
        ("cfg_2.txt", "cfg_2_semantic.txt"),
        ("dfg_2.txt", "dfg_2_semantic.txt"),
    ]
    
    print("=" * 60)
    print("Generating Semantic Data Files")
    print("=" * 60)
    
    for input_file, output_file in files_to_convert:
        input_path = data_dir / input_file
        output_path = data_dir / output_file
        log_path = data_dir / f"{output_file.replace('.txt', '_log.txt')}"
        
        if not input_path.exists():
            print(f"⚠️  Skipping {input_file} (not found)")
            continue
        
        print(f"\n📝 Converting {input_file}")
        print(f"   → {output_file}")
        
        try:
            process_file(str(input_path), str(output_path), str(log_path))
            print(f"✅ Done! Check {log_path} for any unknown instructions")
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print("\nTo use semantic format:")
    print("1. Edit src/config.py")
    print("2. Change USE_SEMANTIC = 'semantic-only'")
    print("3. Run training: python src/train_palmtree.py")

if __name__ == "__main__":
    main()
