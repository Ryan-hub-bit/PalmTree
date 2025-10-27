#!/usr/bin/env python3
"""
Convert test data files (cfg_2.txt and dfg_2.txt) to semantic format.
"""

from pathlib import Path
from convert_to_semantic import process_file

def main():
    # Define paths
    test_dir = Path(__file__).parent.parent / "data" / "test"
    
    files_to_convert = [
        ("cfg_2.txt", "cfg_2_semantic.txt", "cfg_semantic_log.txt"),
        ("dfg_2.txt", "dfg_2_semantic.txt", "dfg_semantic_log.txt"),
    ]
    
    print("=" * 60)
    print("Converting Test Data to Semantic Format")
    print("=" * 60)
    
    for input_file, output_file, log_file in files_to_convert:
        input_path = test_dir / input_file
        output_path = test_dir / output_file
        log_path = test_dir / log_file
        
        if not input_path.exists():
            print(f"\n⚠️  Skipping {input_file} (not found at {input_path})")
            continue
        
        print(f"\n📝 Converting {input_file}")
        print(f"   Input:  {input_path}")
        print(f"   Output: {output_path}")
        
        try:
            process_file(str(input_path), str(output_path), str(log_path))
            print(f"✅ Done! Log: {log_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
