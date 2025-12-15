#!/usr/bin/env python3
"""
Verification script for address-aware PalmTree setup.

Checks:
1. Required files exist
2. Data format is correct
3. Imports work properly
4. Model can be instantiated
"""

import sys
import os
import re
from pathlib import Path

def check_files():
    """Check if all required files exist."""
    print("=" * 80)
    print("Checking Required Files")
    print("=" * 80)
    
    required_files = [
        "train_palmtree_addressaware.py",
        "TRAIN_ADDRESSAWARE.md",
        "DATA_FORMAT_EXAMPLES.md",
        "README_ADDRESSAWARE.md",
        "compare_training.sh",
        "src/palmtree/model/bert_addressaware.py",
        "src/palmtree/model/language_model_addressaware.py",
        "src/palmtree/dataset/dataset_addressaware.py",
        "src/palmtree/trainer/pretrain_addressaware.py",
        "src/palmtree/model/embedding/address_embedding.py"
    ]
    
    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    print()
    return all_exist

def check_data_format(file_path):
    """Check if data file has correct address-aware format."""
    if not Path(file_path).exists():
        print(f"  ✗ File not found: {file_path}")
        return False
    
    print(f"\n  Checking {file_path}...")
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()[:5]]  # Check first 5 lines
    
    if not lines:
        print("  ✗ File is empty")
        return False
    
    has_address_info = False
    for i, line in enumerate(lines, 1):
        # Check for opcode with address: opcode(0xADDR:float:float:float)
        if re.search(r'[a-zA-Z_]\w*\(0x[0-9a-fA-F]+:[\d.]+:[\d.]+:[\d.]+\)', line):
            has_address_info = True
            print(f"  ✓ Line {i}: Has address info")
            print(f"    {line[:80]}...")
        else:
            print(f"  ✗ Line {i}: Missing address info")
            print(f"    {line[:80]}...")
    
    return has_address_info

def check_data_files():
    """Check if data files exist and have correct format."""
    print("=" * 80)
    print("Checking Data Files")
    print("=" * 80)
    
    data_files = [
        "data/training/cdfg_bert_addressaware/cfg_train.txt",
        "data/training/cdfg_bert_addressaware/dfg_train.txt"
    ]
    
    all_valid = True
    for file in data_files:
        if not check_data_format(file):
            all_valid = False
    
    print()
    return all_valid

def check_imports():
    """Check if all imports work."""
    print("=" * 80)
    print("Checking Imports")
    print("=" * 80)
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        print("  Importing palmtree...")
        import palmtree
        print(f"  ✓ palmtree imported from: {palmtree.__file__}")
        
        print("  Importing AddressAwareBERT...")
        from palmtree.model import AddressAwareBERT
        print("  ✓ AddressAwareBERT imported")
        
        print("  Importing AddressAwareBERTLM...")
        from palmtree.model import AddressAwareBERTLM
        print("  ✓ AddressAwareBERTLM imported")
        
        print("  Importing BERTDatasetAddressAware...")
        from palmtree.dataset.dataset_addressaware import BERTDatasetAddressAware
        print("  ✓ BERTDatasetAddressAware imported")
        
        print("  Importing BERTTrainer...")
        from palmtree import trainer
        print("  ✓ BERTTrainer imported")
        
        print()
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print()
        return False

def check_model_instantiation():
    """Check if model can be instantiated."""
    print("=" * 80)
    print("Checking Model Instantiation")
    print("=" * 80)
    
    try:
        sys.path.insert(0, 'src')
        from palmtree.model import AddressAwareBERT
        
        print("  Creating AddressAwareBERT model...")
        model = AddressAwareBERT(
            vocab_size=1000,
            hidden=128,
            n_layers=2,
            attn_heads=4,
            dropout=0.1,
            use_address_embedding=True,
            use_var_embedding=True
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Total parameters: {num_params:,}")
        
        # Check embedding components
        print(f"  ✓ Address embedding: {model.use_address_embedding}")
        print(f"  ✓ Var embedding: {model.use_var_embedding}")
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Model instantiation failed: {e}")
        print()
        return False

def check_torch():
    """Check if PyTorch is available and working."""
    print("=" * 80)
    print("Checking PyTorch")
    print("=" * 80)
    
    try:
        import torch
        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print()
        return True
        
    except ImportError:
        print("  ✗ PyTorch not installed")
        print()
        return False

def main():
    """Run all checks."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Address-Aware PalmTree Setup Verification" + " " * 17 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    results = {
        "PyTorch": check_torch(),
        "Files": check_files(),
        "Data": check_data_files(),
        "Imports": check_imports(),
        "Model": check_model_instantiation()
    }
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("🎉 All checks passed! You're ready to train address-aware PalmTree.")
        print()
        print("Next steps:")
        print("  1. Review configuration in train_palmtree_addressaware.py")
        print("  2. Run: python train_palmtree_addressaware.py")
        print()
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("  - Data files: Generate using cfg_hierarchical_icfg_ida.py and dfg_hierarchical_idfg_ida.py")
        print("  - Imports: Set PYTHONPATH: export PYTHONPATH=\"${PYTHONPATH}:$(pwd)/src\"")
        print("  - PyTorch: Install with: pip install torch")
        print()
        return 1

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)  # Change to script directory
    sys.exit(main())
