#!/usr/bin/env python3
"""
Simple model structure verification (no transformers import needed).

Just checks the Python code structure is correct.
"""

import sys
import os

def test_file_structure():
    """Test that all files exist."""
    print("=" * 60)
    print("Test 1: File Structure")
    print("=" * 60)
    
    required_files = [
        'model_baseline.py',
        'dataloader_baseline.py',
        'train_baseline.py',
        'run_baseline.sh',
        '__init__.py',
        'README.md'
    ]
    
    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"✓ {f} exists")
        else:
            print(f"✗ {f} missing")
            missing.append(f)
    
    if missing:
        print(f"\n✗ Missing files: {missing}")
        return False
    else:
        print("\n✓ All required files present")
        return True


def test_model_structure():
    """Test that model_baseline.py has correct structure."""
    print("\n" + "=" * 60)
    print("Test 2: Model Structure")
    print("=" * 60)
    
    try:
        with open('model_baseline.py', 'r') as f:
            content = f.read()
        
        # Check for required classes and functions
        checks = {
            'BinBertModel': 'class BinBertModel(BertModel):',
            'BaselinePretrainingModel': 'class BaselinePretrainingModel(nn.Module):',
            'create_baseline_model': 'def create_baseline_model(',
            'position_embeddings trick': 'self.embeddings.position_embeddings = self.embeddings.word_embeddings',
            'BertModel import': 'from transformers import BertModel',
        }
        
        all_ok = True
        for name, check_str in checks.items():
            if check_str in content:
                print(f"✓ {name} found")
            else:
                print(f"✗ {name} NOT found")
                all_ok = False
        
        if all_ok:
            print("\n✓ Model structure correct")
            return True
        else:
            print("\n✗ Model structure incomplete")
            return False
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_dataloader_structure():
    """Test that dataloader_baseline.py has correct structure."""
    print("\n" + "=" * 60)
    print("Test 3: Dataloader Structure")
    print("=" * 60)
    
    try:
        with open('dataloader_baseline.py', 'r') as f:
            content = f.read()
        
        checks = {
            'BaselinePretrainingDataset': 'class BaselinePretrainingDataset(Dataset):',
            'create_baseline_dataloaders': 'def create_baseline_dataloaders(',
            '_apply_mlm_mask': 'def _apply_mlm_mask(',
            '_apply_jtp_mask': 'def _apply_jtp_mask(',
            '_extract_jump_positions': 'def _extract_jump_positions(',
        }
        
        all_ok = True
        for name, check_str in checks.items():
            if check_str in content:
                print(f"✓ {name} found")
            else:
                print(f"✗ {name} NOT found")
                all_ok = False
        
        if all_ok:
            print("\n✓ Dataloader structure correct")
            return True
        else:
            print("\n✗ Dataloader structure incomplete")
            return False
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_train_structure():
    """Test that train_baseline.py has correct structure."""
    print("\n" + "=" * 60)
    print("Test 4: Training Script Structure")
    print("=" * 60)
    
    try:
        with open('train_baseline.py', 'r') as f:
            content = f.read()
        
        checks = {
            'train_epoch': 'def train_epoch(',
            'validate_epoch': 'def validate_epoch(',
            'main function': 'def main():',
            'MLM loss': 'mlm_loss',
            'JTP loss': 'jtp_loss',
            'save_pretrained': 'save_pretrained',
        }
        
        all_ok = True
        for name, check_str in checks.items():
            if check_str in content:
                print(f"✓ {name} found")
            else:
                print(f"✗ {name} NOT found")
                all_ok = False
        
        if all_ok:
            print("\n✓ Training script structure correct")
            return True
        else:
            print("\n✗ Training script structure incomplete")
            return False
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_shell_script():
    """Test that run_baseline.sh is executable."""
    print("\n" + "=" * 60)
    print("Test 5: Shell Script")
    print("=" * 60)
    
    try:
        if os.path.exists('run_baseline.sh'):
            is_executable = os.access('run_baseline.sh', os.X_OK)
            
            with open('run_baseline.sh', 'r') as f:
                content = f.read()
            
            has_shebang = content.startswith('#!/bin/bash')
            has_python_call = 'python train_baseline.py' in content
            
            if is_executable:
                print("✓ run_baseline.sh is executable")
            else:
                print("✗ run_baseline.sh is NOT executable")
            
            if has_shebang:
                print("✓ Has bash shebang")
            else:
                print("✗ Missing bash shebang")
            
            if has_python_call:
                print("✓ Calls train_baseline.py")
            else:
                print("✗ Doesn't call train_baseline.py")
            
            if is_executable and has_shebang and has_python_call:
                print("\n✓ Shell script correct")
                return True
            else:
                print("\n✗ Shell script has issues")
                return False
        else:
            print("✗ run_baseline.sh not found")
            return False
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_documentation():
    """Test that documentation exists."""
    print("\n" + "=" * 60)
    print("Test 6: Documentation")
    print("=" * 60)
    
    docs = ['README.md', 'QUICKSTART.md', 'MODEL_UPDATE.md']
    
    all_ok = True
    for doc in docs:
        if os.path.exists(doc):
            size = os.path.getsize(doc)
            print(f"✓ {doc} exists ({size} bytes)")
            if size < 100:
                print(f"  ⚠ Warning: {doc} seems very small")
        else:
            print(f"✗ {doc} missing")
            all_ok = False
    
    if all_ok:
        print("\n✓ Documentation complete")
        return True
    else:
        print("\n✗ Some documentation missing")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Baseline Implementation Structure Verification")
    print("(No runtime imports - just checking code structure)")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("File Structure", test_file_structure()))
    results.append(("Model Structure", test_model_structure()))
    results.append(("Dataloader Structure", test_dataloader_structure()))
    results.append(("Training Script Structure", test_train_structure()))
    results.append(("Shell Script", test_shell_script()))
    results.append(("Documentation", test_documentation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL STRUCTURE TESTS PASSED!")
        print("✓ Code structure is correct!")
        print("\nNote: To test runtime functionality, ensure you have:")
        print("  - transformers library installed")
        print("  - torch library installed")
        print("  - Correct Python environment activated")
    else:
        print("✗ SOME TESTS FAILED!")
        print("✗ Please check the errors above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
