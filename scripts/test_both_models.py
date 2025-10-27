#!/usr/bin/env python3
"""
Test both vanilla (raw) and semantic models across all epochs.
Runs test_vallina_model.py on:
1. Vanilla: cfg_2.txt + dfg_2.txt with data/baseoutput/transformer.ep{0-19}
2. Semantic: cfg_2_semantic.txt + dfg_2_semantic.txt with data/movoutput/transformer.ep{0-19}
"""

import os
import sys
import subprocess
from pathlib import Path

def prepare_test_dir(test_dir, cfg_source, dfg_source):
    """
    Create a test directory with cfg_test.txt and dfg_test.txt symlinks
    pointing to the actual test files.
    """
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_link = test_dir / "cfg_test.txt"
    dfg_link = test_dir / "dfg_test.txt"
    
    # Remove existing symlinks
    if cfg_link.exists() or cfg_link.is_symlink():
        cfg_link.unlink()
    if dfg_link.exists() or dfg_link.is_symlink():
        dfg_link.unlink()
    
    # Create new symlinks
    cfg_link.symlink_to(Path(cfg_source).resolve())
    dfg_link.symlink_to(Path(dfg_source).resolve())
    
    print(f"  Created symlinks in {test_dir}:")
    print(f"    cfg_test.txt -> {cfg_source}")
    print(f"    dfg_test.txt -> {dfg_source}")


def run_test(test_name, cfg_path, dfg_path, vocab_path, model_dir, output_dir):
    """Run test_vallina_model.py with specified parameters."""
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")
    print(f"CFG:    {cfg_path}")
    print(f"DFG:    {dfg_path}")
    print(f"Vocab:  {vocab_path}")
    print(f"Models: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary test directory with properly named symlinks
    test_data_dir = Path(output_dir) / "test_data_tmp"
    prepare_test_dir(test_data_dir, cfg_path, dfg_path)
    
    # Construct command - test_vallina_model.py is in src/ directory
    test_script = Path(__file__).parent.parent / "src" / "test_vallina_model.py"
    cmd = [
        "python3",
        str(test_script),
        "--test_data", str(test_data_dir),
        "--vocab_path", vocab_path,
        "--model_dir", model_dir,
        "--output_dir", output_dir,
        "--batch_size", "256",
        "--num_workers", "4",
        "--cuda"
    ]
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {test_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {test_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {test_name} failed: {e}")
        return False


def main():
    # Base paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    test_data_dir = data_dir / "test"
    src_dir = project_root / "src"
    
    # Check if test_vallina_model.py exists
    test_script = src_dir / "test_vallina_model.py"
    if not test_script.exists():
        print(f"❌ Error: test_vallina_model.py not found at {test_script}")
        sys.exit(1)
    
    results = {}
    
    # ========================================
    # ONLY Test Semantic (LD/ST/CP format)
    # Vanilla test commented out - resume from where stopped
    # ========================================
    print("⚠️  Skipping vanilla test - only testing semantic model")
    print()
    
    # ========================================
    # Test: Semantic (LD/ST/CP format)
    # ========================================
    semantic_cfg = test_data_dir / "cfg_2_semantic.txt"
    semantic_dfg = test_data_dir / "dfg_2_semantic.txt"
    semantic_vocab = data_dir / "movoutput" / "vocab"
    semantic_models = data_dir / "movoutput"
    semantic_output = project_root / "evaluation_results" / "semantic"
    
    if semantic_cfg.exists() and semantic_dfg.exists():
        results["semantic"] = run_test(
            test_name="Semantic (LD/ST/CP format)",
            cfg_path=str(semantic_cfg),
            dfg_path=str(semantic_dfg),
            vocab_path=str(semantic_vocab),
            model_dir=str(semantic_models),
            output_dir=str(semantic_output)
        )
    else:
        print(f"⚠️  Skipping semantic test - test data not found:")
        print(f"   CFG: {semantic_cfg} (exists: {semantic_cfg.exists()})")
        print(f"   DFG: {semantic_dfg} (exists: {semantic_dfg.exists()})")
        results["semantic"] = False
    
    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name:20s}: {status}")
    print(f"{'='*70}\n")
    
    # Exit with error if any test failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
