#!/usr/bin/env python3
"""Test script to verify the setup works."""

import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from engines.base import BaseEngine
        print("  ✓ engines.base")
    except Exception as e:
        print(f"  ✗ engines.base: {e}")
        return False
    
    try:
        from engines.llama_cpp import LlamaCppEngine
        print("  ✓ engines.llama_cpp")
    except Exception as e:
        print(f"  ✗ engines.llama_cpp: {e}")
        return False
    
    try:
        from tuners.base import BaseTuner
        print("  ✓ tuners.base")
    except Exception as e:
        print(f"  ✗ tuners.base: {e}")
        return False
    
    try:
        from tuners.ai_tuner import AITuner
        print("  ✓ tuners.ai_tuner")
    except Exception as e:
        print(f"  ✗ tuners.ai_tuner: {e}")
        return False
    
    try:
        from tuners.grid_search import GridSearchTuner
        print("  ✓ tuners.grid_search")
    except Exception as e:
        print(f"  ✗ tuners.grid_search: {e}")
        return False
    
    try:
        from history.manager import HistoryManager
        print("  ✓ history.manager")
    except Exception as e:
        print(f"  ✗ history.manager: {e}")
        return False
    
    return True


def test_cli():
    """Test CLI argument parsing."""
    print("\nTesting CLI...")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "llm_tuner.py", "--help"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("  ✓ CLI help works")
        return True
    else:
        print(f"  ✗ CLI failed: {result.stderr}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("LLM TUNER - Setup Test")
    print("="*60)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_cli():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    
    return 0 if success else 1


if __name__ == "__main__":
    import os
    sys.exit(main())
