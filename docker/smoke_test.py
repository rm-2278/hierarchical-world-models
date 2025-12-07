#!/usr/bin/env python3
"""
Smoke test for DreamerV3 and TDMPC2 models without Docker.
Tests basic imports and compatibility.
"""

import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent  # Go up two levels from docker/smoke_test.py
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_tdmpc2_imports():
    """Test TDMPC2 basic imports"""
    print("Testing TDMPC2 imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__} imported")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        
        # Test TDMPC2 model import
        from models.tdmpc2 import TDMPC2Agent
        print("  ✓ TDMPC2Agent imported successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ TDMPC2 import failed: {e}")
        return False

def test_dreamerv3_imports():
    """Test DreamerV3 basic imports"""
    print("\nTesting DreamerV3 imports...")
    try:
        try:
            import jax
            print(f"  ✓ JAX {jax.__version__} imported")
            print(f"  ✓ JAX devices: {jax.devices()}")
        except ImportError:
            print("  ⚠ JAX not installed (optional for DreamerV3)")
            return None
        
        # Test DreamerV3 model import
        from models.dreamerv3 import DreamerV3Agent
        print("  ✓ DreamerV3Agent imported successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ DreamerV3 import failed: {e}")
        return False

def test_buffer_patches():
    """Test that torch.nn.Buffer patches are applied"""
    print("\nTesting torch.nn.Buffer patches...")
    all_good = True
    
    try:
        tdmpc2_scale = project_root / "third_party" / "tdmpc2" / "tdmpc2" / "common" / "scale.py"
        tdmpc2_main = project_root / "third_party" / "tdmpc2" / "tdmpc2" / "tdmpc2.py"
        
        if tdmpc2_scale.exists():
            content = tdmpc2_scale.read_text()
            
            # Check if old Buffer import exists
            if "from torch.nn import Buffer" in content:
                print(f"  ✗ Old Buffer import found in {tdmpc2_scale}")
                print("  Run: sed -i 's/from torch.nn import Buffer/# Buffer removed in PyTorch 2.0+/' third_party/tdmpc2/tdmpc2/common/scale.py")
                all_good = False
            else:
                print(f"  ✓ Buffer import removed from scale.py")
            
            # Check if Buffer usage is patched
            if "= Buffer(" in content:
                print(f"  ✗ Unpatched Buffer usage in {tdmpc2_scale}")
                print("  Run: sed -i 's/self.value = Buffer(/self.register_buffer(\"value\", /' third_party/tdmpc2/tdmpc2/common/scale.py")
                print("  Run: sed -i 's/self._percentiles = Buffer(/self.register_buffer(\"_percentiles\", /' third_party/tdmpc2/tdmpc2/common/scale.py")
                all_good = False
            elif "register_buffer(" in content:
                print(f"  ✓ Buffer usage patched in scale.py")
            else:
                print(f"  ⚠ No Buffer usage detected in scale.py")
        
        if tdmpc2_main.exists():
            content = tdmpc2_main.read_text()
            
            # Check if torch.nn.Buffer usage is patched
            if "torch.nn.Buffer(" in content:
                print(f"  ✗ Unpatched Buffer usage in {tdmpc2_main}")
                print("  Run: sed -i 's/self._prev_mean = torch.nn.Buffer(/self.register_buffer(\"_prev_mean\", /' third_party/tdmpc2/tdmpc2/tdmpc2.py")
                all_good = False
            elif "register_buffer(" in content and "_prev_mean" in content:
                print(f"  ✓ Buffer usage patched in tdmpc2.py")
            else:
                print(f"  ⚠ No _prev_mean Buffer usage detected in tdmpc2.py")
        
        return all_good
    except Exception as e:
        print(f"  ✗ Patch check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("DreamerV3 & TDMPC2 Compatibility Smoke Test")
    print("=" * 60)
    
    results = {
        "TDMPC2": test_tdmpc2_imports(),
        "DreamerV3": test_dreamerv3_imports(),
        "Buffer Patches": test_buffer_patches()
    }
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for name, result in results.items():
        if result is True:
            print(f"  ✓ {name}: PASS")
        elif result is False:
            print(f"  ✗ {name}: FAIL")
        elif result is None:
            print(f"  ⚠ {name}: SKIP (optional dependency missing)")
    
    all_passed = all(r is not False for r in results.values())
    
    if all_passed:
        print("\n✓ All tests passed or skipped!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
