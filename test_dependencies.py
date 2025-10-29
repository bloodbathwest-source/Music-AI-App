#!/usr/bin/env python3
"""
Test script to validate dependency resolution and package compatibility.
This script verifies that all main dependencies are correctly installed and importable.
"""

import sys

def test_dependencies():
    """Test that all main dependencies can be imported."""
    print("Testing dependency imports...")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test main dependencies
    dependencies = [
        ('streamlit', 'Streamlit'),
        ('torch', 'PyTorch'),
        ('matplotlib', 'Matplotlib'),
        ('midiutil', 'MIDIUtil'),
        ('pydub', 'PyDub'),
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
    ]
    
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name}: {version}")
            tests_passed += 1
        except ImportError as e:
            print(f"✗ {display_name}: Failed to import - {e}")
            tests_failed += 1
        except Exception as e:
            print(f"✗ {display_name}: Error - {e}")
            tests_failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print(f"{'='*60}\n")
    
    if tests_failed > 0:
        print("❌ Dependency test failed!")
        return False
    else:
        print("✅ All dependencies are correctly installed and importable!")
        return True

def test_compatibility():
    """Test basic compatibility between packages."""
    print("\nTesting package compatibility...")
    
    try:
        import numpy as np
        import torch
        import tensorflow as tf
        
        # Test numpy array creation
        arr = np.array([1, 2, 3])
        print(f"✓ NumPy array creation works: {arr}")
        
        # Test PyTorch tensor creation
        tensor = torch.tensor([1, 2, 3])
        print(f"✓ PyTorch tensor creation works: {tensor}")
        
        # Test TensorFlow constant creation
        const = tf.constant([1, 2, 3])
        print(f"✓ TensorFlow constant creation works: {const}")
        
        print("\n✅ Package compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Package compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Dependency Resolution Validation Test")
    print("="*60)
    print()
    
    deps_ok = test_dependencies()
    compat_ok = test_compatibility()
    
    if deps_ok and compat_ok:
        print("\n" + "="*60)
        print("✅ All tests passed! Dependencies are correctly resolved.")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed. Please check the output above.")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
