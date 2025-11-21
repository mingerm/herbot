#!/usr/bin/env python3
"""Test Coral Edge TPU detection"""

import sys

try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
    print("✓ tflite_runtime imported successfully")
except ImportError as e:
    print(f"✗ Failed to import tflite_runtime: {e}")
    sys.exit(1)

# Try to load EdgeTPU delegate
try:
    delegate = load_delegate('libedgetpu.so.1')
    print("✓ Edge TPU delegate loaded successfully")
    print(f"  Delegate: {delegate}")
except Exception as e:
    print(f"✗ Failed to load Edge TPU delegate: {e}")
    print("\nTrying alternative library name...")
    try:
        delegate = load_delegate('libedgetpu.so.1.0')
        print("✓ Edge TPU delegate loaded (libedgetpu.so.1.0)")
    except Exception as e2:
        print(f"✗ Also failed with libedgetpu.so.1.0: {e2}")
        sys.exit(1)

print("\n=== TPU Status ===")
print("TPU is detected and ready to use!")
print("\nTo use TPU with TFLite model:")
print("  interpreter = Interpreter(")
print("      model_path='model.tflite',")
print("      experimental_delegates=[load_delegate('libedgetpu.so.1')]")
print("  )")
