#!/usr/bin/env python3
"""Test AI integration in Herbot"""

import sys
from pathlib import Path

# Import models
from herbify_inference import HerbClassifier
from plantdoc_inference import PlantDiseaseClassifier

def test_models():
    """Test both models"""

    print("="*70)
    print("Testing AI Model Integration")
    print("="*70)

    # Find a test image
    captures_dir = Path("captures")
    images = list(captures_dir.glob("*.jpg"))

    if not images:
        print("No test images found in captures/")
        return

    test_image = str(images[0])
    print(f"\nTest image: {test_image}\n")

    # Test Herbify
    print("1. Testing Herbify (Herb Species Classification)...")
    try:
        herbify = HerbClassifier(
            model_path="herbify/herbify_edgetpu_ready_edgetpu.tflite",
            class_names_path="herbify/class_names.json",
            use_edgetpu=True
        )

        results, inference_time = herbify.predict(test_image, top_k=3)

        print(f"   Inference time: {inference_time:.2f}ms ({1000/inference_time:.1f} FPS)")
        print(f"   Top predictions:")
        for i, (class_name, confidence) in enumerate(results, 1):
            print(f"     {i}. {class_name}: {confidence*100:.2f}%")

        print("   ✓ Herbify working!\n")

    except Exception as e:
        print(f"   ✗ Herbify failed: {e}\n")
        return False

    # Test PlantDoc
    print("2. Testing PlantDoc (Disease Detection)...")
    try:
        plantdoc = PlantDiseaseClassifier(
            model_path="plantdoc/plantdoc_edgetpu_ready_edgetpu.tflite",
            class_names_path="plantdoc/class_names.json",
            use_edgetpu=True
        )

        results, inference_time = plantdoc.predict(test_image, top_k=3)
        top_class, top_confidence = results[0]

        # Disease detection logic
        is_healthy = top_class.lower().endswith("leaf") and \
                    not any(disease in top_class.lower() for disease in
                           ["blight", "spot", "rust", "mildew", "virus", "mold", "bacterial", "spider"])

        is_diseased = not is_healthy and top_confidence >= 0.5

        print(f"   Inference time: {inference_time:.2f}ms ({1000/inference_time:.1f} FPS)")
        print(f"   Top prediction: {top_class} ({top_confidence*100:.2f}%)")
        print(f"   Status: {'🔴 DISEASED' if is_diseased else '✅ HEALTHY'}")
        print("   ✓ PlantDoc working!\n")

    except Exception as e:
        print(f"   ✗ PlantDoc failed: {e}\n")
        return False

    print("="*70)
    print("✓ All AI models integrated successfully!")
    print("="*70)

    return True

if __name__ == '__main__':
    success = test_models()
    sys.exit(0 if success else 1)
