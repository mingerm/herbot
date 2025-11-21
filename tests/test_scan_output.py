#!/usr/bin/env python3
"""Test scan output format with herb identification and disease detection"""

import sys
from pathlib import Path

# Import models
from herbify_inference import HerbClassifier
from plantdoc_inference import PlantDiseaseClassifier

def test_scan_output():
    """Test scan output format"""

    print("="*70)
    print("Testing Scan Output Format")
    print("="*70)

    # Find test images
    captures_dir = Path("captures")
    images = list(captures_dir.glob("*.jpg"))[:3]  # Test with 3 images

    if not images:
        print("No test images found")
        return

    # Initialize models
    print("\nInitializing models...")
    herbify = HerbClassifier(
        model_path="herbify/herbify_edgetpu_ready_edgetpu.tflite",
        class_names_path="herbify/class_names.json",
        use_edgetpu=True
    )

    plantdoc = PlantDiseaseClassifier(
        model_path="plantdoc/plantdoc_edgetpu_ready_edgetpu.tflite",
        class_names_path="plantdoc/class_names.json",
        use_edgetpu=True
    )

    print("\n" + "="*70)
    print("SIMULATED SCAN OUTPUT")
    print("="*70)

    # Simulate scan for each image
    for i, image_path in enumerate(images, 1):
        print(f"\n  [15.3s] Scan #{i:03d} @ Z={(i-1)*100:03d}mm")

        # Identify herb
        herb_results, herb_time = herbify.predict(str(image_path), top_k=1)
        if herb_results:
            herb_name, herb_confidence = herb_results[0]
            print(f"    🌿 Herb: {herb_name} ({herb_confidence*100:.1f}%)")
        else:
            print(f"    🌿 Herb: Unknown")

        # Detect disease
        disease_results, disease_time = plantdoc.predict(str(image_path), top_k=1)
        disease_class, disease_confidence = disease_results[0]

        # Disease detection logic
        is_healthy = disease_class.lower().endswith("leaf") and \
                    not any(disease in disease_class.lower() for disease in
                           ["blight", "spot", "rust", "mildew", "virus", "mold", "bacterial", "spider"])

        is_diseased = not is_healthy and disease_confidence >= 0.5

        if is_diseased:
            print(f"    🍃 Status: 🔴 DISEASED - {disease_class} ({disease_confidence*100:.1f}%, {disease_time:.1f}ms)")
            print(f"    ✂️ Removing diseased leaf...")
            print(f"    ✓ Removal complete")
        else:
            print(f"    🍃 Status: ✅ HEALTHY - {disease_class} ({disease_confidence*100:.1f}%, {disease_time:.1f}ms)")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)

if __name__ == '__main__':
    test_scan_output()
