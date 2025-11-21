#!/usr/bin/env python3
"""Test improved disease detection logic"""

import sys
from pathlib import Path

# Import model
from plantdoc_inference import PlantDiseaseClassifier

def test_improved_logic():
    """Test improved disease detection"""

    print("="*80)
    print("Testing IMPROVED Disease Detection Logic")
    print("="*80)

    # Get recent images
    captures_dir = Path("captures")
    images = sorted(captures_dir.glob("scan_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]

    if not images:
        print("No images found")
        return

    # Initialize model
    print("\nInitializing PlantDoc model...")
    plantdoc = PlantDiseaseClassifier(
        model_path="plantdoc/plantdoc_edgetpu_ready_edgetpu.tflite",
        class_names_path="plantdoc/class_names.json",
        use_edgetpu=True
    )

    print("\n" + "="*80)
    print("BEFORE vs AFTER Comparison")
    print("="*80)

    threshold = 0.6  # New default
    min_confidence = 0.4  # New minimum

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}] {image_path.name}")

        # Run inference
        results, inference_time = plantdoc.predict(str(image_path), top_k=3)
        top_class, top_confidence = results[0]

        print(f"    Top: {top_class} ({top_confidence*100:.1f}%)")

        # OLD LOGIC (threshold 0.5, no min_confidence)
        disease_keywords_old = ["blight", "spot", "rust", "mildew", "virus", "mold", "bacterial", "spider"]
        is_healthy_old = top_class.lower().endswith("leaf") and \
                        not any(disease in top_class.lower() for disease in disease_keywords_old)
        is_diseased_old = not is_healthy_old and top_confidence >= 0.5

        # NEW LOGIC (threshold 0.6, min_confidence 0.4, added "scab")
        if top_confidence < min_confidence:
            is_diseased_new = False
            reason_new = f"Below min confidence ({top_confidence*100:.1f}% < 40%)"
        else:
            disease_keywords_new = ["blight", "spot", "rust", "scab", "mildew", "virus", "mold", "bacterial", "spider"]
            is_healthy_new = top_class.lower().endswith("leaf") and \
                            not any(disease in top_class.lower() for disease in disease_keywords_new)
            is_diseased_new = not is_healthy_new and top_confidence >= threshold
            reason_new = "Passed all checks"

        # Print comparison
        print(f"    OLD (threshold=0.5): {'🔴 DISEASED → GRIPPER!' if is_diseased_old else '✅ HEALTHY'}")
        print(f"    NEW (threshold=0.6, min=0.4): {'🔴 DISEASED → GRIPPER!' if is_diseased_new else f'✅ HEALTHY ({reason_new})'}")

        if is_diseased_old != is_diseased_new:
            print(f"    ⚠️ CHANGED: Old would {'activate' if is_diseased_old else 'not activate'}, New would {'activate' if is_diseased_new else 'not activate'}")

    print("\n" + "="*80)
    print("✓ Test complete!")
    print("="*80)

if __name__ == '__main__':
    test_improved_logic()
