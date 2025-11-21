#!/usr/bin/env python3
"""Analyze recent scan images to check disease detection logic"""

import sys
from pathlib import Path

# Import models
from plantdoc_inference import PlantDiseaseClassifier

def analyze_images():
    """Analyze recent captured images"""

    print("="*80)
    print("Analyzing Recent Scan Images - Disease Detection Check")
    print("="*80)

    # Get recent images
    captures_dir = Path("captures")
    images = sorted(captures_dir.glob("scan_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]

    if not images:
        print("No images found")
        return

    # Initialize PlantDoc model
    print("\nInitializing PlantDoc model...")
    plantdoc = PlantDiseaseClassifier(
        model_path="plantdoc/plantdoc_edgetpu_ready_edgetpu.tflite",
        class_names_path="plantdoc/class_names.json",
        use_edgetpu=True
    )

    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)

    threshold = 0.5

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}] {image_path.name}")
        print(f"    Size: {image_path.stat().st_size / 1024:.1f} KB")

        # Run inference
        results, inference_time = plantdoc.predict(str(image_path), top_k=5)

        print(f"    Inference: {inference_time:.1f}ms")
        print(f"    Top 5 predictions:")

        for j, (class_name, confidence) in enumerate(results, 1):
            print(f"      {j}. {class_name}: {confidence*100:.1f}%")

        # Check disease detection logic
        top_class, top_confidence = results[0]

        # Current logic
        is_healthy = top_class.lower().endswith("leaf") and \
                    not any(disease in top_class.lower() for disease in
                           ["blight", "spot", "rust", "mildew", "virus", "mold", "bacterial", "spider"])

        is_diseased = not is_healthy and top_confidence >= threshold

        print(f"\n    Detection Logic:")
        print(f"      Top class: '{top_class}'")
        print(f"      Ends with 'leaf': {top_class.lower().endswith('leaf')}")
        print(f"      Has disease keyword: {not is_healthy if top_class.lower().endswith('leaf') else 'N/A'}")
        print(f"      Confidence: {top_confidence*100:.1f}% (threshold: {threshold*100:.0f}%)")

        if is_diseased:
            print(f"      Result: 🔴 DISEASED (GRIPPER WOULD ACTIVATE!)")
        else:
            print(f"      Result: ✅ HEALTHY or LOW CONFIDENCE")

        print("    " + "-"*76)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    analyze_images()
