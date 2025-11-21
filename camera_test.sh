#!/bin/bash
# Herbot Camera Test Script

echo "======================================"
echo "Herbot Camera Test"
echo "======================================"
echo ""

# Create captures directory
mkdir -p captures

echo "1. Camera Info"
echo "--------------------------------------"
libcamera-hello --list-cameras
echo ""

echo "2. Quick Preview Test (5 seconds)"
echo "--------------------------------------"
echo "Starting 5-second camera preview..."
libcamera-hello -t 5000 --nopreview
echo "✓ Preview test complete"
echo ""

echo "3. Capture Test Image"
echo "--------------------------------------"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="captures/test_${TIMESTAMP}.jpg"
echo "Capturing image to: $OUTPUT_FILE"
libcamera-still -o "$OUTPUT_FILE" --width 1920 --height 1080 --nopreview
echo "✓ Image captured: $OUTPUT_FILE"
echo ""

echo "4. List captured images"
echo "--------------------------------------"
ls -lh captures/
echo ""

echo "======================================"
echo "Camera test complete!"
echo "======================================"
