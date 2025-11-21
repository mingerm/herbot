#!/usr/bin/env python3
"""
Herbot Camera Capture
Simple camera interface using libcamera
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


class HerbotCamera:
    """Camera interface for Herbot using libcamera"""

    def __init__(self, output_dir: str = "captures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"Camera initialized - Output directory: {self.output_dir}")

    def capture_image(self, filename: str = None, width: int = 1920, height: int = 1080) -> str:
        """
        Capture a single image

        Args:
            filename: Output filename (auto-generated if None)
            width: Image width (default: 1920)
            height: Image height (default: 1080)

        Returns:
            Path to captured image
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"

        output_path = self.output_dir / filename

        print(f"Capturing image: {output_path}")
        print(f"Resolution: {width}x{height}")

        cmd = [
            "libcamera-still",
            "-o", str(output_path),
            "--width", str(width),
            "--height", str(height),
            "--nopreview"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Image captured: {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to capture image: {e}")
            return None

    def capture_sequence(self, count: int = 5, interval: float = 2.0,
                        width: int = 1920, height: int = 1080) -> list:
        """
        Capture a sequence of images

        Args:
            count: Number of images to capture
            interval: Time between captures in seconds
            width: Image width
            height: Image height

        Returns:
            List of captured image paths
        """
        print(f"\n=== Capturing {count} images ===")
        print(f"Interval: {interval}s")

        captured = []
        for i in range(count):
            print(f"\nCapture {i+1}/{count}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"seq_{timestamp}_{i+1:03d}.jpg"

            path = self.capture_image(filename, width, height)
            if path:
                captured.append(path)

            if i < count - 1:  # Don't wait after last capture
                print(f"Waiting {interval}s...")
                time.sleep(interval)

        print(f"\n✓ Sequence complete: {len(captured)}/{count} images captured")
        return captured

    def list_captures(self) -> list:
        """List all captured images"""
        images = sorted(self.output_dir.glob("*.jpg"))
        print(f"\n=== Captured Images ({len(images)}) ===")
        for img in images:
            size = img.stat().st_size / 1024  # KB
            print(f"  {img.name:40s} {size:8.1f} KB")
        return images


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Herbot Camera Capture",
        epilog="Examples:\n"
               "  python3 camera_capture.py capture\n"
               "  python3 camera_capture.py capture --filename test.jpg\n"
               "  python3 camera_capture.py sequence --count 10 --interval 3\n"
               "  python3 camera_capture.py list",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Capture single image
    capture_parser = subparsers.add_parser('capture', help='Capture single image')
    capture_parser.add_argument('--filename', type=str, help='Output filename (auto if not specified)')
    capture_parser.add_argument('--width', type=int, default=1920, help='Image width (default: 1920)')
    capture_parser.add_argument('--height', type=int, default=1080, help='Image height (default: 1080)')

    # Capture sequence
    seq_parser = subparsers.add_parser('sequence', help='Capture sequence of images')
    seq_parser.add_argument('--count', type=int, default=5, help='Number of images (default: 5)')
    seq_parser.add_argument('--interval', type=float, default=2.0, help='Interval in seconds (default: 2.0)')
    seq_parser.add_argument('--width', type=int, default=1920, help='Image width (default: 1920)')
    seq_parser.add_argument('--height', type=int, default=1080, help='Image height (default: 1080)')

    # List captures
    subparsers.add_parser('list', help='List captured images')

    # Preview (test)
    preview_parser = subparsers.add_parser('preview', help='Show camera preview')
    preview_parser.add_argument('--duration', type=int, default=5, help='Preview duration in seconds (default: 5)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    camera = HerbotCamera()

    try:
        if args.command == 'capture':
            camera.capture_image(args.filename, args.width, args.height)

        elif args.command == 'sequence':
            camera.capture_sequence(args.count, args.interval, args.width, args.height)

        elif args.command == 'list':
            camera.list_captures()

        elif args.command == 'preview':
            print(f"Showing preview for {args.duration} seconds...")
            cmd = ["libcamera-hello", "-t", str(args.duration * 1000), "--nopreview"]
            subprocess.run(cmd, check=True)
            print("✓ Preview complete")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
