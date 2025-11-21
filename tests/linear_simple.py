#!/usr/bin/env python3
"""
Linear Actuator Simple Control (No PWM)
Built-in limit protector - 50mm stroke
Full speed only
"""

import argparse
import sys
import time

try:
    import RPi.GPIO as GPIO
except Exception as exc:
    print("RPi.GPIO import failed.", file=sys.stderr)
    raise


class LinearActuatorSimple:
    """Linear Actuator with built-in limit protector - Simple ON/OFF control"""

    def __init__(self, enable_pin: int, in1_pin: int, in2_pin: int, stby_pin: int, stroke_mm: float = 50.0):
        self.enable_pin = enable_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.stby_pin = stby_pin
        self.stroke_mm = stroke_mm

        # Position tracking (time-based estimation)
        self.current_position_mm = 0.0
        self.is_homed = False
        self.speed_mm_per_sec = 10.0  # Approximate at full speed

        # Setup GPIO pins
        GPIO.setup(self.enable_pin, GPIO.OUT)
        GPIO.setup(self.in1_pin, GPIO.OUT)
        GPIO.setup(self.in2_pin, GPIO.OUT)
        GPIO.setup(self.stby_pin, GPIO.OUT)

        # Enable driver
        GPIO.output(self.stby_pin, GPIO.HIGH)

        print(f"Linear Actuator initialized - Stroke: {stroke_mm}mm (Full speed only)")

    def extend(self, duration: float = None, distance_mm: float = None) -> None:
        """Extend actuator"""
        if distance_mm is not None:
            duration = distance_mm / self.speed_mm_per_sec
            print(f"Extending {distance_mm}mm (~{duration:.1f}s at full speed)...")
        else:
            print(f"Extending for {duration}s at full speed...")

        GPIO.output(self.in1_pin, GPIO.HIGH)
        GPIO.output(self.in2_pin, GPIO.LOW)
        GPIO.output(self.enable_pin, GPIO.HIGH)

        time.sleep(duration)
        self.stop()

        if self.is_homed and distance_mm is not None:
            self.current_position_mm = min(self.current_position_mm + distance_mm, self.stroke_mm)
            print(f"Estimated position: {self.current_position_mm:.1f}mm")

    def retract(self, duration: float = None, distance_mm: float = None) -> None:
        """Retract actuator"""
        if distance_mm is not None:
            duration = distance_mm / self.speed_mm_per_sec
            print(f"Retracting {distance_mm}mm (~{duration:.1f}s at full speed)...")
        else:
            print(f"Retracting for {duration}s at full speed...")

        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        GPIO.output(self.enable_pin, GPIO.HIGH)

        time.sleep(duration)
        self.stop()

        if self.is_homed and distance_mm is not None:
            self.current_position_mm = max(self.current_position_mm - distance_mm, 0.0)
            print(f"Estimated position: {self.current_position_mm:.1f}mm")

    def home(self, max_time: float = 8.0) -> None:
        """Home to minimum position"""
        print(f"Homing: Retracting for {max_time}s to reach MIN limit...")
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        GPIO.output(self.enable_pin, GPIO.HIGH)

        time.sleep(max_time)
        self.stop()

        self.current_position_mm = 0.0
        self.is_homed = True
        print("Homing complete - Position: 0.0mm")

    def move_to(self, target_mm: float) -> None:
        """Move to absolute position"""
        if not self.is_homed:
            print("ERROR: Must home first!")
            return

        if not 0 <= target_mm <= self.stroke_mm:
            print(f"ERROR: Target {target_mm}mm out of range [0, {self.stroke_mm}]")
            return

        delta = target_mm - self.current_position_mm

        if abs(delta) < 0.5:
            print(f"Already at position {self.current_position_mm:.1f}mm")
            return

        if delta > 0:
            self.extend(distance_mm=delta)
        else:
            self.retract(distance_mm=abs(delta))

    def stop(self) -> None:
        """Stop motor"""
        GPIO.output(self.enable_pin, GPIO.LOW)
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.LOW)

    def get_position(self) -> dict:
        """Get position info"""
        return {
            'position_mm': self.current_position_mm,
            'is_homed': self.is_homed,
            'stroke_mm': self.stroke_mm
        }

    def cleanup(self) -> None:
        """Cleanup"""
        self.stop()
        GPIO.output(self.stby_pin, GPIO.LOW)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Linear Actuator Simple Control (no PWM, full speed only)",
        epilog="Examples:\n"
               "  python3 linear_simple.py extend --time 2\n"
               "  python3 linear_simple.py retract --distance 10\n"
               "  python3 linear_simple.py move 25\n"
               "  python3 linear_simple.py home",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Extend
    extend_parser = subparsers.add_parser('extend', help='Extend actuator')
    extend_parser.add_argument('--time', type=float, help='Duration in seconds')
    extend_parser.add_argument('--distance', type=float, help='Distance in mm')

    # Retract
    retract_parser = subparsers.add_parser('retract', help='Retract actuator')
    retract_parser.add_argument('--time', type=float, help='Duration in seconds')
    retract_parser.add_argument('--distance', type=float, help='Distance in mm')

    # Home
    home_parser = subparsers.add_parser('home', help='Home to MIN position')
    home_parser.add_argument('--max-time', type=float, default=8.0, help='Max time (default: 8s)')

    # Move
    move_parser = subparsers.add_parser('move', help='Move to position (requires homing)')
    move_parser.add_argument('position', type=float, help='Target position in mm (0-50)')

    # Position
    subparsers.add_parser('position', help='Show current position')

    # Test
    subparsers.add_parser('test', help='Run test sequence')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Pin configuration
    PINS = {'enable': 12, 'in1': 18, 'in2': 15, 'stby': 23}

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    actuator = None

    try:
        actuator = LinearActuatorSimple(
            PINS['enable'],
            PINS['in1'],
            PINS['in2'],
            PINS['stby'],
            stroke_mm=50.0
        )

        if args.command == 'extend':
            if args.distance:
                actuator.extend(distance_mm=args.distance)
            elif args.time:
                actuator.extend(duration=args.time)
            else:
                print("ERROR: Specify --time or --distance")
                return 1

        elif args.command == 'retract':
            if args.distance:
                actuator.retract(distance_mm=args.distance)
            elif args.time:
                actuator.retract(duration=args.time)
            else:
                print("ERROR: Specify --time or --distance")
                return 1

        elif args.command == 'home':
            actuator.home(max_time=args.max_time)

        elif args.command == 'move':
            actuator.move_to(args.position)

        elif args.command == 'position':
            pos = actuator.get_position()
            print(f"\n=== Position ===")
            print(f"Current: {pos['position_mm']:.1f}mm")
            print(f"Stroke: {pos['stroke_mm']:.1f}mm")
            print(f"Homed: {pos['is_homed']}")

        elif args.command == 'test':
            print("\n=== Test Sequence ===")

            print("\n1. Homing...")
            actuator.home()
            time.sleep(1)

            print("\n2. Extend 20mm...")
            actuator.extend(distance_mm=20)
            time.sleep(1)

            print("\n3. Extend to 40mm...")
            actuator.move_to(40)
            time.sleep(1)

            print("\n4. Retract to 10mm...")
            actuator.move_to(10)
            time.sleep(1)

            print("\n5. Return to home...")
            actuator.move_to(0)

            print("\nTest complete!")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted")
        return 130

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if actuator:
            actuator.cleanup()
        GPIO.cleanup()


if __name__ == "__main__":
    sys.exit(main())
