#!/usr/bin/env python3
"""
TMC2209 Stepper Motor Control with Calibration
Belt-driven carriage system
"""

import argparse
import sys
import time
import json
import os

try:
    import RPi.GPIO as GPIO
except Exception as exc:
    print("RPi.GPIO import failed.", file=sys.stderr)
    raise


class StepperMotor:
    """TMC2209 Stepper Motor with position tracking and calibration"""

    def __init__(self, enable_pin: int, step_pin: int, dir_pin: int,
                 config_file: str = "stepper_config.json"):
        self.enable_pin = enable_pin
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.config_file = config_file

        # Default configuration
        self.steps_per_mm = 100.0  # Will be calibrated
        self.max_position_mm = 300.0  # Will be calibrated
        self.current_step = 0  # Current position in steps
        self.is_homed = False

        # Setup GPIO
        GPIO.setup(self.enable_pin, GPIO.OUT, initial=GPIO.HIGH)  # Disabled initially
        GPIO.setup(self.step_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.dir_pin, GPIO.OUT, initial=GPIO.LOW)

        # Load configuration if exists
        self.load_config()

        print(f"Stepper Motor initialized")
        print(f"Steps/mm: {self.steps_per_mm:.2f}")
        print(f"Max position: {self.max_position_mm:.1f}mm")

    def load_config(self) -> bool:
        """Load calibration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.steps_per_mm = config.get('steps_per_mm', self.steps_per_mm)
                    self.max_position_mm = config.get('max_position_mm', self.max_position_mm)
                print(f"Loaded config from {self.config_file}")
                return True
            except Exception as e:
                print(f"Error loading config: {e}")
                return False
        return False

    def save_config(self) -> None:
        """Save calibration to file"""
        config = {
            'steps_per_mm': self.steps_per_mm,
            'max_position_mm': self.max_position_mm,
            'calibrated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {self.config_file}")

    def enable(self, enabled: bool = True) -> None:
        """Enable/disable motor (TMC2209 ENABLE is active-low)"""
        GPIO.output(self.enable_pin, GPIO.LOW if enabled else GPIO.HIGH)
        status = "enabled" if enabled else "disabled"
        print(f"Motor {status}")

    def set_direction(self, clockwise: bool) -> None:
        """Set rotation direction"""
        GPIO.output(self.dir_pin, GPIO.HIGH if clockwise else GPIO.LOW)

    def step_pulses(self, steps: int, frequency_hz: float = 1000.0) -> None:
        """Send step pulses"""
        if steps <= 0:
            return

        half_period = 1.0 / (2.0 * frequency_hz)
        for _ in range(steps):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(half_period)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(half_period)

    def move_steps(self, steps: int, frequency_hz: float = 1000.0) -> None:
        """Move by number of steps (+ or -)"""
        if steps == 0:
            return

        # Set direction
        clockwise = steps > 0
        self.set_direction(clockwise)

        # Enable motor
        self.enable(True)

        # Send pulses
        abs_steps = abs(steps)
        print(f"Moving {abs_steps} steps {'CW' if clockwise else 'CCW'} at {frequency_hz}Hz...")
        self.step_pulses(abs_steps, frequency_hz)

        # Update position
        self.current_step += steps
        print(f"Position: {self.current_step} steps ({self.get_position_mm():.1f}mm)")

    def move_mm(self, distance_mm: float, frequency_hz: float = 1000.0) -> None:
        """Move by distance in mm"""
        steps = int(distance_mm * self.steps_per_mm)

        # Safety check
        new_position_mm = self.get_position_mm() + distance_mm
        if not 0 <= new_position_mm <= self.max_position_mm:
            print(f"WARNING: Target {new_position_mm:.1f}mm out of range [0, {self.max_position_mm:.1f}]mm")
            if new_position_mm < 0:
                distance_mm = -self.get_position_mm()
            else:
                distance_mm = self.max_position_mm - self.get_position_mm()
            steps = int(distance_mm * self.steps_per_mm)
            print(f"Limiting to {distance_mm:.1f}mm ({steps} steps)")

        self.move_steps(steps, frequency_hz)

    def move_to_mm(self, target_mm: float, frequency_hz: float = 1000.0) -> None:
        """Move to absolute position in mm"""
        if not self.is_homed:
            print("ERROR: Must home first!")
            return

        if not 0 <= target_mm <= self.max_position_mm:
            print(f"ERROR: Target {target_mm:.1f}mm out of range [0, {self.max_position_mm:.1f}]mm")
            return

        current_mm = self.get_position_mm()
        distance_mm = target_mm - current_mm

        print(f"Moving from {current_mm:.1f}mm to {target_mm:.1f}mm...")
        self.move_mm(distance_mm, frequency_hz)

    def home(self) -> None:
        """Set current position as home (0mm)"""
        self.current_step = 0
        self.is_homed = True
        print("Homed - Current position set to 0mm")

    def get_position_mm(self) -> float:
        """Get current position in mm"""
        return self.current_step / self.steps_per_mm

    def calibrate(self) -> None:
        """Interactive calibration procedure"""
        print("\n" + "="*60)
        print("STEPPER MOTOR CALIBRATION")
        print("="*60)

        print("\n1. Position the carriage at the starting position (farthest from motor)")
        input("   Press Enter when ready...")

        # Home at starting position
        self.home()
        print("   ✓ Home position set")

        print("\n2. We will now measure the travel distance")
        print("   You can either:")
        print("   a) Manually move the carriage and measure the distance")
        print("   b) Use the motor to move and measure")

        method = input("\n   Choose method (a/b): ").lower()

        if method == 'a':
            # Manual measurement
            print("\n   Manually move the carriage to the END position")
            distance_mm = float(input("   Enter the distance traveled in mm: "))

            print(f"\n3. Now we'll move the motor back to verify")
            steps_to_test = [100, 200, 400, 800]

            print(f"   Testing different step counts to find steps/mm ratio...")
            print(f"   We'll move the motor and you confirm if it matches your measurement")

            for test_steps in steps_to_test:
                test_steps_per_mm = test_steps / distance_mm
                print(f"\n   Testing {test_steps} steps for {distance_mm}mm")
                print(f"   This gives {test_steps_per_mm:.2f} steps/mm")

                # Reset to start
                print("   Position carriage back at START position")
                input("   Press Enter when ready...")
                self.home()

                # Move
                self.enable(True)
                self.set_direction(True)  # Towards motor
                self.step_pulses(test_steps, frequency_hz=500)

                confirm = input(f"   Did carriage reach END position correctly? (y/n): ")
                if confirm.lower() == 'y':
                    self.steps_per_mm = test_steps_per_mm
                    self.max_position_mm = distance_mm
                    print(f"\n   ✓ Calibration complete!")
                    print(f"   Steps/mm: {self.steps_per_mm:.2f}")
                    print(f"   Max travel: {self.max_position_mm:.1f}mm")
                    break

        else:  # method == 'b'
            # Motor-driven measurement
            print("\n   We'll move the motor in small increments")
            print("   Tell us when the carriage reaches the END position")

            self.enable(True)
            self.set_direction(True)  # Towards motor

            total_steps = 0
            step_increment = 100

            while True:
                print(f"\n   Moving {step_increment} steps... (Total: {total_steps})")
                self.step_pulses(step_increment, frequency_hz=500)
                total_steps += step_increment

                response = input("   Reached END? (y/n/back): ").lower()
                if response == 'y':
                    # Measure actual distance
                    distance_mm = float(input("   Measure the actual distance traveled in mm: "))
                    self.steps_per_mm = total_steps / distance_mm
                    self.max_position_mm = distance_mm
                    print(f"\n   ✓ Calibration complete!")
                    print(f"   Steps/mm: {self.steps_per_mm:.2f}")
                    print(f"   Max travel: {self.max_position_mm:.1f}mm")
                    break
                elif response == 'back':
                    # Go back a bit
                    self.set_direction(False)
                    self.step_pulses(step_increment, frequency_hz=500)
                    total_steps -= step_increment
                    self.set_direction(True)

        # Save configuration
        save = input("\n   Save calibration? (y/n): ")
        if save.lower() == 'y':
            self.save_config()
            print("   ✓ Configuration saved!")

        # Return to home
        print("\n   Returning to home position...")
        self.enable(True)
        self.set_direction(False)
        self.step_pulses(total_steps if method == 'b' else int(distance_mm * self.steps_per_mm), frequency_hz=500)
        self.home()

    def cleanup(self) -> None:
        """Cleanup"""
        self.enable(False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TMC2209 Stepper Motor Control with Calibration",
        epilog="Examples:\n"
               "  python3 stepper_simple.py calibrate\n"
               "  python3 stepper_simple.py home\n"
               "  python3 stepper_simple.py move --distance 50\n"
               "  python3 stepper_simple.py move --position 100",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Calibrate
    subparsers.add_parser('calibrate', help='Run calibration procedure')

    # Home
    subparsers.add_parser('home', help='Set current position as home')

    # Move
    move_parser = subparsers.add_parser('move', help='Move motor')
    move_parser.add_argument('--steps', type=int, help='Move by steps')
    move_parser.add_argument('--distance', type=float, help='Move by distance in mm')
    move_parser.add_argument('--position', type=float, help='Move to absolute position in mm')
    move_parser.add_argument('--speed', type=float, default=1000.0, help='Speed in Hz (default: 1000)')

    # Position
    subparsers.add_parser('position', help='Show current position')

    # Test
    subparsers.add_parser('test', help='Run test sequence')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Pin configuration
    ENABLE_PIN = 2
    STEP_PIN = 3
    DIR_PIN = 4

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    motor = None

    try:
        motor = StepperMotor(ENABLE_PIN, STEP_PIN, DIR_PIN)

        if args.command == 'calibrate':
            motor.calibrate()

        elif args.command == 'home':
            motor.home()

        elif args.command == 'move':
            if args.steps:
                motor.move_steps(args.steps, args.speed)
            elif args.distance:
                motor.move_mm(args.distance, args.speed)
            elif args.position is not None:
                motor.move_to_mm(args.position, args.speed)
            else:
                print("ERROR: Specify --steps, --distance, or --position")
                return 1

        elif args.command == 'position':
            print(f"\n=== Position ===")
            print(f"Steps: {motor.current_step}")
            print(f"Position: {motor.get_position_mm():.1f}mm")
            print(f"Max position: {motor.max_position_mm:.1f}mm")
            print(f"Homed: {motor.is_homed}")
            print(f"Steps/mm: {motor.steps_per_mm:.2f}")

        elif args.command == 'test':
            print("\n=== Test Sequence ===")

            if not motor.is_homed:
                print("Setting home position...")
                motor.home()

            print("\n1. Move 50mm forward...")
            motor.move_mm(50, frequency_hz=800)
            time.sleep(1)

            print("\n2. Move to 100mm...")
            motor.move_to_mm(100, frequency_hz=800)
            time.sleep(1)

            print("\n3. Move back 30mm...")
            motor.move_mm(-30, frequency_hz=800)
            time.sleep(1)

            print("\n4. Return to home...")
            motor.move_to_mm(0, frequency_hz=800)

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
        if motor:
            motor.cleanup()
        GPIO.cleanup()


if __name__ == "__main__":
    sys.exit(main())
