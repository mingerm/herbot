#!/usr/bin/env python3

import argparse
import sys
import time

try:
    import RPi.GPIO as GPIO
except Exception as exc:
    print("RPi.GPIO import failed. Install with: sudo apt install python3-rpi.gpio", file=sys.stderr)
    raise


class LinearActuatorBuiltIn:
    """
    Linear Actuator with built-in limit protector

    No external limit switches - relies on built-in protection
    Position tracking based on time/speed calibration

    Stroke: 50mm
    """

    def __init__(self,
                 pwm_pin: int,
                 in1_pin: int,
                 in2_pin: int,
                 stby_pin: int,
                 stroke_mm: float = 50.0,
                 pwm_freq: int = 1000,
                 speed_mm_per_sec: float = 10.0):  # Calibration parameter

        self.pwm_pin = pwm_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.stby_pin = stby_pin
        self.stroke_mm = stroke_mm
        self.pwm_freq = pwm_freq
        self.speed_mm_per_sec = speed_mm_per_sec  # Speed at 100%, needs calibration

        # Position tracking (estimated)
        self.current_position_mm = 0.0  # Start at MIN (fully retracted)
        self.is_homed = False

        # Setup GPIO pins
        GPIO.setup(self.pwm_pin, GPIO.OUT)
        GPIO.setup(self.in1_pin, GPIO.OUT)
        GPIO.setup(self.in2_pin, GPIO.OUT)
        GPIO.setup(self.stby_pin, GPIO.OUT)

        # Initialize PWM
        self.pwm = GPIO.PWM(self.pwm_pin, self.pwm_freq)
        self.pwm.start(0)

        # Set standby to active
        GPIO.output(self.stby_pin, GPIO.HIGH)

        print(f"Linear Actuator initialized - Stroke: {stroke_mm}mm")
        print(f"Built-in limit protector will prevent over-extension")

    def _move_direction(self, direction: str, speed: int) -> None:
        """Internal method to set motor direction and speed"""
        print(f"DEBUG: Setting direction={direction}, speed={speed}%")
        print(f"DEBUG: Pins - PWM:{self.pwm_pin}, IN1:{self.in1_pin}, IN2:{self.in2_pin}, STBY:{self.stby_pin}")

        if direction == "extend":
            GPIO.output(self.in1_pin, GPIO.HIGH)
            GPIO.output(self.in2_pin, GPIO.LOW)
            print(f"DEBUG: Set IN1=HIGH, IN2=LOW (extend)")
        elif direction == "retract":
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.HIGH)
            print(f"DEBUG: Set IN1=LOW, IN2=HIGH (retract)")
        else:
            raise ValueError("Direction must be 'extend' or 'retract'")

        self.pwm.ChangeDutyCycle(speed)
        print(f"DEBUG: PWM duty cycle set to {speed}%")

    def set_current_position(self, position_mm: float) -> None:
        """
        Manually set current position (for homing)

        Args:
            position_mm: Current position in mm
        """
        if not 0 <= position_mm <= self.stroke_mm:
            raise ValueError(f"Position must be between 0 and {self.stroke_mm}mm")

        self.current_position_mm = position_mm
        self.is_homed = True
        print(f"Position set to {position_mm:.1f}mm")

    def home_manual(self) -> None:
        """
        Manual homing: User confirms actuator is at MIN position

        Call this when actuator is fully retracted (0mm)
        """
        print("\n=== Manual Homing ===")
        print("Make sure the actuator is at MINIMUM (fully retracted) position")
        response = input("Is the actuator fully retracted? (y/n): ")

        if response.lower() == 'y':
            self.current_position_mm = 0.0
            self.is_homed = True
            print("Homing complete - Position set to 0.0mm")
        else:
            print("Homing cancelled")

    def home_auto(self, speed: int = 30, max_time: float = 10.0) -> None:
        """
        Automatic homing: Retract until built-in limit stops motor

        Args:
            speed: Retraction speed 0-100%
            max_time: Maximum time to attempt homing
        """
        print("\n=== Automatic Homing ===")
        print(f"Retracting at {speed}% speed until built-in limit protector stops motor...")
        print(f"Maximum time: {max_time} seconds")

        self._move_direction("retract", speed)
        time.sleep(max_time)
        self.stop()

        # Set position to 0
        self.current_position_mm = 0.0
        self.is_homed = True
        print("Homing complete - Position set to 0.0mm")

    def extend(self, distance_mm: float = None, speed: int = 50, duration: float = None) -> None:
        """
        Extend actuator by distance or for duration

        Args:
            distance_mm: Distance to extend in mm (calculates duration)
            speed: Motor speed 0-100%
            duration: Manual duration override in seconds
        """
        # Calculate duration from distance if provided
        if distance_mm is not None:
            # Estimate duration based on speed calibration
            speed_factor = speed / 100.0
            actual_speed = self.speed_mm_per_sec * speed_factor
            duration = distance_mm / actual_speed

            print(f"Extending {distance_mm}mm at {speed}% speed (~{duration:.2f}s)...")

            # Safety check
            if self.is_homed and (self.current_position_mm + distance_mm) > self.stroke_mm:
                print(f"WARNING: Target {self.current_position_mm + distance_mm:.1f}mm exceeds stroke {self.stroke_mm}mm")
                print(f"Built-in limit will stop motor at {self.stroke_mm}mm")

        elif duration is not None:
            print(f"Extending at {speed}% for {duration}s...")
        else:
            raise ValueError("Must provide either distance_mm or duration")

        self._move_direction("extend", speed)
        time.sleep(duration)
        self.stop()

        # Update estimated position
        if self.is_homed and distance_mm is not None:
            self.current_position_mm = min(self.current_position_mm + distance_mm, self.stroke_mm)
            print(f"Estimated position: {self.current_position_mm:.1f}mm")

    def retract(self, distance_mm: float = None, speed: int = 50, duration: float = None) -> None:
        """
        Retract actuator by distance or for duration

        Args:
            distance_mm: Distance to retract in mm (calculates duration)
            speed: Motor speed 0-100%
            duration: Manual duration override in seconds
        """
        # Calculate duration from distance if provided
        if distance_mm is not None:
            # Estimate duration based on speed calibration
            speed_factor = speed / 100.0
            actual_speed = self.speed_mm_per_sec * speed_factor
            duration = distance_mm / actual_speed

            print(f"Retracting {distance_mm}mm at {speed}% speed (~{duration:.2f}s)...")

            # Safety check
            if self.is_homed and (self.current_position_mm - distance_mm) < 0:
                print(f"WARNING: Target {self.current_position_mm - distance_mm:.1f}mm below 0mm")
                print(f"Built-in limit will stop motor at 0mm")

        elif duration is not None:
            print(f"Retracting at {speed}% for {duration}s...")
        else:
            raise ValueError("Must provide either distance_mm or duration")

        self._move_direction("retract", speed)
        time.sleep(duration)
        self.stop()

        # Update estimated position
        if self.is_homed and distance_mm is not None:
            self.current_position_mm = max(self.current_position_mm - distance_mm, 0.0)
            print(f"Estimated position: {self.current_position_mm:.1f}mm")

    def move_to(self, target_mm: float, speed: int = 50) -> bool:
        """
        Move to absolute position

        Args:
            target_mm: Target position in mm (0 to stroke_mm)
            speed: Motor speed 0-100%

        Returns:
            True if movement attempted, False if not homed
        """
        if not self.is_homed:
            print("ERROR: Must home actuator first!")
            return False

        if not 0 <= target_mm <= self.stroke_mm:
            print(f"ERROR: Target {target_mm}mm out of range [0, {self.stroke_mm}]")
            return False

        delta = target_mm - self.current_position_mm

        if abs(delta) < 0.5:  # Already at target
            print(f"Already at target position: {self.current_position_mm:.1f}mm")
            return True

        if delta > 0:
            self.extend(distance_mm=delta, speed=speed)
        else:
            self.retract(distance_mm=abs(delta), speed=speed)

        return True

    def get_position(self) -> dict:
        """Get current position information"""
        return {
            'position_mm': self.current_position_mm,
            'is_homed': self.is_homed,
            'stroke_mm': self.stroke_mm
        }

    def stop(self) -> None:
        """Stop the motor"""
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.LOW)
        self.pwm.ChangeDutyCycle(0)

    def cleanup(self) -> None:
        """Clean up GPIO and PWM"""
        self.stop()
        if self.pwm:
            self.pwm.stop()
            self.pwm = None
        GPIO.output(self.stby_pin, GPIO.LOW)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Linear Actuator control with built-in limit protector",
        epilog="Example: python3 linear_actuator_builtin.py extend --distance 20 --speed 50"
    )

    # Pin configuration
    parser.add_argument("--pwm", type=int, default=12, help="PWM pin (default: 12)")
    parser.add_argument("--in1", type=int, default=18, help="IN1 pin (default: 18)")
    parser.add_argument("--in2", type=int, default=15, help="IN2 pin (default: 15)")
    parser.add_argument("--stby", type=int, default=23, help="STBY pin (default: 23)")

    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Position command
    subparsers.add_parser('position', help='Show current position')

    # Home commands
    home_manual_parser = subparsers.add_parser('home-manual', help='Manual homing (confirm at MIN position)')

    home_auto_parser = subparsers.add_parser('home-auto', help='Auto homing (retract to limit)')
    home_auto_parser.add_argument('--speed', type=int, default=30, help='Homing speed (default: 30)')
    home_auto_parser.add_argument('--max-time', type=float, default=8.0, help='Max time in seconds (default: 8)')

    # Extend command
    extend_parser = subparsers.add_parser('extend', help='Extend actuator')
    extend_parser.add_argument('--distance', type=float, help='Distance in mm to extend')
    extend_parser.add_argument('--time', type=float, help='Time in seconds (alternative to distance)')
    extend_parser.add_argument('--speed', type=int, default=50, help='Speed 0-100 (default: 50)')

    # Retract command
    retract_parser = subparsers.add_parser('retract', help='Retract actuator')
    retract_parser.add_argument('--distance', type=float, help='Distance in mm to retract')
    retract_parser.add_argument('--time', type=float, help='Time in seconds (alternative to distance)')
    retract_parser.add_argument('--speed', type=int, default=50, help='Speed 0-100 (default: 50)')

    # Move to position command
    move_parser = subparsers.add_parser('move', help='Move to absolute position (requires homing)')
    move_parser.add_argument('position', type=float, help='Target position in mm (0-50)')
    move_parser.add_argument('--speed', type=int, default=50, help='Speed 0-100 (default: 50)')

    # Calibrate command
    calibrate_parser = subparsers.add_parser('calibrate', help='Calibrate speed (full stroke timing)')
    calibrate_parser.add_argument('--speed', type=int, default=100, help='Test speed (default: 100)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    actuator = None

    try:
        actuator = LinearActuatorBuiltIn(
            pwm_pin=args.pwm,
            in1_pin=args.in1,
            in2_pin=args.in2,
            stby_pin=args.stby,
            stroke_mm=50.0,
            speed_mm_per_sec=10.0  # Default calibration
        )

        if args.command == 'position':
            pos = actuator.get_position()
            print(f"\n=== Position Status ===")
            print(f"Current position: {pos['position_mm']:.1f}mm")
            print(f"Stroke: {pos['stroke_mm']:.1f}mm")
            print(f"Homed: {pos['is_homed']}")

        elif args.command == 'home-manual':
            actuator.home_manual()

        elif args.command == 'home-auto':
            actuator.home_auto(speed=args.speed, max_time=args.max_time)

        elif args.command == 'extend':
            if args.distance:
                actuator.extend(distance_mm=args.distance, speed=args.speed)
            elif args.time:
                actuator.extend(duration=args.time, speed=args.speed)
            else:
                print("ERROR: Must specify --distance or --time")
                return 1

        elif args.command == 'retract':
            if args.distance:
                actuator.retract(distance_mm=args.distance, speed=args.speed)
            elif args.time:
                actuator.retract(duration=args.time, speed=args.speed)
            else:
                print("ERROR: Must specify --distance or --time")
                return 1

        elif args.command == 'move':
            if not actuator.move_to(args.position, speed=args.speed):
                return 1

        elif args.command == 'calibrate':
            print("\n=== Calibration Mode ===")
            print("This will move the actuator full stroke to measure speed")
            print(f"Test speed: {args.speed}%")
            response = input("Continue? (y/n): ")

            if response.lower() != 'y':
                print("Calibration cancelled")
                return 0

            print("\n1. First, homing to MIN position...")
            actuator.home_auto(speed=30, max_time=8.0)

            print("\n2. Extending to MAX position...")
            print("   Timing the movement...")
            start_time = time.time()
            actuator._move_direction("extend", args.speed)

            print(f"   Running for estimated time... (press Ctrl+C when fully extended)")
            time.sleep(8.0)  # Max time

            actuator.stop()
            elapsed = time.time() - start_time

            measured_speed = 50.0 / elapsed * (args.speed / 100.0)
            print(f"\n3. Results:")
            print(f"   Time elapsed: {elapsed:.2f}s")
            print(f"   Measured speed at {args.speed}%: {measured_speed:.2f} mm/s")
            print(f"\n   Update your code with: speed_mm_per_sec={measured_speed:.2f}")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
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
