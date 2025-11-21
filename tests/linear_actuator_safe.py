#!/usr/bin/env python3

import argparse
import sys
import time
import threading

try:
    import RPi.GPIO as GPIO
except Exception as exc:
    print("RPi.GPIO import failed. Install with: sudo apt install python3-rpi.gpio", file=sys.stderr)
    raise


class SafeLinearActuator:
    """
    Safe Linear Actuator controller with limit switches

    Stroke: 50mm
    """

    def __init__(self,
                 pwm_pin: int,
                 in1_pin: int,
                 in2_pin: int,
                 stby_pin: int,
                 limit_min_pin: int,
                 limit_max_pin: int,
                 stroke_mm: float = 50.0,
                 pwm_freq: int = 1000):

        self.pwm_pin = pwm_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.stby_pin = stby_pin
        self.limit_min_pin = limit_min_pin
        self.limit_max_pin = limit_max_pin
        self.stroke_mm = stroke_mm
        self.pwm_freq = pwm_freq

        # Position tracking
        self.current_position_mm = 0.0  # Will be calibrated
        self.is_homed = False
        self.emergency_stop = False

        # Setup GPIO pins
        GPIO.setup(self.pwm_pin, GPIO.OUT)
        GPIO.setup(self.in1_pin, GPIO.OUT)
        GPIO.setup(self.in2_pin, GPIO.OUT)
        GPIO.setup(self.stby_pin, GPIO.OUT)

        # Setup limit switches (pull-up, active LOW when pressed)
        GPIO.setup(self.limit_min_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.limit_max_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Initialize PWM
        self.pwm = GPIO.PWM(self.pwm_pin, self.pwm_freq)
        self.pwm.start(0)

        # Set standby to active
        GPIO.output(self.stby_pin, GPIO.HIGH)

        # Add limit switch interrupts for safety
        GPIO.add_event_detect(self.limit_min_pin, GPIO.FALLING,
                              callback=self._limit_min_callback, bouncetime=50)
        GPIO.add_event_detect(self.limit_max_pin, GPIO.FALLING,
                              callback=self._limit_max_callback, bouncetime=50)

    def _limit_min_callback(self, channel):
        """Emergency stop when minimum limit is hit"""
        print("\n!!! MIN LIMIT SWITCH TRIGGERED !!!")
        self.emergency_stop = True
        self.stop()
        self.current_position_mm = 0.0
        self.is_homed = True

    def _limit_max_callback(self, channel):
        """Emergency stop when maximum limit is hit"""
        print("\n!!! MAX LIMIT SWITCH TRIGGERED !!!")
        self.emergency_stop = True
        self.stop()
        self.current_position_mm = self.stroke_mm

    def check_limit_min(self) -> bool:
        """Check if minimum limit switch is pressed"""
        return GPIO.input(self.limit_min_pin) == GPIO.LOW

    def check_limit_max(self) -> bool:
        """Check if maximum limit switch is pressed"""
        return GPIO.input(self.limit_max_pin) == GPIO.LOW

    def get_limit_status(self) -> dict:
        """Get current limit switch status"""
        return {
            'min': self.check_limit_min(),
            'max': self.check_limit_max(),
            'position_mm': self.current_position_mm,
            'is_homed': self.is_homed
        }

    def home(self, speed: int = 30) -> bool:
        """
        Home the actuator to minimum position

        Returns:
            True if homing successful, False otherwise
        """
        print("Homing linear actuator to MIN position...")

        # If already at min, we're done
        if self.check_limit_min():
            print("Already at MIN limit")
            self.current_position_mm = 0.0
            self.is_homed = True
            return True

        # Move towards min until limit is hit
        self.emergency_stop = False
        self._move_direction("retract", speed)

        # Wait for limit or timeout (max 10 seconds)
        timeout = 10.0
        start_time = time.time()

        while not self.check_limit_min():
            if time.time() - start_time > timeout:
                print("ERROR: Homing timeout!")
                self.stop()
                return False
            time.sleep(0.01)

        self.stop()
        self.current_position_mm = 0.0
        self.is_homed = True
        print("Homing complete - Position: 0.0mm")
        return True

    def _move_direction(self, direction: str, speed: int) -> None:
        """Internal method to set motor direction and speed"""
        if direction == "extend":
            GPIO.output(self.in1_pin, GPIO.HIGH)
            GPIO.output(self.in2_pin, GPIO.LOW)
        elif direction == "retract":
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.HIGH)
        else:
            raise ValueError("Direction must be 'extend' or 'retract'")

        self.pwm.ChangeDutyCycle(speed)

    def move_to_position(self, target_mm: float, speed: int = 50, timeout: float = 10.0) -> bool:
        """
        Move to absolute position (requires homing first)

        Args:
            target_mm: Target position in mm (0 to stroke_mm)
            speed: Motor speed 0-100%
            timeout: Maximum time to wait for movement

        Returns:
            True if successful, False otherwise
        """
        if not self.is_homed:
            print("ERROR: Must home actuator first!")
            return False

        if not 0 <= target_mm <= self.stroke_mm:
            print(f"ERROR: Target {target_mm}mm out of range [0, {self.stroke_mm}]")
            return False

        delta = target_mm - self.current_position_mm

        if abs(delta) < 0.5:  # Already at target (within 0.5mm)
            print(f"Already at target position: {self.current_position_mm:.1f}mm")
            return True

        direction = "extend" if delta > 0 else "retract"
        print(f"Moving from {self.current_position_mm:.1f}mm to {target_mm:.1f}mm ({direction})")

        self.emergency_stop = False
        self._move_direction(direction, speed)

        # Estimate time needed (you'll need to calibrate this)
        # Assuming ~10mm/sec at 50% speed (adjust based on your actuator)
        estimated_time = abs(delta) / (10.0 * speed / 50.0)
        time.sleep(estimated_time)

        self.stop()

        # Update position (rough estimate - better with encoder)
        self.current_position_mm = target_mm
        print(f"Moved to position: {self.current_position_mm:.1f}mm")

        return True

    def extend(self, speed: int = 50, duration: float = 1.0) -> None:
        """Extend actuator for specified duration"""
        if self.check_limit_max():
            print("Already at MAX limit!")
            return

        print(f"Extending at {speed}% for {duration}s...")
        self.emergency_stop = False
        self._move_direction("extend", speed)

        start_time = time.time()
        while time.time() - start_time < duration:
            if self.check_limit_max() or self.emergency_stop:
                print("Limit reached during extend")
                break
            time.sleep(0.01)

        self.stop()

    def retract(self, speed: int = 50, duration: float = 1.0) -> None:
        """Retract actuator for specified duration"""
        if self.check_limit_min():
            print("Already at MIN limit!")
            return

        print(f"Retracting at {speed}% for {duration}s...")
        self.emergency_stop = False
        self._move_direction("retract", speed)

        start_time = time.time()
        while time.time() - start_time < duration:
            if self.check_limit_min() or self.emergency_stop:
                print("Limit reached during retract")
                break
            time.sleep(0.01)

        self.stop()

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
    parser = argparse.ArgumentParser(description="Safe Linear Actuator control with limit switches")

    # Pin configuration
    parser.add_argument("--pwm", type=int, default=14, help="PWM pin (default: 14)")
    parser.add_argument("--in1", type=int, default=18, help="IN1 pin (default: 18)")
    parser.add_argument("--in2", type=int, default=15, help="IN2 pin (default: 15)")
    parser.add_argument("--stby", type=int, default=23, help="STBY pin (default: 23)")
    parser.add_argument("--limit-min", type=int, default=24, help="MIN limit switch pin (default: 24)")
    parser.add_argument("--limit-max", type=int, default=25, help="MAX limit switch pin (default: 25)")

    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Status command
    subparsers.add_parser('status', help='Check limit switch status')

    # Home command
    home_parser = subparsers.add_parser('home', help='Home actuator to MIN position')
    home_parser.add_argument('--speed', type=int, default=30, help='Homing speed (default: 30)')

    # Extend command
    extend_parser = subparsers.add_parser('extend', help='Extend actuator')
    extend_parser.add_argument('--speed', type=int, default=50, help='Speed 0-100 (default: 50)')
    extend_parser.add_argument('--duration', type=float, default=1.0, help='Duration in seconds (default: 1.0)')

    # Retract command
    retract_parser = subparsers.add_parser('retract', help='Retract actuator')
    retract_parser.add_argument('--speed', type=int, default=50, help='Speed 0-100 (default: 50)')
    retract_parser.add_argument('--duration', type=float, default=1.0, help='Duration in seconds (default: 1.0)')

    # Move to position command
    move_parser = subparsers.add_parser('move', help='Move to absolute position (requires homing)')
    move_parser.add_argument('position', type=float, help='Target position in mm (0-50)')
    move_parser.add_argument('--speed', type=int, default=50, help='Speed 0-100 (default: 50)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    actuator = None

    try:
        actuator = SafeLinearActuator(
            pwm_pin=args.pwm,
            in1_pin=args.in1,
            in2_pin=args.in2,
            stby_pin=args.stby,
            limit_min_pin=args.limit_min,
            limit_max_pin=args.limit_max,
            stroke_mm=50.0
        )

        if args.command == 'status':
            status = actuator.get_limit_status()
            print(f"\n=== Linear Actuator Status ===")
            print(f"MIN limit: {'PRESSED' if status['min'] else 'Released'}")
            print(f"MAX limit: {'PRESSED' if status['max'] else 'Released'}")
            print(f"Position: {status['position_mm']:.1f}mm")
            print(f"Homed: {status['is_homed']}")

        elif args.command == 'home':
            actuator.home(speed=args.speed)

        elif args.command == 'extend':
            actuator.extend(speed=args.speed, duration=args.duration)

        elif args.command == 'retract':
            actuator.retract(speed=args.speed, duration=args.duration)

        elif args.command == 'move':
            if actuator.move_to_position(args.position, speed=args.speed):
                print("Move successful")
            else:
                print("Move failed")
                return 1

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
