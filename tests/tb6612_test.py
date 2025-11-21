#!/usr/bin/env python3

import argparse
import sys
import time

try:
    import RPi.GPIO as GPIO
except Exception as exc:  # pragma: no cover
    print("RPi.GPIO import failed. Install with: sudo apt install python3-rpi.gpio", file=sys.stderr)
    raise


class TB6612Motor:
    """TB6612FNG motor driver controller for a single motor"""

    def __init__(self, pwm_pin: int, in1_pin: int, in2_pin: int, stby_pin: int, pwm_freq: int = 1000):
        self.pwm_pin = pwm_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.stby_pin = stby_pin
        self.pwm_freq = pwm_freq
        self.pwm = None

        # Setup GPIO pins
        GPIO.setup(self.pwm_pin, GPIO.OUT)
        GPIO.setup(self.in1_pin, GPIO.OUT)
        GPIO.setup(self.in2_pin, GPIO.OUT)
        GPIO.setup(self.stby_pin, GPIO.OUT)

        # Initialize PWM
        self.pwm = GPIO.PWM(self.pwm_pin, self.pwm_freq)
        self.pwm.start(0)

        # Set standby to active (HIGH)
        GPIO.output(self.stby_pin, GPIO.HIGH)

    def set_speed(self, speed: int, direction: str = "forward") -> None:
        """
        Set motor speed and direction

        Args:
            speed: 0-100 (percentage)
            direction: "forward", "reverse", "brake", or "stop"
        """
        print(f"DEBUG: set_speed called - speed={speed}, direction={direction}")
        print(f"DEBUG: PWM pin={self.pwm_pin}, IN1={self.in1_pin}, IN2={self.in2_pin}, STBY={self.stby_pin}")

        if not 0 <= speed <= 100:
            raise ValueError("Speed must be between 0 and 100")

        # Set direction pins based on direction
        if direction == "forward":
            GPIO.output(self.in1_pin, GPIO.HIGH)
            GPIO.output(self.in2_pin, GPIO.LOW)
            print(f"DEBUG: Set IN1=HIGH, IN2=LOW (forward)")
        elif direction == "reverse":
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.HIGH)
            print(f"DEBUG: Set IN1=LOW, IN2=HIGH (reverse)")
        elif direction == "brake":
            GPIO.output(self.in1_pin, GPIO.HIGH)
            GPIO.output(self.in2_pin, GPIO.HIGH)
            print(f"DEBUG: Set IN1=HIGH, IN2=HIGH (brake)")
        elif direction == "stop":
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.LOW)
            print(f"DEBUG: Set IN1=LOW, IN2=LOW (stop)")
        else:
            raise ValueError("Invalid direction. Use 'forward', 'reverse', 'brake', or 'stop'")

        # Set PWM duty cycle
        self.pwm.ChangeDutyCycle(speed)
        print(f"DEBUG: PWM duty cycle set to {speed}%")

    def stop(self) -> None:
        """Stop the motor"""
        self.set_speed(0, "stop")

    def brake(self) -> None:
        """Brake the motor (short brake)"""
        GPIO.output(self.in1_pin, GPIO.HIGH)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        self.pwm.ChangeDutyCycle(100)

    def cleanup(self) -> None:
        """Clean up GPIO and PWM"""
        if self.pwm:
            self.pwm.stop()
            self.pwm = None
        GPIO.output(self.stby_pin, GPIO.LOW)


def configure_gpio() -> None:
    """Initialize GPIO mode"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)


def test_motor(motor: TB6612Motor, name: str) -> None:
    """Run a simple test sequence on a motor"""
    print(f"\n=== Testing {name} ===")

    # Forward at 50% speed
    print(f"{name}: Forward at 50% speed for 2 seconds")
    motor.set_speed(50, "forward")
    time.sleep(2)

    # Stop
    print(f"{name}: Stop for 1 second")
    motor.stop()
    time.sleep(1)

    # Forward at 100% speed
    print(f"{name}: Forward at 100% speed for 2 seconds")
    motor.set_speed(100, "forward")
    time.sleep(2)

    # Brake
    print(f"{name}: Brake for 1 second")
    motor.brake()
    time.sleep(1)

    # Reverse at 50% speed
    print(f"{name}: Reverse at 50% speed for 2 seconds")
    motor.set_speed(50, "reverse")
    time.sleep(2)

    # Stop
    print(f"{name}: Stop for 1 second")
    motor.stop()
    time.sleep(1)

    # Reverse at 100% speed
    print(f"{name}: Reverse at 100% speed for 2 seconds")
    motor.set_speed(100, "reverse")
    time.sleep(2)

    # Final stop
    print(f"{name}: Final stop")
    motor.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="TB6612FNG motor driver test for Linear Actuator and DC Motor")
    parser.add_argument("--motor", choices=["linear", "dc", "both"], default="both",
                        help="Which motor to test (default: both)")
    parser.add_argument("--speed", type=int, default=50,
                        help="Speed percentage 0-100 (default: 50)")
    parser.add_argument("--direction", choices=["forward", "reverse"], default="forward",
                        help="Direction to run (default: forward)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duration in seconds (default: 2.0)")
    parser.add_argument("--test-sequence", action="store_true",
                        help="Run full test sequence instead of simple run")

    args = parser.parse_args()

    # Pin definitions
    # Linear Actuator: PWMA=GPIO12, AIN2=GPIO15, AIN1=GPIO18, STBY=GPIO23
    LINEAR_PINS = {
        'pwm': 12,
        'in1': 18,
        'in2': 15,
        'stby': 23
    }

    # DC Motor: PWMA=GPIO25, AIN2=GPIO8, AIN1=GPIO7, STBY=GPIO1
    DC_PINS = {
        'pwm': 25,
        'in1': 7,
        'in2': 8,
        'stby': 1
    }

    try:
        configure_gpio()

        linear_motor = None
        dc_motor = None

        # Initialize motors based on selection
        if args.motor in ["linear", "both"]:
            linear_motor = TB6612Motor(
                LINEAR_PINS['pwm'],
                LINEAR_PINS['in1'],
                LINEAR_PINS['in2'],
                LINEAR_PINS['stby']
            )

        if args.motor in ["dc", "both"]:
            dc_motor = TB6612Motor(
                DC_PINS['pwm'],
                DC_PINS['in1'],
                DC_PINS['in2'],
                DC_PINS['stby']
            )

        if args.test_sequence:
            # Run full test sequence
            if linear_motor:
                test_motor(linear_motor, "Linear Actuator")
            if dc_motor:
                test_motor(dc_motor, "DC Motor")
        else:
            # Simple run with specified parameters
            print(f"\nRunning motor(s) at {args.speed}% speed in {args.direction} direction for {args.duration} seconds")

            if linear_motor:
                print("Starting Linear Actuator...")
                linear_motor.set_speed(args.speed, args.direction)

            if dc_motor:
                print("Starting DC Motor...")
                dc_motor.set_speed(args.speed, args.direction)

            time.sleep(args.duration)

            print("\nStopping motor(s)...")
            if linear_motor:
                linear_motor.stop()
            if dc_motor:
                dc_motor.stop()

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
        # Cleanup
        try:
            if linear_motor:
                linear_motor.cleanup()
            if dc_motor:
                dc_motor.cleanup()
            GPIO.cleanup()
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    sys.exit(main())
