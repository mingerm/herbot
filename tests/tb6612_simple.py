#!/usr/bin/env python3
"""
TB6612 Motor Driver - Simple version without PWM
Uses direct GPIO HIGH/LOW for full speed control
"""

import argparse
import sys
import time

try:
    import RPi.GPIO as GPIO
except Exception as exc:
    print("RPi.GPIO import failed.", file=sys.stderr)
    raise


class TB6612MotorSimple:
    """TB6612FNG motor driver - Simple ON/OFF control without PWM"""

    def __init__(self, enable_pin: int, in1_pin: int, in2_pin: int, stby_pin: int):
        self.enable_pin = enable_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.stby_pin = stby_pin

        # Setup GPIO pins
        GPIO.setup(self.enable_pin, GPIO.OUT)
        GPIO.setup(self.in1_pin, GPIO.OUT)
        GPIO.setup(self.in2_pin, GPIO.OUT)
        GPIO.setup(self.stby_pin, GPIO.OUT)

        # Enable driver
        GPIO.output(self.stby_pin, GPIO.HIGH)

    def forward(self) -> None:
        """Run motor forward at full speed"""
        GPIO.output(self.in1_pin, GPIO.HIGH)
        GPIO.output(self.in2_pin, GPIO.LOW)
        GPIO.output(self.enable_pin, GPIO.HIGH)
        print("Motor: FORWARD (full speed)")

    def reverse(self) -> None:
        """Run motor reverse at full speed"""
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        GPIO.output(self.enable_pin, GPIO.HIGH)
        print("Motor: REVERSE (full speed)")

    def stop(self) -> None:
        """Stop motor"""
        GPIO.output(self.enable_pin, GPIO.LOW)
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.LOW)
        print("Motor: STOP")

    def brake(self) -> None:
        """Brake motor"""
        GPIO.output(self.in1_pin, GPIO.HIGH)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        GPIO.output(self.enable_pin, GPIO.HIGH)
        print("Motor: BRAKE")

    def cleanup(self) -> None:
        """Cleanup GPIO"""
        self.stop()
        GPIO.output(self.stby_pin, GPIO.LOW)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple TB6612 motor control (no PWM - full speed only)",
        epilog="Examples:\n"
               "  python3 tb6612_simple.py forward --motor linear --duration 3\n"
               "  python3 tb6612_simple.py reverse --motor dc\n"
               "  python3 tb6612_simple.py test --motor both",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["forward", "reverse", "stop", "test"],
                        help="Command to execute")
    parser.add_argument("--motor", choices=["linear", "dc", "both"], default="linear",
                        help="Which motor to control (default: linear)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duration in seconds (default: 2.0)")

    args = parser.parse_args()

    # Pin definitions
    LINEAR_PINS = {'enable': 12, 'in1': 18, 'in2': 15, 'stby': 23}
    DC_PINS = {'enable': 25, 'in1': 7, 'in2': 8, 'stby': 1}

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    try:
        motors = []

        if args.motor in ["linear", "both"]:
            linear = TB6612MotorSimple(
                LINEAR_PINS['enable'],
                LINEAR_PINS['in1'],
                LINEAR_PINS['in2'],
                LINEAR_PINS['stby']
            )
            motors.append(("Linear Actuator", linear))

        if args.motor in ["dc", "both"]:
            dc = TB6612MotorSimple(
                DC_PINS['enable'],
                DC_PINS['in1'],
                DC_PINS['in2'],
                DC_PINS['stby']
            )
            motors.append(("DC Motor", dc))

        if args.command == "test":
            for name, motor in motors:
                print(f"\n=== Testing {name} ===")
                print("Forward 2s...")
                motor.forward()
                time.sleep(2)
                motor.stop()
                time.sleep(1)

                print("Reverse 2s...")
                motor.reverse()
                time.sleep(2)
                motor.stop()
                time.sleep(1)

                print("Test complete")

        else:
            for name, motor in motors:
                print(f"{name}:")
                if args.command == "forward":
                    motor.forward()
                elif args.command == "reverse":
                    motor.reverse()
                elif args.command == "stop":
                    motor.stop()

            if args.command != "stop":
                time.sleep(args.duration)
                for name, motor in motors:
                    motor.stop()

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    sys.exit(main())
