#!/usr/bin/env python3
"""
Servo Gripper Test
Pin 33 (GPIO 13) - Hardware PWM
For gripper with blade attachment
"""

import argparse
import sys
import time

try:
    import RPi.GPIO as GPIO
except Exception as exc:
    print("RPi.GPIO import failed.", file=sys.stderr)
    raise


class ServoGripper:
    """Servo motor gripper controller"""

    def __init__(self, servo_pin: int = 13, pwm_freq: int = 50):
        """
        Initialize servo gripper

        Args:
            servo_pin: GPIO pin for servo signal (default: 13 for Pin 33)
            pwm_freq: PWM frequency in Hz (default: 50Hz for standard servos)
        """
        self.servo_pin = servo_pin
        self.pwm_freq = pwm_freq
        self.pwm = None

        # Servo angle calibration (adjust these values for your servo)
        self.angle_open = 90     # Fully open position (degrees)
        self.angle_close = 0     # Fully closed position (degrees)
        self.current_angle = None

        # Setup GPIO
        GPIO.setup(self.servo_pin, GPIO.OUT)

        # Initialize PWM (50Hz for standard servos)
        self.pwm = GPIO.PWM(self.servo_pin, self.pwm_freq)
        self.pwm.start(0)

        print(f"Servo Gripper initialized on GPIO {self.servo_pin}")
        print(f"PWM Frequency: {self.pwm_freq}Hz")

    def angle_to_duty_cycle(self, angle: float) -> float:
        """
        Convert angle to PWM duty cycle

        Standard servo:
        - 0° = 1ms pulse = 5% duty cycle
        - 90° = 1.5ms pulse = 7.5% duty cycle
        - 180° = 2ms pulse = 10% duty cycle

        Args:
            angle: Servo angle in degrees (0-180)

        Returns:
            Duty cycle percentage (2-12%)
        """
        if not 0 <= angle <= 180:
            raise ValueError("Angle must be between 0 and 180 degrees")

        # Linear mapping: 0° -> 2.5%, 180° -> 12.5%
        # Adjust these values if your servo behaves differently
        duty = 2.5 + (angle / 180.0) * 10.0
        return duty

    def set_angle(self, angle: float, hold_time: float = 0.5) -> None:
        """
        Set servo to specific angle

        Args:
            angle: Target angle in degrees (0-180)
            hold_time: Time to hold position in seconds
        """
        duty = self.angle_to_duty_cycle(angle)
        print(f"Setting angle to {angle}° (duty cycle: {duty:.2f}%)")

        self.pwm.ChangeDutyCycle(duty)
        time.sleep(hold_time)

        # Stop PWM signal to prevent jittering
        self.pwm.ChangeDutyCycle(0)
        self.current_angle = angle

    def open(self, hold_time: float = 0.5) -> None:
        """Open gripper (release)"""
        print("Opening gripper...")
        self.set_angle(self.angle_open, hold_time)
        print("✓ Gripper opened")

    def close(self, hold_time: float = 0.5) -> None:
        """Close gripper (grip/cut)"""
        print("Closing gripper...")
        self.set_angle(self.angle_close, hold_time)
        print("✓ Gripper closed")

    def cut(self, cut_time: float = 1.0) -> None:
        """
        Perform cutting action (open -> close)

        Args:
            cut_time: Time to hold closed position for cutting
        """
        print("Performing cut action...")
        # Always open first to ensure proper cutting motion
        self.open(hold_time=0.3)
        # Then close to cut
        self.close(hold_time=cut_time)
        print("✓ Cut complete")

    def calibrate(self) -> None:
        """Interactive calibration to find open/close angles"""
        print("\n" + "="*60)
        print("SERVO GRIPPER CALIBRATION")
        print("="*60)

        print("\n1. Finding OPEN position")
        print("   We'll test different angles to find the fully open position")

        test_angles = [0, 10, 20, 30, 45, 60, 75, 90]

        for angle in test_angles:
            self.set_angle(angle, hold_time=1.0)
            response = input(f"   Angle {angle}°: Is gripper fully OPEN? (y/n/skip): ").lower()
            if response == 'y':
                self.angle_open = angle
                print(f"   ✓ Open position set to {angle}°")
                break
            elif response == 'skip':
                break

        print("\n2. Finding CLOSE position")
        print("   We'll test angles to find the fully closed position")

        test_angles = [90, 100, 110, 120, 135, 150, 165, 180]

        for angle in test_angles:
            self.set_angle(angle, hold_time=1.0)
            response = input(f"   Angle {angle}°: Is gripper fully CLOSED? (y/n/skip): ").lower()
            if response == 'y':
                self.angle_close = angle
                print(f"   ✓ Close position set to {angle}°")
                break
            elif response == 'skip':
                break

        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print(f"Open angle: {self.angle_open}°")
        print(f"Close angle: {self.angle_close}°")
        print("="*60)
        print("\nUpdate these values in your code:")
        print(f"  self.angle_open = {self.angle_open}")
        print(f"  self.angle_close = {self.angle_close}")

    def test_sequence(self) -> None:
        """Run a test sequence"""
        print("\n=== Gripper Test Sequence ===")

        print("\n1. Opening gripper...")
        self.open(hold_time=1.0)
        time.sleep(1)

        print("\n2. Closing gripper...")
        self.close(hold_time=1.0)
        time.sleep(1)

        print("\n3. Opening gripper...")
        self.open(hold_time=1.0)
        time.sleep(1)

        print("\n4. Testing cut action...")
        self.cut(cut_time=1.5)
        time.sleep(1)

        print("\n5. Opening gripper...")
        self.open(hold_time=1.0)

        print("\n✓ Test sequence complete!")

    def sweep_test(self) -> None:
        """Sweep through angles for testing"""
        print("\n=== Sweep Test (0° to 180°) ===")

        print("Sweeping forward (0° → 180°)...")
        for angle in range(0, 181, 10):
            print(f"  Angle: {angle}°")
            self.set_angle(angle, hold_time=0.3)

        time.sleep(1)

        print("Sweeping backward (180° → 0°)...")
        for angle in range(180, -1, -10):
            print(f"  Angle: {angle}°")
            self.set_angle(angle, hold_time=0.3)

        print("✓ Sweep test complete!")

    def cleanup(self) -> None:
        """Cleanup PWM"""
        if self.pwm:
            self.pwm.stop()
            self.pwm = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Servo Gripper Test (GPIO 13 / Pin 33)",
        epilog="Examples:\n"
               "  python3 servo_gripper_test.py open\n"
               "  python3 servo_gripper_test.py close\n"
               "  python3 servo_gripper_test.py angle --degrees 45\n"
               "  python3 servo_gripper_test.py cut\n"
               "  python3 servo_gripper_test.py test\n"
               "  python3 servo_gripper_test.py calibrate",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Open
    subparsers.add_parser('open', help='Open gripper')

    # Close
    subparsers.add_parser('close', help='Close gripper')

    # Cut
    cut_parser = subparsers.add_parser('cut', help='Perform cut action')
    cut_parser.add_argument('--duration', type=float, default=1.0, help='Cut duration (default: 1.0s)')

    # Angle
    angle_parser = subparsers.add_parser('angle', help='Set specific angle')
    angle_parser.add_argument('--degrees', type=float, required=True, help='Angle in degrees (0-180)')
    angle_parser.add_argument('--hold', type=float, default=1.0, help='Hold time in seconds (default: 1.0)')

    # Test
    subparsers.add_parser('test', help='Run test sequence')

    # Sweep
    subparsers.add_parser('sweep', help='Sweep through all angles')

    # Calibrate
    subparsers.add_parser('calibrate', help='Interactive calibration')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Initialize GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    gripper = None

    try:
        # GPIO 13 (Pin 33) - Hardware PWM
        gripper = ServoGripper(servo_pin=13)

        if args.command == 'open':
            gripper.open()

        elif args.command == 'close':
            gripper.close()

        elif args.command == 'cut':
            gripper.cut(cut_time=args.duration)

        elif args.command == 'angle':
            gripper.set_angle(args.degrees, hold_time=args.hold)

        elif args.command == 'test':
            gripper.test_sequence()

        elif args.command == 'sweep':
            gripper.sweep_test()

        elif args.command == 'calibrate':
            gripper.calibrate()

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
        if gripper:
            gripper.cleanup()
        GPIO.cleanup()


if __name__ == "__main__":
    sys.exit(main())
