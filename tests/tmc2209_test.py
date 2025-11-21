#!/usr/bin/env python3

import argparse
import sys
import time

try:
    import RPi.GPIO as GPIO
except Exception as exc:  # pragma: no cover
    print("RPi.GPIO import failed. Install with: sudo apt install python3-rpi.gpio", file=sys.stderr)
    raise


def configure_pins(enable_pin: int, step_pin: int, dir_pin: int) -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(enable_pin, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(step_pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(dir_pin, GPIO.OUT, initial=GPIO.LOW)


def set_enabled(enable_pin: int, enabled: bool, active_low: bool = True) -> None:
    if active_low:
        level = GPIO.LOW if enabled else GPIO.HIGH
    else:
        level = GPIO.HIGH if enabled else GPIO.LOW
    GPIO.output(enable_pin, level)


def set_direction(dir_pin: int, clockwise: bool) -> None:
    GPIO.output(dir_pin, GPIO.HIGH if clockwise else GPIO.LOW)


def step_pulses(step_pin: int, steps: int, frequency_hz: float) -> None:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0")
    half_period = 1.0 / (2.0 * frequency_hz)
    for _ in range(steps):
        GPIO.output(step_pin, GPIO.HIGH)
        time.sleep(half_period)
        GPIO.output(step_pin, GPIO.LOW)
        time.sleep(half_period)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple STEP/DIR test for TMC2209 with Raspberry Pi GPIO")
    parser.add_argument("--enable", type=int, default=2, help="BCM GPIO for ENABLE (default: 2)")
    parser.add_argument("--step", type=int, default=3, help="BCM GPIO for STEP (default: 3)")
    parser.add_argument("--dir", type=int, default=4, help="BCM GPIO for DIR (default: 4)")
    parser.add_argument("--steps", type=int, default=800, help="Number of step pulses to send (default: 800)")
    parser.add_argument("--freq", type=float, default=1000.0, help="Step frequency in Hz (default: 1000)")
    parser.add_argument("--cw", action="store_true", help="Rotate clockwise (DIR=1). Default is counter-clockwise (DIR=0)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable-active-low", dest="enable_active_low", action="store_true", help="ENABLE pin is active-low (default)")
    group.add_argument("--enable-active-high", dest="enable_active_low", action="store_false", help="ENABLE pin is active-high")
    parser.set_defaults(enable_active_low=True)
    parser.add_argument("--keep-enabled", action="store_true", help="Keep driver enabled at end (default disables)")

    args = parser.parse_args()

    try:
        configure_pins(args.enable, args.step, args.dir)

        # TMC2209 ENABLE is active-low by default
        set_enabled(args.enable, True, active_low=args.enable_active_low)
        set_direction(args.dir, clockwise=args.cw)

        step_pulses(args.step, args.steps, args.freq)

        if not args.keep_enabled:
            set_enabled(args.enable, False, active_low=args.enable_active_low)
        return 0
    except KeyboardInterrupt:
        set_enabled(args.enable, False, active_low=True)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    sys.exit(main())


