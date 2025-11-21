#!/usr/bin/env python3

import sys
import time
import RPi.GPIO as GPIO

# Pin definitions
PWM_PIN = 12
IN1_PIN = 18
IN2_PIN = 15
STBY_PIN = 23

print("=== GPIO Wiring Check ===\n")
print("Expected connections:")
print(f"  GPIO {PWM_PIN} → TB6612 PWMA")
print(f"  GPIO {IN1_PIN} → TB6612 AIN1")
print(f"  GPIO {IN2_PIN} → TB6612 AIN2")
print(f"  GPIO {STBY_PIN} → TB6612 STBY")
print("\nPower connections:")
print("  Motor Driver VM → 6V")
print("  Motor Driver VCC → 5V (Logic)")
print("  Motor Driver GND → GND (both power and logic)")
print("\nPress Ctrl+C to stop\n")

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(PWM_PIN, GPIO.OUT)
GPIO.setup(IN1_PIN, GPIO.OUT)
GPIO.setup(IN2_PIN, GPIO.OUT)
GPIO.setup(STBY_PIN, GPIO.OUT)

try:
    # Test 1: STBY test
    print("Test 1: STBY pin toggle")
    print("  Setting STBY LOW (driver disabled)...")
    GPIO.output(STBY_PIN, GPIO.LOW)
    time.sleep(1)
    print("  Setting STBY HIGH (driver enabled)...")
    GPIO.output(STBY_PIN, GPIO.HIGH)
    time.sleep(1)
    print("  ✓ STBY toggle complete\n")

    # Test 2: Direction pins
    print("Test 2: Direction pins (without PWM)")
    print("  Setting FORWARD (IN1=HIGH, IN2=LOW)...")
    GPIO.output(IN1_PIN, GPIO.HIGH)
    GPIO.output(IN2_PIN, GPIO.LOW)
    GPIO.output(PWM_PIN, GPIO.HIGH)  # Full power without PWM
    print("  Motor should be running FORWARD now for 3 seconds...")
    time.sleep(3)

    print("  Stopping...")
    GPIO.output(PWM_PIN, GPIO.LOW)
    time.sleep(1)

    print("  Setting REVERSE (IN1=LOW, IN2=HIGH)...")
    GPIO.output(IN1_PIN, GPIO.LOW)
    GPIO.output(IN2_PIN, GPIO.HIGH)
    GPIO.output(PWM_PIN, GPIO.HIGH)  # Full power without PWM
    print("  Motor should be running REVERSE now for 3 seconds...")
    time.sleep(3)

    print("  Stopping...")
    GPIO.output(PWM_PIN, GPIO.LOW)
    GPIO.output(IN1_PIN, GPIO.LOW)
    GPIO.output(IN2_PIN, GPIO.LOW)

    print("\n✓ Test complete!")
    print("\nDid the motor move? If not, check:")
    print("  1. 6V power supply connected to VM and GND")
    print("  2. 5V logic power connected to VCC")
    print("  3. Motor connected to A01 and A02 terminals")
    print("  4. Wiring matches the pin numbers above")

except KeyboardInterrupt:
    print("\n\nTest interrupted")

finally:
    GPIO.cleanup()
