#!/usr/bin/env python3

import sys
import time
import RPi.GPIO as GPIO

# Pin definitions - same as linear actuator
PWM_PIN = 12
IN1_PIN = 18
IN2_PIN = 15
STBY_PIN = 23

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup pins
GPIO.setup(PWM_PIN, GPIO.OUT)
GPIO.setup(IN1_PIN, GPIO.OUT)
GPIO.setup(IN2_PIN, GPIO.OUT)
GPIO.setup(STBY_PIN, GPIO.OUT)

print("Testing GPIO pins...")
print(f"PWM: {PWM_PIN}, IN1: {IN1_PIN}, IN2: {IN2_PIN}, STBY: {STBY_PIN}")

# Enable driver
print("\n1. Setting STBY to HIGH (enable driver)")
GPIO.output(STBY_PIN, GPIO.HIGH)
time.sleep(0.5)

# Setup PWM
print("2. Setting up PWM at 1000Hz")
pwm = GPIO.PWM(PWM_PIN, 1000)
pwm.start(0)

# Test extend direction
print("3. Testing EXTEND direction (IN1=HIGH, IN2=LOW)")
GPIO.output(IN1_PIN, GPIO.HIGH)
GPIO.output(IN2_PIN, GPIO.LOW)
print("   Setting PWM to 50%")
pwm.ChangeDutyCycle(50)

print("   Motor should be running now for 2 seconds...")
time.sleep(2)

print("4. Stopping motor")
GPIO.output(IN1_PIN, GPIO.LOW)
GPIO.output(IN2_PIN, GPIO.LOW)
pwm.ChangeDutyCycle(0)

print("\nTest complete - did the motor move?")

# Cleanup
pwm.stop()
GPIO.cleanup()
