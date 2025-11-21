#!/usr/bin/env python3
"""
Herbot - Herb Management Robot
3-axis cylindrical coordinate system for wilted leaf removal

Axes:
- Z-axis (height): NEMA17 stepper motor with belt-driven carriage (0~750mm)
- R-axis (radius): Linear actuator attached to carriage (0~50mm)
- θ-axis (rotation): DC motor rotating plant base (0~360°)
"""

import argparse
import sys
import time
import json
import os
import math
import threading
import subprocess
import random
from datetime import datetime
from pathlib import Path

try:
    import RPi.GPIO as GPIO
except Exception as exc:
    print("RPi.GPIO import failed.", file=sys.stderr)
    raise

# Import AI inference modules
from herbify_inference import HerbClassifier
from plantdoc_inference import PlantDiseaseClassifier


class StepperMotorZ:
    """NEMA17 Stepper Motor for Z-axis (height) control"""

    def __init__(self, enable_pin: int, step_pin: int, dir_pin: int, config_file: str = "stepper_config.json"):
        self.enable_pin = enable_pin
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.config_file = config_file
        self.state_file = "stepper_state.json"

        # Load calibration
        self.steps_per_mm = 4.27
        self.max_position_mm = 750.0
        self.current_step = 0
        self.is_homed = False

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.steps_per_mm = config.get('steps_per_mm', self.steps_per_mm)
                    self.max_position_mm = config.get('max_position_mm', self.max_position_mm)
            except Exception:
                pass

        # Load saved state (current position)
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_step = state.get('current_step', 0)
                    self.is_homed = state.get('is_homed', False)
                    print(f"Loaded Z-axis state: {self.current_step} steps ({self.get_z_mm():.1f}mm), homed={self.is_homed}")
            except Exception as e:
                print(f"Warning: Could not load Z-axis state: {e}")

        # Setup GPIO
        GPIO.setup(self.enable_pin, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.step_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.dir_pin, GPIO.OUT, initial=GPIO.LOW)

    def enable(self, enabled: bool = True) -> None:
        """Enable/disable motor"""
        GPIO.output(self.enable_pin, GPIO.LOW if enabled else GPIO.HIGH)

    def set_direction(self, up: bool) -> None:
        """Set direction (up or down)"""
        GPIO.output(self.dir_pin, GPIO.HIGH if up else GPIO.LOW)

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

    def save_state(self) -> None:
        """Save current position to file"""
        try:
            state = {
                'current_step': self.current_step,
                'is_homed': self.is_homed
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Warning: Could not save Z-axis state: {e}")

    def move_steps(self, steps: int, frequency_hz: float = 1000.0) -> None:
        """Move by steps (+ = up, - = down)"""
        if steps == 0:
            return

        up = steps > 0
        self.set_direction(up)
        self.enable(True)

        abs_steps = abs(steps)
        self.step_pulses(abs_steps, frequency_hz)
        self.current_step += steps

        # Save state after movement
        self.save_state()

    def move_to_z(self, z_mm: float, frequency_hz: float = 1000.0) -> None:
        """Move to absolute Z position"""
        if not self.is_homed:
            print("WARNING: Z-axis not homed!")

        if not 0 <= z_mm <= self.max_position_mm:
            print(f"ERROR: Z={z_mm}mm out of range [0, {self.max_position_mm}]")
            return

        current_mm = self.current_step / self.steps_per_mm
        distance_mm = z_mm - current_mm
        steps = int(distance_mm * self.steps_per_mm)

        print(f"Z-axis: {current_mm:.1f}mm → {z_mm:.1f}mm")
        self.move_steps(steps, frequency_hz)

    def home(self) -> None:
        """
        Home Z-axis by moving to bottom position (0mm, 0 steps)
        Moves from current position to step 0
        """
        print(f"Z-axis homing (current: {self.current_step} steps, {self.get_z_mm():.1f}mm)...")

        # Calculate steps needed to reach home (step 0)
        steps_to_home = self.current_step

        if steps_to_home > 0:
            # Currently above home position - move down
            print(f"Moving down {steps_to_home} steps to reach bottom...")
            self.set_direction(False)  # Down
            self.enable(True)
            self.step_pulses(steps_to_home, frequency_hz=800.0)
        elif steps_to_home < 0:
            # Currently below home position (shouldn't happen) - move up
            print(f"Moving up {-steps_to_home} steps to reach bottom...")
            self.set_direction(True)  # Up
            self.enable(True)
            self.step_pulses(-steps_to_home, frequency_hz=800.0)
        else:
            print("Already at home position (0mm)")

        # Set current position as home (0mm)
        self.current_step = 0
        self.is_homed = True
        print("Z-axis homed at 0mm")

        # Save state
        self.save_state()

    def get_z_mm(self) -> float:
        """Get current Z position in mm"""
        return self.current_step / self.steps_per_mm

    def cleanup(self) -> None:
        """Cleanup"""
        self.enable(False)


class LinearActuatorR:
    """Linear Actuator for R-axis (radius) control - attached to Z carriage"""

    def __init__(self, enable_pin: int, in1_pin: int, in2_pin: int, stby_pin: int, stroke_mm: float = 50.0):
        self.enable_pin = enable_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.stby_pin = stby_pin
        self.stroke_mm = stroke_mm
        self.state_file = "actuator_state.json"

        self.current_position_mm = 0.0  # Retracted = 0mm
        self.is_homed = False

        # Load saved state
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_position_mm = state.get('current_position_mm', 0.0)
                    self.is_homed = state.get('is_homed', False)
                    print(f"Loaded R-axis state: {self.current_position_mm:.1f}mm, homed={self.is_homed}")
            except Exception as e:
                print(f"Warning: Could not load R-axis state: {e}")

        # Setup GPIO
        GPIO.setup(self.enable_pin, GPIO.OUT)
        GPIO.setup(self.in1_pin, GPIO.OUT)
        GPIO.setup(self.in2_pin, GPIO.OUT)
        GPIO.setup(self.stby_pin, GPIO.OUT)

        # Enable driver
        GPIO.output(self.stby_pin, GPIO.HIGH)

    def extend_raw(self, duration: float) -> None:
        """Extend for duration (raw control)"""
        GPIO.output(self.in1_pin, GPIO.HIGH)
        GPIO.output(self.in2_pin, GPIO.LOW)
        GPIO.output(self.enable_pin, GPIO.HIGH)
        time.sleep(duration)
        self.stop()

    def retract_raw(self, duration: float) -> None:
        """Retract for duration (raw control)"""
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        GPIO.output(self.enable_pin, GPIO.HIGH)
        time.sleep(duration)
        self.stop()

    def save_state(self) -> None:
        """Save current position to file"""
        try:
            state = {
                'current_position_mm': self.current_position_mm,
                'is_homed': self.is_homed
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Warning: Could not save R-axis state: {e}")

    def move_to_r(self, r_mm: float, speed_mm_per_sec: float = 10.0) -> None:
        """Move to absolute R position (requires calibration/homing)"""
        if not 0 <= r_mm <= self.stroke_mm:
            print(f"ERROR: R={r_mm}mm out of range [0, {self.stroke_mm}]")
            return

        distance = r_mm - self.current_position_mm
        duration = abs(distance) / speed_mm_per_sec

        print(f"R-axis: {self.current_position_mm:.1f}mm → {r_mm:.1f}mm")

        if distance > 0:
            self.extend_raw(duration)
        else:
            self.retract_raw(duration)

        self.current_position_mm = r_mm

        # Save state after movement
        self.save_state()

    def home(self, max_time: float = 6.0) -> None:
        """Home to fully retracted position"""
        print("R-axis homing (retracting)...")
        self.retract_raw(max_time)
        self.current_position_mm = 0.0
        self.is_homed = True
        print("R-axis homed at 0mm")

        # Save state
        self.save_state()

    def stop(self) -> None:
        """Stop motor"""
        GPIO.output(self.enable_pin, GPIO.LOW)
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.LOW)

    def cleanup(self) -> None:
        """Cleanup"""
        self.stop()
        GPIO.output(self.stby_pin, GPIO.LOW)


class ServoGripper:
    """Servo Gripper for cutting wilted leaves"""

    def __init__(self, servo_pin: int = 13, pwm_freq: int = 50):
        self.servo_pin = servo_pin
        self.pwm_freq = pwm_freq
        self.pwm = None

        # Servo angle calibration
        self.angle_open = 90     # Fully open position
        self.angle_close = 0     # Fully closed position
        self.current_angle = None

        # Setup GPIO
        GPIO.setup(self.servo_pin, GPIO.OUT)

        # Initialize PWM (50Hz for servos)
        self.pwm = GPIO.PWM(self.servo_pin, self.pwm_freq)
        self.pwm.start(0)

    def angle_to_duty_cycle(self, angle: float) -> float:
        """Convert angle to PWM duty cycle (0-180° -> 2.5-12.5%)"""
        if not 0 <= angle <= 180:
            raise ValueError("Angle must be between 0 and 180 degrees")
        return 2.5 + (angle / 180.0) * 10.0

    def set_angle(self, angle: float, hold_time: float = 0.5) -> None:
        """Set servo to specific angle"""
        duty = self.angle_to_duty_cycle(angle)
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(hold_time)
        self.pwm.ChangeDutyCycle(0)  # Stop PWM to prevent jittering
        self.current_angle = angle

    def open(self, hold_time: float = 0.5) -> None:
        """Open gripper (release)"""
        self.set_angle(self.angle_open, hold_time)

    def close(self, hold_time: float = 0.5) -> None:
        """Close gripper (grip)"""
        self.set_angle(self.angle_close, hold_time)

    def cut(self, cut_time: float = 1.0) -> None:
        """Perform cutting action (open -> close)"""
        # Always open first to ensure proper cutting motion
        self.open(hold_time=0.3)
        # Then close to cut
        self.close(hold_time=cut_time)

    def cleanup(self) -> None:
        """Cleanup PWM"""
        if self.pwm:
            self.pwm.stop()
            self.pwm = None


class DCMotorTheta:
    """DC Motor for θ-axis (rotation) control - rotates plant base"""

    def __init__(self, enable_pin: int, in1_pin: int, in2_pin: int, stby_pin: int, pwm_freq: int = 1000):
        self.enable_pin = enable_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.stby_pin = stby_pin
        self.pwm_freq = pwm_freq

        self.current_angle_deg = 0.0
        self.is_homed = False
        self.pwm = None

        # Setup GPIO
        GPIO.setup(self.enable_pin, GPIO.OUT)
        GPIO.setup(self.in1_pin, GPIO.OUT)
        GPIO.setup(self.in2_pin, GPIO.OUT)
        GPIO.setup(self.stby_pin, GPIO.OUT)

        # Initialize PWM for speed control
        self.pwm = GPIO.PWM(self.enable_pin, self.pwm_freq)
        self.pwm.start(0)

        # Enable driver
        GPIO.output(self.stby_pin, GPIO.HIGH)

    def rotate_cw(self, duration: float, speed: int = 100) -> None:
        """
        Rotate clockwise for duration

        Args:
            duration: Time in seconds
            speed: Speed 0-100% (default: 100)
        """
        print(f"Rotating CW for {duration}s at {speed}% speed...")
        GPIO.output(self.in1_pin, GPIO.HIGH)
        GPIO.output(self.in2_pin, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)
        time.sleep(duration)
        self.stop()

    def rotate_ccw(self, duration: float, speed: int = 100) -> None:
        """
        Rotate counter-clockwise for duration

        Args:
            duration: Time in seconds
            speed: Speed 0-100% (default: 100)
        """
        print(f"Rotating CCW for {duration}s at {speed}% speed...")
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        self.pwm.ChangeDutyCycle(speed)
        time.sleep(duration)
        self.stop()

    def scan_rotate(self, duration: float = 60.0, clockwise: bool = True, speed: int = 3) -> None:
        """
        Slowly rotate for scanning (1 full rotation in ~60s)

        Args:
            duration: Time in seconds (default: 60s for full rotation)
            clockwise: Rotation direction
            speed: Speed 0-100% (default: 3 for very slow scanning)
        """
        print(f"Scan mode: rotating {'CW' if clockwise else 'CCW'} for {duration}s at {speed}% speed")
        if clockwise:
            self.rotate_cw(duration, speed)
        else:
            self.rotate_ccw(duration, speed)

    def home(self) -> None:
        """Set current angle as 0°"""
        self.current_angle_deg = 0.0
        self.is_homed = True
        print("θ-axis homed at 0°")

    def start_rotation_cw(self, speed: int = 3) -> None:
        """Start continuous clockwise rotation (non-blocking)"""
        GPIO.output(self.in1_pin, GPIO.HIGH)
        GPIO.output(self.in2_pin, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)

    def start_rotation_ccw(self, speed: int = 3) -> None:
        """Start continuous counter-clockwise rotation (non-blocking)"""
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        self.pwm.ChangeDutyCycle(speed)

    def stop(self) -> None:
        """Stop motor"""
        self.pwm.ChangeDutyCycle(0)
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.LOW)

    def cleanup(self) -> None:
        """Cleanup"""
        self.stop()
        if self.pwm:
            self.pwm.stop()
            self.pwm = None
        GPIO.output(self.stby_pin, GPIO.LOW)


class Herbot:
    """
    Herbot - Herb Management Robot
    3-axis cylindrical coordinate system control
    """

    def __init__(self):
        print("\n" + "="*60)
        print("HERBOT - Herb Management Robot")
        print("="*60)

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Initialize motors
        print("\nInitializing motors...")

        # Z-axis: NEMA17 stepper (carriage)
        self.z_motor = StepperMotorZ(
            enable_pin=2,
            step_pin=3,
            dir_pin=4
        )
        print(f"✓ Z-axis (NEMA17): 0~{self.z_motor.max_position_mm:.0f}mm")

        # R-axis: Linear actuator (on carriage)
        self.r_motor = LinearActuatorR(
            enable_pin=12,
            in1_pin=18,
            in2_pin=15,
            stby_pin=23,
            stroke_mm=50.0
        )
        print(f"✓ R-axis (Linear Actuator): 0~{self.r_motor.stroke_mm:.0f}mm")

        # θ-axis: DC motor (plant rotation)
        self.theta_motor = DCMotorTheta(
            enable_pin=25,
            in1_pin=7,
            in2_pin=8,
            stby_pin=1
        )
        print("✓ θ-axis (DC Motor): Plant rotation")

        # Gripper: Servo motor
        self.gripper = ServoGripper(servo_pin=13)
        print("✓ Gripper (Servo): Leaf cutting")

        # Initialize AI models
        print("\nInitializing AI models...")

        try:
            # Herbify: Herb species classification (91 classes)
            self.herbify = HerbClassifier(
                model_path="herbify/herbify_edgetpu_ready_edgetpu.tflite",
                class_names_path="herbify/class_names.json",
                use_edgetpu=True
            )
            print("✓ Herbify: Herb species classification (91 classes)")
        except Exception as e:
            print(f"⚠ Herbify model failed to load: {e}")
            self.herbify = None

        try:
            # PlantDoc: Disease detection (28 classes)
            self.plantdoc = PlantDiseaseClassifier(
                model_path="plantdoc/plantdoc_edgetpu_ready_edgetpu.tflite",
                class_names_path="plantdoc/class_names.json",
                use_edgetpu=True
            )
            print("✓ PlantDoc: Disease detection (28 classes)")
        except Exception as e:
            print(f"⚠ PlantDoc model failed to load: {e}")
            self.plantdoc = None

        print("\nHerbot ready!")

    def home_all(self) -> None:
        """Home all axes - moves to physical home positions"""
        print("\n=== Homing All Axes ===")

        # Home R-axis first (retract actuator fully)
        self.r_motor.home()
        time.sleep(0.5)

        # Home Z-axis (move from current position to bottom = 0 steps)
        self.z_motor.home()
        time.sleep(0.5)

        # Home θ-axis (set current angle as 0°)
        self.theta_motor.home()

        print("✓ All axes homed!\n")

    def move_to_position(self, z_mm: float = None, r_mm: float = None,
                         theta_duration: float = None, theta_cw: bool = True,
                         z_speed: float = 1000.0) -> None:
        """
        Move to cylindrical coordinates position

        Args:
            z_mm: Height in mm (0~750)
            r_mm: Radius in mm (0~50)
            theta_duration: Rotation duration in seconds
            theta_cw: Rotate clockwise if True
            z_speed: Z-axis speed in Hz
        """
        print(f"\n=== Moving to Position ===")

        # Move Z-axis if specified
        if z_mm is not None:
            self.z_motor.move_to_z(z_mm, z_speed)
            time.sleep(0.3)

        # Rotate θ-axis if specified
        if theta_duration is not None:
            if theta_cw:
                self.theta_motor.rotate_cw(theta_duration)
            else:
                self.theta_motor.rotate_ccw(theta_duration)
            time.sleep(0.3)

        # Move R-axis if specified
        if r_mm is not None:
            self.r_motor.move_to_r(r_mm)
            time.sleep(0.3)

        print("✓ Position reached\n")

    def scan_mode(self, duration: float = 60.0, clockwise: bool = True, speed: int = 3) -> None:
        """
        Scan mode: slowly rotate plant for camera scanning

        Args:
            duration: Total rotation duration (default: 60s for full rotation)
            clockwise: Rotation direction
            speed: Rotation speed 0-100% (default: 3 for very slow scanning)
        """
        print(f"\n=== Scan Mode ===")
        print(f"Rotating plant for {duration}s at {speed}% speed")
        self.theta_motor.scan_rotate(duration, clockwise, speed)
        print("✓ Scan complete\n")

    def approach_leaf(self, z_mm: float, theta_duration: float, r_mm: float,
                      z_speed: float = 800.0, cut: bool = False) -> None:
        """
        Approach a wilted leaf at given position

        Args:
            z_mm: Leaf height
            theta_duration: Rotation to align with leaf
            r_mm: Extension distance to reach leaf
            z_speed: Z movement speed
            cut: Perform cut action after reaching
        """
        print(f"\n=== Approaching Wilted Leaf ===")
        print(f"Position: Z={z_mm}mm, R={r_mm}mm, θ={theta_duration}s rotation")

        # 0. Open gripper before approaching
        print("0. Opening gripper...")
        self.gripper.open()
        time.sleep(0.3)

        # 1. Move to height
        print("1. Moving to leaf height...")
        self.z_motor.move_to_z(z_mm, z_speed)
        time.sleep(0.5)

        # 2. Rotate to align
        print("2. Rotating to align with leaf...")
        self.theta_motor.rotate_cw(theta_duration)
        time.sleep(0.5)

        # 3. Extend to reach
        print("3. Extending to reach leaf...")
        self.r_motor.move_to_r(r_mm)
        time.sleep(0.5)

        print("✓ Leaf reached!")

        # 4. Cut if requested
        if cut:
            print("4. Cutting leaf...")
            self.gripper.cut(cut_time=1.5)
            time.sleep(0.5)
            print("✓ Leaf cut!")

        print()

    def _capture_image(self, scan_id: int, z_mm: int) -> str:
        """Capture image and return filepath"""
        captures_dir = Path("captures")
        captures_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{scan_id:03d}_z{z_mm:03d}_{timestamp}.jpg"
        filepath = captures_dir / filename

        # Capture using libcamera-still
        cmd = [
            "libcamera-still",
            "-o", str(filepath),
            "--width", "640",
            "--height", "480",
            "--nopreview",
            "-t", "100"  # 100ms timeout
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=5)
            return str(filepath)
        except Exception as e:
            print(f"  Warning: Camera capture failed: {e}")
            return None

    def identify_herb(self, image_path: str, top_k: int = 3) -> tuple:
        """
        Identify herb species using Herbify model

        Args:
            image_path: Path to captured image
            top_k: Number of top predictions to return

        Returns:
            (results, inference_time_ms) where results is list of (class_name, confidence) tuples
        """
        if image_path is None or self.herbify is None:
            return [], 0.0

        try:
            results, inference_time = self.herbify.predict(image_path, top_k=top_k)
            return results, inference_time
        except Exception as e:
            print(f"Herb identification error: {e}")
            return [], 0.0

    def detect_disease(self, image_path: str, threshold: float = 0.6, min_confidence: float = 0.4) -> tuple:
        """
        Detect disease using PlantDoc model

        Args:
            image_path: Path to captured image
            threshold: Confidence threshold for disease detection (default: 0.6)
            min_confidence: Minimum confidence to consider valid detection (default: 0.4)

        Returns:
            (is_diseased, confidence, class_name, inference_time_ms)
        """
        if image_path is None or self.plantdoc is None:
            return False, 0.0, "Unknown", 0.0

        try:
            # Run inference
            results, inference_time = self.plantdoc.predict(image_path, top_k=3)

            # Get top prediction
            top_class, top_confidence = results[0]

            # Filter out low confidence predictions (likely empty space/background)
            if top_confidence < min_confidence:
                return False, top_confidence, f"No clear detection ({top_class})", inference_time

            # Disease detection logic:
            # Classes with "leaf" only (e.g., "Apple leaf", "Tomato leaf") = healthy
            # Classes with disease names (e.g., "blight", "spot", "rust", "scab") = diseased
            disease_keywords = ["blight", "spot", "rust", "scab", "mildew", "virus", "mold", "bacterial", "spider"]

            is_healthy = top_class.lower().endswith("leaf") and \
                        not any(disease in top_class.lower() for disease in disease_keywords)

            is_diseased = not is_healthy and top_confidence >= threshold

            return is_diseased, top_confidence, top_class, inference_time

        except Exception as e:
            print(f"Disease detection error: {e}")
            return False, 0.0, "Error", 0.0

    def scan_and_manage(self,
                       scan_duration: float = 60.0,
                       z_min: int = 0,
                       z_max: int = 700,
                       z_step: int = 100,
                       z_speed: float = 1000.0,
                       r_extend: int = 30,
                       theta_speed: int = 3,
                       disease_threshold: float = 0.6) -> dict:
        """
        Scan entire plant and manage diseased leaves

        Args:
            scan_duration: Total scan duration in seconds (default: 60s)
            z_min: Minimum Z position in mm (default: 0)
            z_max: Maximum Z position in mm (default: 700)
            z_step: Z step size in mm (default: 100, gives 8 points)
            z_speed: Z-axis movement speed in Hz (default: 1000)
            r_extend: R-axis extension distance when disease detected (default: 30mm)
            theta_speed: Theta rotation speed 0-100% (default: 3)
            disease_threshold: Confidence threshold for disease detection (default: 0.6)

        Returns:
            Dictionary with scan results
        """
        print("\n" + "="*60)
        print("PLANT SCAN AND MANAGEMENT")
        print("="*60)
        print(f"Duration: {scan_duration}s")
        print(f"Z range: {z_min}~{z_max}mm (step: {z_step}mm)")
        print(f"Theta speed: {theta_speed}%")
        print(f"R extension: {r_extend}mm")
        print("="*60 + "\n")

        # Results tracking
        results = {
            'total_images': 0,
            'diseased_detected': 0,
            'cuts_performed': 0,
            'scan_positions': []
        }

        # Ensure homed
        if not self.z_motor.is_homed:
            print("WARNING: Z-axis not homed! Results may be inaccurate.")
        if not self.r_motor.is_homed:
            print("WARNING: R-axis not homed!")

        # Ensure gripper is open and R is retracted
        print("Preparing for scan...")
        self.gripper.open()
        self.r_motor.move_to_r(0.0)
        time.sleep(0.5)

        # Start DC motor rotation (continuous)
        print(f"Starting plant rotation at {theta_speed}% speed...")
        self.theta_motor.start_rotation_cw(speed=theta_speed)

        start_time = time.time()
        scan_id = 0
        direction_up = True

        try:
            print("\nBeginning Z-axis scan...\n")

            while time.time() - start_time < scan_duration:
                # Generate Z positions based on direction
                if direction_up:
                    z_positions = range(z_min, z_max + 1, z_step)
                    print(f"→ Scanning UP: {z_min}mm → {z_max}mm")
                else:
                    z_positions = range(z_max, z_min - 1, -z_step)
                    print(f"← Scanning DOWN: {z_max}mm → {z_min}mm")

                # Scan each Z position
                for z_mm in z_positions:
                    # Check if time expired
                    if time.time() - start_time >= scan_duration:
                        print("\n⏱ Scan duration reached")
                        break

                    # Move to Z position
                    self.z_motor.move_to_z(z_mm, z_speed)
                    time.sleep(0.1)  # Stabilization

                    # Capture image
                    scan_id += 1
                    elapsed = time.time() - start_time
                    print(f"\n  [{elapsed:5.1f}s] Scan #{scan_id:03d} @ Z={z_mm:03d}mm")

                    image_path = self._capture_image(scan_id, z_mm)
                    results['total_images'] += 1

                    # Identify herb species using Herbify model
                    herb_results, herb_time = self.identify_herb(image_path, top_k=1)
                    if herb_results:
                        herb_name, herb_confidence = herb_results[0]
                        print(f"    🌿 Herb: {herb_name} ({herb_confidence*100:.1f}%)")
                    else:
                        print(f"    🌿 Herb: Unknown")

                    # Detect disease using PlantDoc model
                    is_diseased, confidence, disease_class, inference_time = self.detect_disease(
                        image_path, threshold=disease_threshold
                    )

                    if is_diseased:
                        print(f"    🍃 Status: 🔴 DISEASED - {disease_class} ({confidence*100:.1f}%, {inference_time:.1f}ms)")
                        results['diseased_detected'] += 1

                        # Pause rotation
                        self.theta_motor.stop()
                        time.sleep(0.2)

                        # Perform removal
                        print(f"    ✂️ Removing diseased leaf...")
                        self.gripper.open()
                        time.sleep(0.2)

                        # Extend to reach
                        self.r_motor.move_to_r(r_extend)
                        time.sleep(0.3)

                        # Cut
                        self.gripper.cut(cut_time=1.5)
                        results['cuts_performed'] += 1
                        time.sleep(0.3)

                        # Retract
                        self.gripper.open()
                        time.sleep(0.2)
                        self.r_motor.move_to_r(0.0)
                        time.sleep(0.3)

                        print(f"    ✓ Removal complete")

                        # Resume rotation
                        self.theta_motor.start_rotation_cw(speed=theta_speed)
                    else:
                        print(f"    🍃 Status: ✅ HEALTHY - {disease_class} ({confidence*100:.1f}%, {inference_time:.1f}ms)")

                    # Save scan position
                    results['scan_positions'].append({
                        'scan_id': scan_id,
                        'z_mm': z_mm,
                        'time': elapsed,
                        'image': image_path,
                        'herb_name': herb_name if herb_results else 'Unknown',
                        'herb_confidence': herb_confidence if herb_results else 0.0,
                        'disease_class': disease_class,
                        'disease_confidence': confidence,
                        'diseased': is_diseased
                    })

                # Flip direction for next cycle
                direction_up = not direction_up
                print()

        finally:
            # Stop rotation
            print("\nStopping rotation...")
            self.theta_motor.stop()

            # Return to home position
            print("Returning to home position...")
            self.r_motor.move_to_r(0.0)
            self.z_motor.move_to_z(0, z_speed)
            self.gripper.open()

        # Print summary
        print("\n" + "="*60)
        print("SCAN COMPLETE")
        print("="*60)
        print(f"Total images captured: {results['total_images']}")
        print(f"Diseased leaves detected: {results['diseased_detected']}")
        print(f"Cuts performed: {results['cuts_performed']}")
        print(f"Scan duration: {time.time() - start_time:.1f}s")
        print("="*60 + "\n")

        return results

    def get_status(self) -> dict:
        """Get current status of all axes"""
        return {
            'z_mm': self.z_motor.get_z_mm(),
            'z_homed': self.z_motor.is_homed,
            'r_mm': self.r_motor.current_position_mm,
            'r_homed': self.r_motor.is_homed,
            'theta_deg': self.theta_motor.current_angle_deg,
            'theta_homed': self.theta_motor.is_homed
        }

    def cleanup(self) -> None:
        """Cleanup all motors"""
        print("\nCleaning up...")
        self.z_motor.cleanup()
        self.r_motor.cleanup()
        self.theta_motor.cleanup()
        self.gripper.cleanup()
        GPIO.cleanup()
        print("✓ Cleanup complete")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Herbot - Herb Management Robot Control",
        epilog="Examples:\n"
               "  python3 herbot.py home\n"
               "  python3 herbot.py move --z 200\n"
               "  python3 herbot.py actuator extend --duration 3.0\n"
               "  python3 herbot.py actuator retract --duration 5.0\n"
               "  python3 herbot.py scan --duration 60\n"
               "  python3 herbot.py approach --z 200 --theta 5 --r 40",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Home
    subparsers.add_parser('home', help='Home all axes')

    # Move
    move_parser = subparsers.add_parser('move', help='Move to position')
    move_parser.add_argument('--z', type=float, help='Z position in mm (0~750)')
    move_parser.add_argument('--r', type=float, help='R position in mm (0~50)')
    move_parser.add_argument('--theta', type=float, help='θ rotation duration in seconds')
    move_parser.add_argument('--theta-ccw', action='store_true', help='Rotate counter-clockwise')
    move_parser.add_argument('--z-speed', type=float, default=1000.0, help='Z-axis speed (Hz)')

    # Scan
    scan_parser = subparsers.add_parser('scan', help='Scan mode (rotate plant slowly)')
    scan_parser.add_argument('--duration', type=float, default=60.0, help='Scan duration (s)')
    scan_parser.add_argument('--ccw', action='store_true', help='Rotate counter-clockwise')
    scan_parser.add_argument('--speed', type=int, default=3, help='Rotation speed 0-100%% (default: 3 for very slow scan)')

    # Approach
    approach_parser = subparsers.add_parser('approach', help='Approach wilted leaf')
    approach_parser.add_argument('--z', type=float, required=True, help='Leaf height (mm)')
    approach_parser.add_argument('--theta', type=float, required=True, help='Rotation duration (s)')
    approach_parser.add_argument('--r', type=float, required=True, help='Extension distance (mm)')
    approach_parser.add_argument('--cut', action='store_true', help='Cut the leaf after approaching')

    # Gripper control
    gripper_parser = subparsers.add_parser('gripper', help='Control gripper')
    gripper_parser.add_argument('action', choices=['open', 'close', 'cut'], help='Gripper action')

    # Actuator control (raw forward/reverse)
    actuator_parser = subparsers.add_parser('actuator', help='Control linear actuator (raw)')
    actuator_parser.add_argument('action', choices=['extend', 'retract'], help='Actuator action')
    actuator_parser.add_argument('--duration', type=float, default=3.0, help='Duration in seconds (default: 3.0)')

    # Manage - Full scan and disease management
    manage_parser = subparsers.add_parser('manage', help='Full plant scan and management')
    manage_parser.add_argument('--duration', type=float, default=60.0, help='Scan duration in seconds (default: 60)')
    manage_parser.add_argument('--z-step', type=int, default=100, help='Z-axis step size in mm (default: 100)')
    manage_parser.add_argument('--threshold', type=float, default=0.6, help='Disease detection confidence threshold (default: 0.6)')

    # Status
    subparsers.add_parser('status', help='Show current status')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    herbot = None

    try:
        herbot = Herbot()

        if args.command == 'home':
            herbot.home_all()

        elif args.command == 'move':
            herbot.move_to_position(
                z_mm=args.z,
                r_mm=args.r,
                theta_duration=args.theta,
                theta_cw=not args.theta_ccw,
                z_speed=args.z_speed
            )

        elif args.command == 'scan':
            herbot.scan_mode(duration=args.duration, clockwise=not args.ccw, speed=args.speed)

        elif args.command == 'approach':
            herbot.approach_leaf(
                z_mm=args.z,
                theta_duration=args.theta,
                r_mm=args.r,
                cut=args.cut
            )

        elif args.command == 'gripper':
            if args.action == 'open':
                herbot.gripper.open()
                print("✓ Gripper opened")
            elif args.action == 'close':
                herbot.gripper.close()
                print("✓ Gripper closed")
            elif args.action == 'cut':
                herbot.gripper.cut()
                print("✓ Cut complete")

        elif args.command == 'actuator':
            if args.action == 'extend':
                print(f"Extending actuator for {args.duration}s...")
                herbot.r_motor.extend_raw(args.duration)
                print("✓ Actuator extended")
            elif args.action == 'retract':
                print(f"Retracting actuator for {args.duration}s...")
                herbot.r_motor.retract_raw(args.duration)
                print("✓ Actuator retracted")

        elif args.command == 'manage':
            results = herbot.scan_and_manage(
                scan_duration=args.duration,
                z_step=args.z_step,
                disease_threshold=args.threshold
            )

        elif args.command == 'status':
            status = herbot.get_status()
            print("\n=== Herbot Status ===")
            print(f"Z-axis: {status['z_mm']:.1f}mm (homed: {status['z_homed']})")
            print(f"R-axis: {status['r_mm']:.1f}mm (homed: {status['r_homed']})")
            print(f"θ-axis: {status['theta_deg']:.1f}° (homed: {status['theta_homed']})")

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
        if herbot:
            herbot.cleanup()


if __name__ == "__main__":
    sys.exit(main())
