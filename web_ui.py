#!/usr/bin/env python3
"""
Herbot Web UI - Control and Monitor Herbot
Simple Streamlit-based web interface with green theme
"""

import streamlit as st
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Herbot Control Panel",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for green theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2d7a3e;
        --secondary-color: #4caf50;
        --background-color: #f1f8f4;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2d7a3e 0%, #4caf50 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #2d7a3e;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Status card */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* Success/Warning messages */
    .stSuccess {
        background-color: #d4edda;
        border-left: 5px solid #4caf50;
    }

    .stWarning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8fdf9;
    }

    /* Image gallery */
    .image-card {
        border: 2px solid #4caf50;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌿 Herbot Control Panel</h1>
    <p>Herb Management Robot - Monitor & Control Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'command_output' not in st.session_state:
    st.session_state.command_output = ""
if 'last_scan_results' not in st.session_state:
    st.session_state.last_scan_results = None


def run_herbot_command(cmd_args: list) -> tuple:
    """Run herbot command and return (success, output)"""
    try:
        result = subprocess.run(
            ["python3", "herbot.py"] + cmd_args,
            cwd="/home/pi/Ai-embed/projects/Herbot",
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out (>120s)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_captured_images():
    """Get list of captured images"""
    captures_dir = Path("/home/pi/Ai-embed/projects/Herbot/captures")
    if not captures_dir.exists():
        return []

    images = sorted(captures_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return images


def format_file_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0


# Sidebar - Quick Controls
with st.sidebar:
    st.markdown("### ⚡ Quick Controls")

    # Emergency Stop
    if st.button("🛑 EMERGENCY STOP", use_container_width=True):
        st.error("Emergency stop not implemented - use Ctrl+C on running process")

    st.divider()

    # Quick Actions
    st.markdown("### 🎯 Quick Actions")

    col1, col2 = st.columns(2)

    if st.button("🏠 Home All", use_container_width=True):
        with st.spinner("Homing all axes..."):
            success, output = run_herbot_command(["home"])
            st.session_state.command_output = output
            if success:
                st.success("Homed!")
            else:
                st.error("Failed")

    # Gripper Controls
    st.markdown("### ✂️ Gripper")
    gcol1, gcol2, gcol3 = st.columns(3)

    with gcol1:
        if st.button("Open", use_container_width=True):
            success, output = run_herbot_command(["gripper", "open"])
            st.session_state.command_output = output

    with gcol2:
        if st.button("Close", use_container_width=True):
            success, output = run_herbot_command(["gripper", "close"])
            st.session_state.command_output = output

    with gcol3:
        if st.button("Cut", use_container_width=True):
            success, output = run_herbot_command(["gripper", "cut"])
            st.session_state.command_output = output

    st.divider()

    # Status
    st.markdown("### 📊 System Status")
    if st.button("🔄 Refresh Status", use_container_width=True):
        success, output = run_herbot_command(["status"])
        st.session_state.command_output = output
        if success:
            st.success("Status updated!")


# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Control", "📸 Camera & Scans", "🖼️ Gallery", "📋 Logs"])

# TAB 1: Control
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔧 Manual Control")

        # Z-axis control
        with st.expander("📏 Z-Axis (Height)", expanded=True):
            z_pos = st.slider("Z Position (mm)", 0, 750, 0, 10)
            z_speed = st.slider("Z Speed (Hz)", 100, 2000, 1000, 100)

            if st.button("Move Z", use_container_width=True):
                with st.spinner(f"Moving to Z={z_pos}mm..."):
                    success, output = run_herbot_command([
                        "move", "--z", str(z_pos), "--z-speed", str(z_speed)
                    ])
                    st.session_state.command_output = output
                    if success:
                        st.success(f"Moved to Z={z_pos}mm")

        # R-axis control (Linear Actuator - Forward/Reverse only)
        with st.expander("↔️ R-Axis (Linear Actuator)", expanded=True):
            st.markdown("**Manual Control** (Forward = Extend, Reverse = Contract)")

            r_duration = st.slider("Duration (seconds)", 0.5, 10.0, 3.0, 0.5)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("⏩ Forward (Extend)", use_container_width=True):
                    with st.spinner(f"Extending for {r_duration}s..."):
                        success, output = run_herbot_command([
                            "actuator", "extend", "--duration", str(r_duration)
                        ])
                        st.session_state.command_output = output
                        if success:
                            st.success("Extended!")

            with col2:
                if st.button("⏪ Reverse", use_container_width=True):
                    with st.spinner(f"Reversing for {r_duration}s..."):
                        success, output = run_herbot_command([
                            "actuator", "retract", "--duration", str(r_duration)
                        ])
                        st.session_state.command_output = output
                        if success:
                            st.success("Reversed!")

        # Theta control
        with st.expander("🔄 Θ-Axis (Rotation)", expanded=True):
            theta_duration = st.slider("Rotation Duration (s)", 0.5, 10.0, 2.0, 0.5)
            theta_direction = st.radio("Direction", ["Clockwise", "Counter-Clockwise"])

            if st.button("Rotate", use_container_width=True):
                cmd = ["move", "--theta", str(theta_duration)]
                if theta_direction == "Counter-Clockwise":
                    cmd.append("--theta-ccw")

                with st.spinner(f"Rotating for {theta_duration}s..."):
                    success, output = run_herbot_command(cmd)
                    st.session_state.command_output = output
                    if success:
                        st.success(f"Rotated {theta_direction}")

    with col2:
        st.markdown("### 🎯 Advanced Operations")

        # Approach Leaf
        with st.expander("🍃 Approach Leaf", expanded=True):
            st.markdown("Approach a specific leaf position")

            approach_z = st.number_input("Leaf Height (mm)", 0, 750, 200, 10)
            approach_theta = st.number_input("Rotation (s)", 0.0, 10.0, 2.0, 0.5)
            approach_r = st.number_input("Extension (mm)", 0, 50, 30, 1)
            approach_cut = st.checkbox("Cut after approaching")

            if st.button("Execute Approach", use_container_width=True):
                cmd = [
                    "approach",
                    "--z", str(approach_z),
                    "--theta", str(approach_theta),
                    "--r", str(approach_r)
                ]
                if approach_cut:
                    cmd.append("--cut")

                with st.spinner("Approaching leaf..."):
                    success, output = run_herbot_command(cmd)
                    st.session_state.command_output = output
                    if success:
                        st.success("Approach complete!")

        # Simple Scan
        with st.expander("🔍 Simple Scan", expanded=False):
            st.markdown("Rotate plant slowly for observation")

            scan_duration = st.number_input("Duration (s)", 10, 300, 60, 10)
            scan_speed = st.slider("Speed (%)", 1, 20, 3, 1)

            if st.button("Start Simple Scan", use_container_width=True):
                cmd = [
                    "scan",
                    "--duration", str(scan_duration),
                    "--speed", str(scan_speed)
                ]

                with st.spinner(f"Scanning for {scan_duration}s..."):
                    success, output = run_herbot_command(cmd)
                    st.session_state.command_output = output
                    if success:
                        st.success("Scan complete!")

# TAB 2: Camera & Scans
with tab2:
    st.markdown("### 🌱 Full Plant Management")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Scan Configuration")

        manage_duration = st.number_input(
            "Total Scan Duration (seconds)",
            min_value=10,
            max_value=600,
            value=60,
            step=10,
            help="Total time for continuous scanning"
        )

        manage_z_step = st.selectbox(
            "Z-Axis Step Size (mm)",
            [50, 100, 150, 200],
            index=1,
            help="Distance between scan points (100mm = 8 points)"
        )

        manage_threshold = st.slider(
            "Disease Detection Threshold",
            0.0, 1.0, 0.6, 0.05,
            help="Confidence threshold for disease detection (0.6 = 60% confidence, recommended)"
        )

        st.info(f"""
        **Scan Configuration:**
        - 📏 Z-axis: 0-700mm in {manage_z_step}mm steps = {int(700/manage_z_step)+1} points
        - 🔄 Continuous plant rotation at 3% speed
        - 📸 Camera capture at each position
        - ✂️ Automatic leaf removal on disease detection
        """)

    with col2:
        st.markdown("#### Status")

        captures_dir = Path("/home/pi/Ai-embed/projects/Herbot/captures")
        if captures_dir.exists():
            image_count = len(list(captures_dir.glob("*.jpg")))
            st.metric("Total Captures", image_count)
        else:
            st.metric("Total Captures", 0)

        if st.session_state.last_scan_results:
            results = st.session_state.last_scan_results
            st.metric("Last Scan Images", results.get('total_images', 0))
            st.metric("Diseases Detected", results.get('diseased_detected', 0))
            st.metric("Cuts Performed", results.get('cuts_performed', 0))

    st.divider()

    # Start Management button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 START FULL MANAGEMENT SCAN", use_container_width=True, type="primary"):
            st.markdown("### 🔄 Scan in Progress...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            cmd = [
                "manage",
                "--duration", str(manage_duration),
                "--z-step", str(manage_z_step),
                "--threshold", str(manage_threshold)
            ]

            status_text.text("Starting scan...")

            # Run in background and show progress
            start_time = time.time()
            process = subprocess.Popen(
                ["python3", "herbot.py"] + cmd,
                cwd="/home/pi/Ai-embed/projects/Herbot",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Monitor progress
            output_lines = []
            while process.poll() is None:
                elapsed = time.time() - start_time
                progress = min(elapsed / manage_duration, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Scanning... {elapsed:.1f}s / {manage_duration}s")
                time.sleep(0.5)

            stdout, stderr = process.communicate()
            output = stdout + stderr
            st.session_state.command_output = output

            progress_bar.progress(1.0)

            if process.returncode == 0:
                st.success("✅ Management scan complete!")

                # Try to parse results from output
                if "Total images captured:" in output:
                    st.balloons()
                    st.markdown("### 📊 Scan Results")
                    st.code(output)
            else:
                st.error("❌ Scan failed - check logs")

# TAB 3: Gallery
with tab3:
    st.markdown("### 🖼️ Captured Images Gallery")

    images = get_captured_images()

    if not images:
        st.info("No images captured yet. Run a scan to capture images.")
    else:
        # Filter options
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            show_count = st.selectbox("Show", [10, 25, 50, 100, "All"], index=0)
            if show_count != "All":
                images = images[:show_count]

        with col2:
            sort_by = st.selectbox("Sort by", ["Newest First", "Oldest First"])
            if sort_by == "Oldest First":
                images = list(reversed(images))

        with col3:
            st.metric("Total Images", len(get_captured_images()))

        st.divider()

        # Display images in grid
        cols_per_row = 3
        for i in range(0, len(images), cols_per_row):
            cols = st.columns(cols_per_row)

            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(images):
                    img_path = images[idx]

                    with col:
                        st.image(str(img_path), use_container_width=True)

                        # Image info
                        file_stat = img_path.stat()
                        file_size = format_file_size(file_stat.st_size)
                        file_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                        st.caption(f"**{img_path.name}**")
                        st.caption(f"📅 {file_time}")
                        st.caption(f"💾 {file_size}")

# TAB 4: Logs
with tab4:
    st.markdown("### 📋 Command Output & Logs")

    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown("#### Latest Command Output")

    with col2:
        if st.button("Clear Logs"):
            st.session_state.command_output = ""
            st.success("Logs cleared")

    if st.session_state.command_output:
        st.code(st.session_state.command_output, language="text")
    else:
        st.info("No command output yet. Run a command to see logs here.")

    st.divider()

    # System info
    st.markdown("#### 💻 System Information")

    try:
        # Get system info
        import platform

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Platform", platform.system())

        with col2:
            st.metric("Python Version", platform.python_version())

        with col3:
            # Check if herbot.py exists
            herbot_path = Path("/home/pi/Ai-embed/projects/Herbot/herbot.py")
            if herbot_path.exists():
                st.metric("Herbot Status", "✅ Ready")
            else:
                st.metric("Herbot Status", "❌ Not Found")
    except Exception as e:
        st.error(f"Could not get system info: {e}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌿 Herbot Control Panel v1.0 | Herb Management Robot</p>
    <p style='font-size: 0.9rem;'>Use caution when controlling the robot. Always ensure safe operation.</p>
</div>
""", unsafe_allow_html=True)
