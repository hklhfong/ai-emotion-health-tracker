"""
PyInstaller Build Script for AI Emotion Health Tracker.

Produces two standalone executables in the dist/ folder:
  1. AI Tracker.exe       — Interactive visual HUD window (with console)
  2. AI Tracker Tray.exe  — Silent system tray background mode (no console)

Usage:
    pip install pyinstaller
    python build_exe.py
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# All sibling modules that must be bundled as hidden imports
HIDDEN_IMPORTS = [
    "face_engine", "health_engine", "emotion_engine",
    "database", "report_organizer", "input_monitor",
    "notifier", "ai_coach", "run_coach", "export_history",
    "service", "tray_app", "cli",
    # Third-party packages that PyInstaller may miss
    "mediapipe",
    "onnxruntime",
    "hsemotion_onnx",
    "cv2",
    "scipy.signal",
    "scipy.fft",
    "PIL",
    "pystray",
    "pandas",
    "numpy",
    "requests",
    "ctypes",
]

# Data files that must be collected (model weights, configs, etc.)
COLLECT_DATA = [
    "mediapipe",
    "hsemotion_onnx",
]


def build_exe(script, name, windowed=False):
    """Build a single PyInstaller executable."""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--clean",
        "--name", name,
        "--distpath", os.path.join(SCRIPT_DIR, "dist"),
        "--workpath", os.path.join(SCRIPT_DIR, "build"),
        "--specpath", os.path.join(SCRIPT_DIR, "build"),
    ]

    if windowed:
        cmd.append("--noconsole")

    for mod in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", mod])

    for pkg in COLLECT_DATA:
        cmd.extend(["--collect-data", pkg])

    cmd.append(os.path.join(SCRIPT_DIR, script))

    print(f"\n{'='*60}")
    print(f"  Building: {name}")
    print(f"  Source:   {script}")
    print(f"  Mode:     {'Windowed (no console)' if windowed else 'Console'}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n[ERROR] Build failed for {name}. See output above.")
        return False
    print(f"\n[OK] Successfully built: dist/{name}.exe")
    return True


def main():
    print("=" * 60)
    print("  AI Emotion Health Tracker — PyInstaller Build")
    print("=" * 60)

    # Check PyInstaller is installed
    try:
        import PyInstaller
        print(f"[+] PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("[!] PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    success = True

    # Build 1: Visual HUD (with console for debug output)
    if not build_exe("main.py", "AI Tracker", windowed=False):
        success = False

    # Build 2: System Tray (no console window)
    if not build_exe("tray_app.py", "AI Tracker Tray", windowed=True):
        success = False

    print("\n" + "=" * 60)
    if success:
        dist_dir = os.path.join(SCRIPT_DIR, "dist")
        print("  BUILD COMPLETE!")
        print(f"  Output folder: {dist_dir}")
        print()
        print("  Executables created:")
        print(f"    1. AI Tracker.exe       (Visual HUD mode)")
        print(f"    2. AI Tracker Tray.exe  (System tray background mode)")
        print()
        print("  You can distribute these .exe files to anyone with Windows 10/11.")
        print("  No Python installation required on the target machine.")
    else:
        print("  BUILD FAILED — see errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
