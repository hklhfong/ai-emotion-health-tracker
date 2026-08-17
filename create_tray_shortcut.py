import os
import sys

desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
shortcut_path = os.path.join(desktop, "AI Tracker (Tray Mode).bat")
target_bat = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_tray.bat")

with open(shortcut_path, "w", encoding="utf-8") as f:
    f.write(f'@echo off\ncall "{target_bat}"\n')

print(f"[+] Tray mode launcher created at: {shortcut_path}")
