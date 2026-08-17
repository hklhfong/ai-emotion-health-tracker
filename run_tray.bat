@echo off
title AI Health Tracker (Tray Launcher)
cd /d "%~dp0"

echo Starting AI Tracker in Windows System Tray...
start "" pythonw tray_app.py
