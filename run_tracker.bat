@echo off
title AI Real-Time Emotion, Health and Face Tracker
cd /d "%~dp0"

echo =========================================================
echo    AI Emotion, Health and Face Tracker
echo =========================================================
echo.

python service.py --gui

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [-] An error occurred while running the tracker.
    pause
)
