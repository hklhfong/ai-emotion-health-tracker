@echo off
echo ============================================================
echo   AI Emotion Health Tracker - Build Standalone .exe
echo ============================================================
echo.

REM Check if PyInstaller is available
python -m PyInstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] PyInstaller not found. Installing now...
    pip install pyinstaller
    echo.
)

REM Run the build script
python "%~dp0build_exe.py"

echo.
pause
