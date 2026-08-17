@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "TRAY_BAT=%SCRIPT_DIR%run_tray.bat"
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\AITrackerTray.vbs"

echo =========================================================
echo    AI Tracker System Tray Auto-Start Setup
echo =========================================================
echo.

(
echo Set WshShell = CreateObject^("WScript.Shell"^)
echo WshShell.CurrentDirectory = "%SCRIPT_DIR:~0,-1%"
echo WshShell.Run "pythonw """ ^& "%SCRIPT_DIR%tray_app.py" ^& """", 0, False
) > "%SHORTCUT_PATH%"

echo [+] Registered System Tray App in Windows Startup!
echo [+] Starting System Tray App now...
start "" pythonw "%SCRIPT_DIR%tray_app.py"
echo [+] Running! Check your Windows taskbar tray (bottom-right near the clock).
echo.
pause
