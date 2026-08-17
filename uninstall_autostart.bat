@echo off
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\AITrackerService.vbs"

echo Stopping any running tracker instances...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq AITracker*" >nul 2>&1

if exist "%SHORTCUT_PATH%" (
    del "%SHORTCUT_PATH%"
    echo [+] Removed from Windows Startup.
) else (
    echo [i] No startup shortcut found.
)

pause
