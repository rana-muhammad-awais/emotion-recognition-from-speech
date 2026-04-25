@echo off
cd /d "%~dp0"

echo ========================================
echo   Emotion Recognition - GUI
echo ========================================
echo.

py -3.11 -c "import tkinter" 2>nul
if errorlevel 1 (
    echo Tkinter not available!
    pause
    exit /b 1
)

echo Starting desktop GUI...
echo.

py -3.11 gui.py

pause