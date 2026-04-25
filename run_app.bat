@echo off
cd /d "%~dp0"

echo ========================================
echo   Emotion Recognition - Web Interface
echo ========================================
echo.

py -3.11 -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not installed!
    echo Please run setup.bat first to install dependencies.
    pause
    exit /b 1
)

echo Starting Streamlit web interface...
echo If this doesn't open automatically,
echo go to: http://localhost:8501
echo.

py -3.11 -m streamlit run app.py