@echo off
cd /d "%~dp0"

echo ========================================
echo   Emotion Recognition - Web Interface
echo ========================================
echo.

python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install streamlit librosa scikit-learn soundfile numpy scipy pandas matplotlib tqdm
    echo.
)

echo Starting Streamlit web interface...
echo If this doesn't open automatically,
echo go to: http://localhost:8501
echo.

streamlit run app.py

pause