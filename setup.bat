@echo off
cd /d "%~dp0"

echo ========================================
echo   Emotion Recognition - Setup
echo ========================================
echo.
echo Installing required packages...
echo This may take 5-10 minutes on first run...
echo.

py -3.11 -m pip install numpy scipy pandas scikit-learn matplotlib tqdm soundfile

echo.
echo Installing streamlit and librosa...
echo (This may take a while)
py -3.11 -m pip install streamlit librosa

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Now you can use these files:
echo.
echo 1. train_model.bat   - Train the emotion model
echo 2. run_app.bat     - Open web interface
echo 3. run_gui.bat    - Open desktop GUI
echo.

pause