@echo off
cd /d "%~dp0"

echo ========================================
echo   Emotion Recognition - Training
echo ========================================
echo.

py -3.11 src/training/train_sklearn.py --data-dir data/raw

pause