@echo off
REM ============================================================
REM  AI-Powered Grade Change Intelligence - Launch Dashboard
REM  Double-click this file any time after running setup.bat
REM ============================================================

if not exist venv\Scripts\activate.bat (
    echo [ERROR] No virtual environment found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist models\bw_deviation_model.pkl (
    echo No trained model found yet - training now ...
    python train.py
)

echo Launching dashboard - your browser will open automatically...
streamlit run app.py

pause
