@echo off
REM ============================================================
REM  AI-Powered Grade Change Intelligence - Windows Setup
REM  Double-click this file (or run it from a terminal) once.
REM ============================================================

echo.
echo ==========================================================
echo   Setting up AI-Powered Grade Change Intelligence
echo ==========================================================
echo.
 
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] "python" was not found on your PATH.
    echo Install Python from https://www.python.org/downloads/
    echo During install, make sure to check "Add python.exe to PATH".
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment in .\venv ...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create the virtual environment.
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment ...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies from requirements.txt ...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Some packages may have failed ^(often "shap"^).
    echo The app still works without it - it will fall back to a
    echo built-in explanation method automatically. Continuing...
    echo.
)

echo [4/4] Generating historical data and training the model ...
python train.py
if %errorlevel% neq 0 (
    echo [ERROR] Training failed. Scroll up to see the error message.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo   Setup complete! Double-click start.bat to launch the app.
echo ==========================================================
pause
