@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo EmoPyLab - Windows Python dependency installer
echo ============================================================
echo.
echo This script installs Python packages globally for the selected
echo Python interpreter. Ollama and the local model are NOT installed.
echo.

set "PYTHON_CMD="
py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.11"

if not defined PYTHON_CMD (
    py -3 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo [ERROR] Python was not found.
    echo Install Python 3.11.x from https://www.python.org/downloads/windows/
    echo and enable the option "Add python.exe to PATH".
    exit /b 1
)

echo [INFO] Using:
%PYTHON_CMD% --version
echo.

echo [1/3] Updating pip...
%PYTHON_CMD% -m pip install --upgrade pip
if errorlevel 1 goto :install_error

echo.
echo [2/3] Installing EmoPyLab dependencies...
%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 goto :install_error

echo.
echo [3/3] Validating required imports...
%PYTHON_CMD% -c "import numpy, scipy, matplotlib, psutil, PySide6, qt_material, qt_material_icons; print('[OK] Required Python imports are available.')"
if errorlevel 1 goto :install_error

echo.
echo ============================================================
echo Python dependencies installed successfully.
echo ============================================================
echo Ollama remains a manual prerequisite for LARC_NSGA3.
echo Install Ollama separately, start it, and pull the configured model:
echo     ollama pull qwen2.5:1.5b
echo.
echo Start the desktop application with:
echo     %PYTHON_CMD% EmoPyLab.py
exit /b 0

:install_error
echo.
echo [ERROR] Installation failed. Review the messages above.
exit /b 1
