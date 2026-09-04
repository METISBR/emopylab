@echo off
setlocal
cd /d "%~dp0"
title EmoPyLab - Windows setup (Python + dependencias)

echo ============================================================
echo EmoPyLab - instalador completo para Windows
echo   1. Localiza ou instala o Python 3.11
echo   2. Cria o ambiente virtual .venv no projeto
echo   3. Instala as dependencias (requirements.txt)
echo   4. Valida imports e o benchmark GCS-MaOEA (dry-run)
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM [1/4] Python 3.11 (recomendado pelo requirements.txt)
REM ---------------------------------------------------------------
set "PYTHON_CMD="
py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.11"
if defined PYTHON_CMD goto :have_python

echo [INFO] Python 3.11 nao encontrado. Tentando instalar via winget...
winget --version >nul 2>&1
if errorlevel 1 goto :no_winget

winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements

py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.11"
if defined PYTHON_CMD goto :have_python
goto :reopen_terminal

:no_winget
echo [INFO] winget indisponivel. Baixando o instalador oficial (python.org)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile \"$env:TEMP\python-3.11.9-amd64.exe\""
if errorlevel 1 goto :download_error

echo [INFO] Instalando Python 3.11.9 (silencioso, somente usuario atual)...
"%TEMP%\python-3.11.9-amd64.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1
if errorlevel 1 goto :install_error

py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.11"
if defined PYTHON_CMD goto :have_python
goto :reopen_terminal

:have_python
echo [INFO] Usando:
%PYTHON_CMD% --version
echo.

REM ---------------------------------------------------------------
REM [2/4] Ambiente virtual do projeto (.venv)
REM ---------------------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [INFO] .venv ja existe - reutilizando.
) else (
    echo [INFO] Criando ambiente virtual .venv ...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :install_error
)
set "VPY=%CD%\.venv\Scripts\python.exe"

REM ---------------------------------------------------------------
REM [3/4] Dependencias
REM   mlx (Apple Silicon) tem marcador "darwin" no requirements.txt
REM   e e ignorado automaticamente no Windows - backend cai para cpu.
REM ---------------------------------------------------------------
echo.
echo [INFO] Atualizando pip...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :install_error

echo.
echo [INFO] Instalando dependencias do EmoPyLab...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :install_error

echo.
echo [INFO] Instalando pytest (testes de contrato do benchmark)...
"%VPY%" -m pip install "pytest>=9,<10"
if errorlevel 1 goto :install_error

REM ---------------------------------------------------------------
REM [4/4] Validacao
REM ---------------------------------------------------------------
echo.
echo [INFO] Validando imports obrigatorios...
"%VPY%" -c "import numpy, scipy, matplotlib, psutil, PySide6, qt_material, qt_material_icons; print('[OK] Imports obrigatorios disponiveis.')"
if errorlevel 1 goto :install_error

echo.
echo [INFO] Validando o benchmark GCS-MaOEA (dry-run, nada e executado)...
"%VPY%" tests\gcs_maoea_benchmark.py --dry-run --debug --n-runs 2 --output-dir "%TEMP%\emopylab_setup_check"
if errorlevel 1 goto :install_error

echo.
echo ============================================================
echo Instalacao concluida com sucesso.
echo ============================================================
echo.
echo Como usar (a partir desta pasta):
echo   Aplicativo:  .venv\Scripts\python.exe EmoPyLab.py
echo   Benchmark:   .venv\Scripts\python.exe tests\gcs_maoea_benchmark.py --help
echo   Testes:      .venv\Scripts\python.exe -m pytest tests\test_gcs_maoea_benchmark_contracts.py -q
echo.
echo Dicas para runs longos no Windows (nao existe caffeinate):
echo   powercfg /change standby-timeout-ac 0
echo   powercfg /change monitor-timeout-ac 10
echo.
echo Ollama (opcional, apenas para LARC_NSGA3): instale manualmente,
echo inicie o servico e rode:  ollama pull qwen2.5:1.5b
echo.
pause
exit /b 0

:reopen_terminal
echo.
echo [AVISO] O Python foi instalado, mas este terminal ainda nao o enxerga.
echo Feche esta janela, abra um NOVO Prompt de Comando e execute
echo setup_windows.bat novamente (ele continua de onde parou).
pause
exit /b 1

:download_error
echo.
echo [ERRO] Falha ao baixar o instalador do Python.
echo Instale manualmente o Python 3.11.x de https://www.python.org/downloads/windows/
echo marcando "Add python.exe to PATH", e rode este script de novo.
pause
exit /b 1

:install_error
echo.
echo [ERRO] A instalacao falhou. Revise as mensagens acima.
echo Se o erro for de rede/proxy, tente novamente; o script e idempotente.
pause
exit /b 1
