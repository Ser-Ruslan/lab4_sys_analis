@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv_lab5\Scripts\python.exe" (
    py -3 -m venv .venv_lab5
    if errorlevel 1 goto failed
)
".venv_lab5\Scripts\python.exe" -c "import numpy, scipy, matplotlib" >nul 2>&1
if errorlevel 1 (
    ".venv_lab5\Scripts\python.exe" -m pip install -r requirements_lab5.txt
    if errorlevel 1 goto failed
)
".venv_lab5\Scripts\python.exe" lab5_variant1.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Ошибка запуска. Нужен Python 3.9+ и интернет для установки библиотек.
pause
exit /b 1
