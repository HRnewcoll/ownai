@echo off
:: ============================================================
:: OwnAI – One-click installer for Windows
:: Usage: Double-click setup.bat or run in a Command Prompt
:: ============================================================

echo.
echo   ^[^[1m OwnAI Setup^[^[0m
echo   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo   [FAIL] Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo   [OK] Python %PY_VER% found

:: Create virtual environment
if not exist venv (
    echo.
    echo   Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo   [OK] Virtual environment ready

:: Upgrade pip
pip install --upgrade pip --quiet

:: Install dependencies
echo.
echo   Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt --quiet
echo   [OK] Dependencies installed

:: Copy .env if missing
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo   [OK] Created .env from .env.example
        echo   [!!] Edit .env to set your SECRET_KEY and other settings
    )
)

:: Check GPU
echo.
python -c "try:\n    import torch\n    if torch.cuda.is_available():\n        name=torch.cuda.get_device_name(0)\n        mem=round(torch.cuda.get_device_properties(0).total_memory/1e9,1)\n        print(f'  [GPU] {name} ({mem} GB)')\n    else:\n        print('  [WARN] No CUDA GPU detected - training will run on CPU (slower)')\nexcept: print('  [WARN] Could not check GPU')"

:: Done
echo.
echo   ============================================
echo   Setup complete!
echo.
echo   To start OwnAI:
echo     venv\Scripts\activate
echo     python app.py
echo.
echo   Then open  http://localhost:5000  in your browser.
echo   ============================================
echo.
pause
