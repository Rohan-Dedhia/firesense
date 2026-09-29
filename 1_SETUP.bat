@echo off
setlocal
set PY=
where py >nul 2>&1 && set PY=py -3
if "%PY%"=="" (where python >nul 2>&1 && set PY=python)
if "%PY%"=="" (
  echo.
  echo Python was not found on your PATH.
  echo Open "Anaconda Prompt" from the Start menu and run these there instead:
  echo     cd "%%USERPROFILE%%\Desktop\firesense"
  echo     python -m pip install -r requirements.txt
  echo     python check_setup.py
  echo.
  pause
  exit /b 1
)
cd /d "%~dp0"

echo ==========================================================
echo  FireSense - one-time setup
echo ==========================================================
echo.
echo [1/3] Installing dependencies (this can take a few minutes)...
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
echo.
echo [2/3] Checking your machine (Python and packages)...
%PY% check_setup.py
echo.
echo [3/3] Running the test suite...
%PY% run_tests.py
echo.
echo ==========================================================
echo  If every line above says PASS, setup is DONE.
echo  Next: double-click 2_RUN_DASHBOARD.bat
echo ==========================================================
pause
