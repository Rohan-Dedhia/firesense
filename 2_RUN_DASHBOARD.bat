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

echo Starting the FireSense dashboard...
echo A browser tab will open. Close this window to stop it.
echo.
%PY% -m streamlit run app.py
pause
