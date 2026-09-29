@echo off
setlocal
set PY=
where py >nul 2>&1 && set PY=py -3
if "%PY%"=="" (where python >nul 2>&1 && set PY=python)
cd /d "%~dp0"
echo ==========================================================
echo  Checking the USB link to the node
echo ==========================================================
echo.
echo CLOSE the Arduino IDE Serial Monitor before continuing!
echo.
pause
%PY% -m pip install pyserial -q
%PY% read_serial.py
pause
