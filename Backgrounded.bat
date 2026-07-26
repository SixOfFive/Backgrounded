@echo off
rem ===========================================================================
rem  Backgrounded - double-click launcher
rem
rem  Verifies Python and the three dependencies using a console interpreter, so
rem  any problem is visible, then relaunches via pythonw.exe so the app runs
rem  with no console window behind it.
rem
rem  Once started it lives in the system tray - look for the candle icon near
rem  the clock and right-click it for the menu. Running it twice is harmless:
rem  the app holds a single-instance mutex and the second copy just exits.
rem ===========================================================================
setlocal
cd /d "%~dp0"
title Backgrounded

set "PY="
set "PYW="
where python.exe  >nul 2>&1 && set "PY=python.exe"
where pythonw.exe >nul 2>&1 && set "PYW=pythonw.exe"
if not defined PY (
    where py.exe >nul 2>&1 && set "PY=py.exe"
)
if not defined PYW (
    if defined PY set "PYW=%PY%"
)

if not defined PY goto :nopython

%PY% -c "import pygame, PIL, numpy" >nul 2>&1
if errorlevel 1 goto :deps

:launch
start "" %PYW% "%~dp0run.pyw" %*
exit /b 0

:deps
echo.
echo   Backgrounded needs three packages: pygame-ce, pillow, numpy
echo.
choice /c YN /n /m "   Install them now? [Y/N] "
if errorlevel 2 exit /b 1
echo.
%PY% -m pip install --upgrade pygame-ce pillow numpy
if errorlevel 1 goto :pipfailed
%PY% -c "import pygame, PIL, numpy" >nul 2>&1
if errorlevel 1 goto :pipfailed
goto :launch

:pipfailed
echo.
echo   Could not install the packages. Try this yourself:
echo       %PY% -m pip install pygame-ce pillow numpy
echo.
pause
exit /b 1

:nopython
echo.
echo   Python was not found on your PATH.
echo.
echo   Install Python 3.11 or newer from https://python.org/downloads
echo   and tick "Add python.exe to PATH" during setup.
echo.
pause
exit /b 1
