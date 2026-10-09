@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failure
echo Installation complete. Run start_windows.cmd.
pause
exit /b 0
:failure
echo Installation failed. Python 3.11 and internet access are required.
pause
exit /b 1
