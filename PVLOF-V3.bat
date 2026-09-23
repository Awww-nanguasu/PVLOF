@echo off
setlocal EnableExtensions
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0PVLOF-V3.ps1"
set "PVLOF_EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %PVLOF_EXIT_CODE%
