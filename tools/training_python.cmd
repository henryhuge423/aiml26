@echo off
setlocal
set "PROJECT=%~dp0.."
set "PATH=%PROJECT%\.envs\training\Library\bin;%PROJECT%\.envs\training\Scripts;%PROJECT%\.envs\training;%PATH%"
set "PYTHONNOUSERSITE=1"
"%PROJECT%\.envs\training\python.exe" %*
exit /b %errorlevel%
