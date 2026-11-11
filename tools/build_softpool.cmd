@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
set "PROJECT=E:\jhu\1-AIML_for_GlobalHealth\team_project"
set "PATH=%PROJECT%\.envs\training\Library\bin;%PROJECT%\.envs\training\Scripts;%PROJECT%\.envs\training;%PATH%"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "TORCH_CUDA_ARCH_LIST=8.6"
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=2"
set "TEMP=%PROJECT%\.build\tmp"
set "TMP=%TEMP%"
"%PROJECT%\.envs\training\python.exe" -m pip install --disable-pip-version-check --no-build-isolation --no-deps "%PROJECT%\.build\softpool"
exit /b %errorlevel%
