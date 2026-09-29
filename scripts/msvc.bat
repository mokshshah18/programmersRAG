@echo off
rem Runs a command inside the MSVC x64 developer environment, with Ninja on PATH.
rem Usage: scripts\msvc.bat cmake -S . -B build\core -G Ninja ...
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo error: vswhere.exe not found. Install Visual Studio with the "Desktop development with C++" workload.
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if "%VSROOT%"=="" (
  echo error: no Visual Studio installation with the C++ toolset was found.
  exit /b 1
)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
if errorlevel 1 exit /b 1
where ninja >nul 2>&1
if errorlevel 1 set "PATH=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
%*
exit /b %ERRORLEVEL%
