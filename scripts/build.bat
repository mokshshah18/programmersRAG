@echo off
rem Configure and build programmersRAG with MSVC + Ninja.
rem Usage: scripts\build.bat [preset] [extra cmake --build args]
rem Presets: release (default), debug, core (no llama.cpp, fast unit-test build)
setlocal
set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=release"

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

pushd "%~dp0.."
cmake --preset %PRESET%
if errorlevel 1 (popd & exit /b 1)
shift
cmake --build --preset %PRESET% %1 %2 %3 %4 %5 %6
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
