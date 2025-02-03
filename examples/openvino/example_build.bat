@echo off
setlocal enabledelayedexpansion

:: Exit immediately if a command fails
set "EXECUTORCH_ROOT="
for /F "delims=" %%i in ('powershell -NoProfile -Command "(Get-Location).Path + '\\..\\..'"') do set "EXECUTORCH_ROOT=%%i"
echo EXECUTORCH_ROOT=%EXECUTORCH_ROOT%

:: Define the main procedure
:main
:: Set build directory
set "build_dir=cmake-openvino-out"

:: Navigate to EXECUTORCH_ROOT
cd /d "%EXECUTORCH_ROOT%"
:: Remove build directory if it exists
if exist "%build_dir%" rd /s /q "%build_dir%"

:: Path to gflags include and library
set "GFLAGS_INCLUDE=%EXECUTORCH_ROOT%\third-party\gflags\build\include"
set "GFLAGS_LIB=%EXECUTORCH_ROOT%\third-party\gflags\build\lib\Release"

:: Build example
set "example_dir=examples/openvino"
set "example_build_dir=%build_dir%\%example_dir%"
set "cmake_prefix_path=%EXECUTORCH_ROOT%\%build_dir%\lib\cmake\ExecuTorch;%EXECUTORCH_ROOT%\%build_dir%\third-party\gflags;"

:: Remove the example build directory if it exists
if exist "%example_build_dir%" rd /s /q "%example_build_dir%"

:: OpenVINO original
cmake -DCMAKE_CXX_STANDARD=17 ^
      -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
      -DCMAKE_PREFIX_PATH="%cmake_prefix_path%" ^
      -DCMAKE_LIBRARY_PATH="%EXECUTORCH_ROOT%/backends/openvino/cmake-openvino-out/Release" ^
      -DCMAKE_INCLUDE_PATH="%GFLAGS_INCLUDE%" ^
      -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH ^
      -B"%example_build_dir%" ^
      %EXECUTORCH_ROOT%\%example_dir%

cmake --build "%example_build_dir%" -j5 --config Release

:: Switch back to the original directory
cd /d "%~dp0"

:: Print a success message
echo Build successfully completed.
exit /b 0
