@echo off
REM Set environment variables for C++ standard and build settings
setlocal

REM Set C++ standard (this is typically handled by CMake or MSBuild directly)
REM No direct equivalent to setting the C++ standard in a batch script.
REM This would be handled in the CMakeLists.txt file or project settings.

REM Export compile commands (handled by CMake)
REM Batch scripts don't directly control exporting compile commands; use CMake flags.

REM Define common include directories
set COMMON_INCLUDE_DIRECTORIES=%~dp0..\..\..\..\..
echo Common Include Directories: %COMMON_INCLUDE_DIRECTORIES%

REM Define ExecuteTorch root directory
if not defined EXECUTORCH_ROOT (
    set EXECUTORCH_ROOT=%~dp0..\..
)
echo ExecuteTorch Root: %EXECUTORCH_ROOT%

REM Include directories (handled in CMake)
REM No direct equivalent in batch script for `include_directories`.

REM Define OpenVINO include directories
set OPENVINO_INCLUDE_DIRS=C:\Users\ynimmaga\win_build\openvino\src\core\include;C:\Users\ynimmaga\win_build\openvino\src\inference\include
echo OpenVINO Include Directories: %OPENVINO_INCLUDE_DIRS%

#REM Define GFLAGS include directory
#set GFLAGS_INCLUDE_DIRS=%EXECUTORCH_ROOT%\third-party\gflags\build\include
#echo GFLAGS Include Directory: %GFLAGS_INCLUDE_DIRS%

REM Define OpenVINO library paths
set OPENVINO_LIB_PATH=C:\Users\ynimmaga\win_build\openvino\bin\intel64\Release
set OPENVINO_LIBS=%OPENVINO_LIB_PATH%\openvino.lib;%OPENVINO_LIB_PATH%\openvino_ir_frontend.lib;%OPENVINO_LIB_PATH%\openvino_c.lib;%OPENVINO_LIB_PATH%\openvino_intel_cpu_plugin.lib;%OPENVINO_LIB_PATH%\openvino_intel_gpu_plugin.lib;%OPENVINO_LIB_PATH%\openvino_auto_plugin.lib
echo OpenVINO Library Paths: %OPENVINO_LIB_PATH%

REM Build and link OpenVINO backend library
REM Generate build files with CMake
cmake -DCMAKE_CXX_STANDARD=17 ^
      -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
      -DCMAKE_CXX_FLAGS="-I C:/Users/ynimmaga/win_build/openvino/src/core/include;C:/Users/ynimmaga/win_build/openvino/src/inference/include;C:/Users/ynimmaga/win_build/executorch/third-party/gflags/build/include" ^
      -DCMAKE_LIBRARY_PATH=%OPENVINO_LIB_PATH% ^
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ^
      -DCMAKE_INSTALL_PREFIX=cmake-openvino-out ^
      -DEXECUTORCH_BUILD_OPENVINO=ON ^
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON ^
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON ^
      -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON ^
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON ^
      -G "Visual Studio 17 2022" ^
      -B cmake-openvino-out

if errorlevel 1 (
    echo Error: CMake configuration failed.
    exit /b 1
)

REM Build the project using MSBuild or CMake
cmake --build cmake-openvino-out --config Release

if errorlevel 1 (
    echo Error: Build failed.
    exit /b 1
)

REM Install the library
cmake --install cmake-openvino-out --config Release

if errorlevel 1 (
    echo Error: Installation failed.
    exit /b 1
)

echo Build and installation of OpenVINO backend completed successfully.

endlocal
