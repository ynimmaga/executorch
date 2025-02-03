set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.19)
project(openvino_runner_example)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Include OpenVINO backend headers
include_directories("${EXECUTORCH_ROOT}/backends/openvino/runtime")

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
#set(EXECUTORCH_SRCS_FILE
#    "${CMAKE_CURRENT_BINARY_DIR}/../../../build/executorch_srcs.cmake"
#)
#message("executorch sources files")
#message(${EXECUTORCH_SRCS_FILE})
#extract_sources(${EXECUTORCH_SRCS_FILE})
#include(${EXECUTORCH_SRCS_FILE})

set(_openvino_executor_runner__srcs ${CMAKE_CURRENT_LIST_DIR}/../openvino/executor_runner/openvino_executor_runner.cpp) 

# preprocess executor runner src files
list(PREPEND _openvino_executor_runner__srcs
     ${CMAKE_CURRENT_LIST_DIR}/../openvino/executor_runner/openvino_executor_runner.cpp
)

# build executor runner
add_executable(openvino_executor_runner ${_openvino_executor_runner__srcs})
target_include_directories(
  openvino_executor_runner PUBLIC ${_common_include_directories}
)

set(GFLAGS_INCLUDE_DIRS ${EXECUTORCH_ROOT}/third-party/gflags/build/include)
target_include_directories(
  openvino_executor_runner PUBLIC ${GFLAGS_INCLUDE_DIRS}
)

set(INTEL_OPENVINO_DIR "$ENV{INTEL_OPENVINO_DIR}")

# OpenVINO include directories
set(OPENVINO_INCLUDE_DIRS 
    "${INTEL_OPENVINO_DIR}/runtime/include"
    "${INTEL_OPENVINO_DIR}/src/core/include"
    "${INTEL_OPENVINO_DIR}/src/inference/include"
)
target_include_directories(
  openvino_executor_runner PUBLIC ${OPENVINO_INCLUDE_DIRS}
)

# Set the path to the library directory
set(LIBRARY_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

# List the libraries you want to link (without the 'lib' prefix and file extension)
set(LIBRARIES_TO_LINK
    "${EXECUTORCH_ROOT}/third-party/gflags/build/lib/Release/gflags.lib"
    "${EXECUTORCH_ROOT}/pip-out/temp.win-amd64-cpython-313/cmake-out/extension/data_loader/Release/extension_data_loader.lib"
    "${EXECUTORCH_ROOT}/backends/openvino/cmake-openvino-out/Release/openvino_backend.lib"  
    "${EXECUTORCH_ROOT}/pip-out/temp.win-amd64-cpython-313/cmake-out/Release/executorch.lib"
    "${EXECUTORCH_ROOT}/pip-out/temp.win-amd64-cpython-313/cmake-out/Release/executorch_core.lib"  
    "${EXECUTORCH_ROOT}/pip-out/temp.win-amd64-cpython-313/cmake-out/backends/xnnpack/third-party/pthreadpool/Release/pthreadpool.lib"
    "${EXECUTORCH_ROOT}/pip-out/temp.win-amd64-cpython-313/cmake-out/extension/runner_util/Release/extension_runner_util.lib"
    "${EXECUTORCH_ROOT}/pip-out/temp.win-amd64-cpython-313/cmake-out/extension/tensor/Release/extension_tensor.lib"
)

# Add the library directory to the link search path
# link_directories(${LIBRARY_DIR})

# Link all libraries at once
target_link_libraries(openvino_executor_runner PRIVATE ${LIBRARIES_TO_LINK})
target_link_options(openvino_executor_runner PRIVATE "/OPT:NOREF")
target_link_options(openvino_executor_runner PRIVATE "/WHOLEARCHIVE:C:/Users/ynimmaga/win_build/executorch/backends/openvino/cmake-openvino-out/Release/openvino_backend.lib")


#set_target_properties(
#  openvino_executor_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
#)


get_filename_component(
  EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)
