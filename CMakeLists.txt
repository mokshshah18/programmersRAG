cmake_minimum_required(VERSION 3.24)
project(programmersRAG VERSION 2.0.0 LANGUAGES C CXX)

# The language standard is set per target (C++20 for this project, see
# cmake/PragModule.cmake) so that third-party code keeps its own standard.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(MSVC)
  # Static runtime keeps the executables self-contained; llama.cpp follows the same setting.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(PRAG_WITH_LLAMA "Link llama.cpp for local inference, embeddings, fine-tuning and quantisation" ON)
option(PRAG_BUILD_TESTS "Build unit tests" ON)
option(PRAG_BUILD_BENCH "Build the experiment and benchmark drivers" ON)
option(PRAG_NATIVE_SIMD "Compile with AVX2 on x86-64" ON)
set(PRAG_LLAMA_TAG "b11257" CACHE STRING "llama.cpp release tag")
set(PRAG_DEMO_REPO "${CMAKE_SOURCE_DIR}/Demo repo/mimalloc" CACHE PATH "Corpus used by integration tests and experiments")

set(PRAG_ALL_MODULES core exec graph parse embed index kde retrieve context edit llm tune engine server)
set(PRAG_MODULES "${PRAG_ALL_MODULES}" CACHE STRING "Semicolon-separated list of modules to build")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CTest)
include(PragDeps)
include(PragModule)

foreach(module IN LISTS PRAG_MODULES)
  if(EXISTS "${CMAKE_SOURCE_DIR}/src/${module}/CMakeLists.txt")
    add_subdirectory("src/${module}")
  endif()
endforeach()

if(EXISTS "${CMAKE_SOURCE_DIR}/src/cli/CMakeLists.txt" AND "server" IN_LIST PRAG_MODULES)
  add_subdirectory(src/cli)
endif()
if(PRAG_BUILD_BENCH AND EXISTS "${CMAKE_SOURCE_DIR}/bench/CMakeLists.txt" AND "server" IN_LIST PRAG_MODULES)
  add_subdirectory(bench)
endif()
