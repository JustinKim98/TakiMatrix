# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.

cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project ("TakiMatrix" LANGUAGES CXX CUDA)

#policies
cmake_policy(SET CMP0074 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Build/CMake")


#incudes
include(${CMAKE_CURRENT_SOURCE_DIR}/Build/CMAKE/IncludeCuda.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Build/CMAKE/CompileOptions.cmake)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

message(STATUS "PROJECT_BINARY_DIR: " ${PROJECT_BINARY_DIR})

# Include sub-projects.
add_subdirectory (TakiMatrix/Tests/UnitTests)
add_subdirectory (TakiMatrix/sources)


# Build type - Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#


# Includes
include_directories(TakiMatrix/Includes)
include_directories(TakiMatrix/Libraries)
include_directories(TakiMatrix/Libraries/googletest/googletest/include)

# Overrides
set(CMAKE_MACOSX_RPATH ON)

# Project modules
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(TakiMatrix/Libraries/googletest EXCLUDE_FROM_ALL)
cmake_policy(SET CMP0063 NEW)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(TakiMatrix/Libraries/googlebenchmark)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
