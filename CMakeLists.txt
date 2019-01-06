# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("TakiMatrix" LANGUAGES CXX CUDA)

#policies
cmake_policy(SET CMP0074 NEW)

#incudes
include(${CMAKE_CURRENT_SOURCE_DIR}/Build/CMAKE/CompileOptions.cmake)


# Include sub-projects.
add_subdirectory ("TakiMatrix")

find_package(CUDA QUIET)
if(CUDA_FOUND)
	find_package(CUDA REQUIRED)
	include_directories(${CUDA_INCLUDE_DIRS})
endif()