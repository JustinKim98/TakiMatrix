# Target name
set(target UnitTests)

# Includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../includes)

# Sources
file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build executable
add_executable(${target}
    ${sources} UnitTests.cpp)

# Project options{
set_target_properties(${target}
		PROPERTIES

		${DEFAULT_PROJECT_OPTIONS}
		)

target_compile_options(${target}
		PRIVATE
		${DEFAULT_COMPILE_OPTIONS}
		)
target_compile_definitions(${target}
		PRIVATE
		)

# Link libraries
target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
	TakiMatrix
    gtest)