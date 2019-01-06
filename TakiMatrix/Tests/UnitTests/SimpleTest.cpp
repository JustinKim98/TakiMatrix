include_directories(${ CMAKE_CURRENT_SOURCE_DIR })

# Sources
file(GLOB sources
	${ CMAKE_CURRENT_SOURCE_DIR }/*.cpp)

# Build executable
add_executable(${target}
	${sources} SimpleTest.hpp)

# Project options{
set_target_properties(${target}
	PROPERTIES

	${DEFAULT_PROJECT_OPTIONS}
)

# Compile options
# GCC and Clang compiler options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
		# /wd4996       # -> disable warning: non-Standard std::tr1 namespace and TR1-only machinery (because of gtest)
		-Wno-unused-variable
		)
endif()
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
	CubbyDNN
	gtest)