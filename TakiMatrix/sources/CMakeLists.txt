set(target TakiMatrix)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries
		${CMAKE_CURRENT_SOURCE_DIR}/../../Includes
)

add_library(${target} test.cpp test.cu
		../includes/test.h
		cpu_emulator/processor_util/instruction_set.cpp
		../includes/cpu_emulator/processor_util/instruction_set.hpp
		cpu_emulator/processor_util/matrix_object.cpp
		../includes/cpu_emulator/processor_util/matrix_object.hpp
		cpu_emulator/front_end/fetch.cpp
		../includes/cpu_emulator/front_end/fetch.hpp
		cpu_emulator/front_end/instruction_cache.cpp
		../includes/cpu_emulator/front_end/instruction_cache.hpp
		cpu_emulator/system_agent/system_agent.cpp
		../includes/cpu_emulator/system_agent/system_agent.hpp
		../includes/util/matrix.hpp util/matrix.cpp)

set_target_properties( ${target}
		PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        )

# Compile options
target_compile_options(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}

        INTERFACE
        )

target_link_libraries(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        ${DEFAULT_LIBRARIES}

        INTERFACE
        )

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include)
