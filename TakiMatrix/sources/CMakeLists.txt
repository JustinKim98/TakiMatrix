set(target TakiMatrix)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries
		${CMAKE_CURRENT_SOURCE_DIR}/../../Includes
)

add_library(${target} test.cpp test.cu ../includes/test.h cpu_emulator/util/instruction_set.cpp ../includes/cpu_emulator/util/instruction_set.hpp)
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
