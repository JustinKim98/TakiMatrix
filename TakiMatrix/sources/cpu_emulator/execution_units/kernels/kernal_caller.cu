//
// Created by jwkim98 on 19/02/03.
//

#include "../../../../includes/cpu_emulator/execution_units/kernels/kernal_caller.h"

namespace kernel {
    void call_add(float* operand_a, float* operand_b, size_t size)
    {
        float* device_operand_a;
        float* device_operand_b; // stores operand_second and result

        size_t byte_size = size*sizeof(float);

        cudaMalloc((void**) &device_operand_a, byte_size);
        cudaMalloc((void**) &device_operand_b, byte_size);

        cudaMemcpy(device_operand_a, operand_a, byte_size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_operand_b, operand_b, byte_size, cudaMemcpyHostToDevice);

        add_kernel <<<1, 1024 >>>(device_operand_a, device_operand_b, size);

        cudaMemcpy(operand_b, device_operand_b, byte_size, cudaMemcpyDeviceToHost);
        cudaFree(device_operand_a);
        cudaFree(device_operand_b);
    }

    void call_sub(float* operand_a, float* operand_b, size_t size)
    {
        float* device_operand_a;
        float* device_operand_b; // stores operand_second and result

        size_t byte_size = size*sizeof(float);

        cudaMalloc((void**) &device_operand_a, byte_size);
        cudaMalloc((void**) &device_operand_b, byte_size);

        cudaMemcpy(device_operand_a, operand_a, byte_size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_operand_b, operand_b, byte_size, cudaMemcpyHostToDevice);

        sub_kernel <<<1, 1024 >>>(device_operand_a, device_operand_b, size);

        cudaMemcpy(operand_b, device_operand_b, byte_size, cudaMemcpyDeviceToHost);
        cudaFree(device_operand_a);
        cudaFree(device_operand_b);
    }

    void call_mul(float* operand_a, float* operand_b, float* result,
            size_t middle_count, size_t first_row_count,
            size_t second_col_count, size_t dimension_count)
    {
        float* device_operand_a;
        float* device_operand_b;
        float* device_result;

        size_t operand_a_size = first_row_count*middle_count*sizeof(float);
        size_t operand_b_size = middle_count*second_col_count*sizeof(float);
        size_t result_size = first_row_count*second_col_count*sizeof(float);

        cudaMalloc((void**) &device_operand_a, operand_a_size);
        cudaMalloc((void**) &device_operand_b, operand_b_size);
        cudaMalloc((void**) &device_result, result_size);

        cudaMemcpy(device_operand_a, operand_a, operand_a_size,
                cudaMemcpyHostToDevice);
        cudaMemcpy(device_operand_b, operand_b, operand_b_size,
                cudaMemcpyHostToDevice);

        mul_kernel <<<1, 1024 >>>(device_operand_a, device_operand_b, device_result,
                middle_count, first_row_count, second_col_count,
                dimension_count);
    }

    template<typename Func>
    void call_dot(float* operand_a, Func func, size_t size)
    {
        float* device_operand_a;

        size_t byte_size = size*sizeof(float);

        cudaMalloc((void**) &device_operand_a, byte_size);

        cudaMemcpy(device_operand_a, operand_a, byte_size, cudaMemcpyHostToDevice);

        dot_kernel <<<1, 1024 >>>(device_operand_a, func, size);

        cudaMemcpy(operand_a, device_operand_a, byte_size, cudaMemcpyDeviceToHost);
        cudaFree(device_operand_a);
    }

} // namespace kernel