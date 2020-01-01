//
// Created by jwkim98 on 19/02/02.
//

#include "../../../../includes/cpu_emulator/execution_units/kernels/kernels.h"

namespace kernel {
    __global__ void add_kernel(float* first, float* second, size_t size)
    {
        size_t compute_stride = blockDim.x*gridDim.x;
        size_t compute_index = blockIdx.x*blockDim.x+threadIdx.x;

        while (compute_index<size) {
            second[compute_index] = first[compute_index]+second[compute_index];
            /// for cases when matrix size exceeds compute_index
            compute_index += compute_stride;
        }
    }

    __global__ void sub_kernel(float* first, float* second, size_t size)
    {
        size_t compute_stride = blockDim.x*gridDim.x;
        size_t compute_index = blockIdx.x*blockDim.x+threadIdx.x;

        while (compute_index<size) {
            second[compute_index] = first[compute_index]-second[compute_index];
            /// for cases when matrix size exceeds compute_index
            compute_index += compute_stride;
        }
    }

    __global__ void mul_kernel(float* first, float* second, float* result,
            size_t middle_size, size_t first_row_num,
            size_t second_col_num, size_t dimension_num)
    {
        /// result will have first_row rows and second_col columns
        size_t col_dim = first_row_num;
        size_t row_dim = second_col_num;
        size_t dim_size = col_dim*row_dim;
        size_t size = col_dim*row_dim*dimension_num;

        size_t compute_stride = blockDim.x*gridDim.x;
        size_t compute_index = blockIdx.x*blockDim.x+threadIdx.x;

        while (compute_index<size) {
            float sum = 0;
            size_t compute_row = compute_index/row_dim;
            size_t compute_col = compute_index%row_dim;
            size_t compute_dim = compute_index/dim_size;

            for (int count = 0; count<middle_size; count++) {
                sum += first[compute_row*col_dim+count]*
                        second[compute_col+count*row_dim + compute_dim*dim_size];
            }
            result[compute_row*row_dim+compute_col] = sum;
            compute_index += compute_stride;
        }
    }

    template<typename Func>
    __global__ void dot_kernel(float* data, Func func, size_t size)
    {
        size_t compute_stride = blockDim.x*gridDim.x;
        size_t compute_index = blockIdx.x*blockDim.x+threadIdx.x;

        while (compute_index<size) {
            data[compute_index] = func(data);
            /// for cases when matrix size exceeds compute_index
            compute_index += compute_stride;
        }
    }
} // namespace kernel
