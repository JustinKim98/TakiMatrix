//
// Created by jwkim98 on 19/02/02.
//

#ifndef TAKIMATRIX_ESSENTIAL_KERNELS_H
#define TAKIMATRIX_ESSENTIAL_KERNELS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace kernel {
    __global__ void add_kernel(float* first, float* second, size_t size);

    __global__ void sub_kernel(float* first, float* second, size_t size);

    __global__ void mul_kernel(float* first, float* second, float* result,
            size_t first_row_size, size_t first_row_num,
            size_t second_col_size, size_t dimension_num);

    template<typename Func>
    __global__ void dot_kernel(float* data, Func func, size_t size);
} // namespace kernel
#endif // TAKIMATRIX_ESSENTIAL_KERNELS_H
