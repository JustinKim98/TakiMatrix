//
// Created by jwkim98 on 19. 1. 6.
//

#ifndef TAKIMATRIX_KERNEL_H
#define TAKIMATRIX_KERNEL_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void addKernel(const float* first, const float* second, size_t size);

__global__ void subKernel(const float* first, const float* second, size_t size);

__global__ void multiplyKernel(const float* first, const float* second,
        float* result, size_t firstRowNum,
        size_t middle, size_t secondColNum);

#endif // TAKIMATRIX_KERNEL_H
