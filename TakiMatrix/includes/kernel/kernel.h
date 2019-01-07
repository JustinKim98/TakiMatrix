//
// Created by jwkim98 on 19. 1. 6.
//

#ifndef TAKIMATRIX_KERNEL_H
#define TAKIMATRIX_KERNEL_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void addKernel(const float* first, const float* second, unsigned int size);

__global__ void subKernel(const float* first, const float* second, unsigned int size);

__global__ void multiplyKernel(const float* first, const float* second,
        float* result, unsigned int firstRowNum,
        unsigned int middle, unsigned int secondColNum);

#endif // TAKIMATRIX_KERNEL_H
