//
// Created by jwkim98 on 19. 1. 6.
//

#ifndef TAKIMATRIX_KERNEL_H
#define TAKIMATRIX_KERNEL_H
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void addKernel(float* first, float* second, unsigned int size);

void subKernel(float* first, float* second, unsigned int size);


#endif //TAKIMATRIX_KERNEL_H
