#ifndef CUBBYDNN_TEST_H
#define CUBBYDNN_TEST_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
__global__ void add_kernel(int *a, int *b, int size);
void add_with_cuda();

#endif