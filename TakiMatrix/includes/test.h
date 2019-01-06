#ifndef CUBBYDNN_TEST_H
#define CUBBYDNN_TEST_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

__global__ void test_kernel(int* a, int* b, int size);
void add_with_cuda();

#endif