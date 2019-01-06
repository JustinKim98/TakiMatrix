#ifndef CUBBYDNN_TEST_H
#define CUBBYDNN_TEST_H

#include "jetbrains_parser.h"

__global__ void add_kernel(int *a, int *b, int size);
void add_with_cuda();

#endif