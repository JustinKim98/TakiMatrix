
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void callMultiply(float* leftDataCpuPtr, float* rightDataCpuPtr,
        float* resultDataCpuPtr, size_t resultRow, size_t resultCol,
        size_t middleSize);