#include "../../includes/kernel.h"

void addKernel(float* first, float* second, unsigned int size)
{
    unsigned int computeStride = blockDim.x*gridDim.x;
    unsigned int computeIndex = blockIdx.x*blockDim.x+threadIdx.x;

    while(computeIndex < size){
        second[computeIndex] = first[computeIndex] + second[computeIndex];
        computeIndex += computeStride;
    }
}

void subKernel(float* first, float* second, unsigned int size)
{
    unsigned int computeStride = blockDim.x*gridDim.x;
    unsigned int computeIndex = blockIdx.x*blockDim.x+threadIdx.x;

    while(computeIndex < size){
        second[computeIndex] = first[computeIndex] - second[computeIndex];
        computeIndex += computeStride;
    }
}