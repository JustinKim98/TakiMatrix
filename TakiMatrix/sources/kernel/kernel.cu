#include "../../includes/kernel.h"
#include <curand_mtgp32_kernel.h>

__global__ void addKernel(float* first, float* second, unsigned int size)
{
    unsigned int computeStride = blockDim.x*gridDim.x;
    unsigned int computeIndex = blockIdx.x*blockDim.x+threadIdx.x;

    while (computeIndex<size) {
        second[computeIndex] = first[computeIndex]+second[computeIndex];
        computeIndex += computeStride;
    }
}

__global__ void subKernel(float* first, float* second, unsigned int size)
{
    unsigned int computeStride = blockDim.x*gridDim.x;
    unsigned int computeIndex = blockIdx.x*blockDim.x+threadIdx.x;

    while (computeIndex<size) {
        second[computeIndex] = first[computeIndex]-second[computeIndex];
        computeIndex += computeStride;
    }
}

__global__ void multiplyKernel(const float* first, const float* second,
        float* result, unsigned int firstRowNum,
        unsigned int middle, unsigned int secondColNum)
{

    /// result will have first_row rows and second_col columns
    unsigned int resultRowNum = firstRowNum;
    unsigned int resultColNum = secondColNum;
    unsigned int size = firstRowNum*secondColNum;

    unsigned int computeStride = blockDim.x*gridDim.x;
    unsigned int computeIndex = blockIdx.x*blockDim.x+threadIdx.x;

    while (computeIndex<size) {
        float sum = 0;
        unsigned int computeRow = computeIndex/resultColNum;
        unsigned int computeCol = computeIndex%resultColNum;

        for (int count = 0; count<middle; count++) {
            sum +=
                    first[computeRow+count]*second[computeCol+count*resultRowNum];
        }
        result[computeRow*resultRowNum+computeCol] = sum;
        computeIndex += computeStride;
    }
}