
#include "../../includes/kernel/kernelCaller.h"

void callMultiply(float* leftDataCpuPtr, float* rightDataCpuPtr,
        float* resultDataCpuPtr, size_t resultRow, size_t resultCol,
        size_t middleSize){

    size_t leftDataSize = sizeof(float)*resultRow*middleSize;
    size_t rightDataSize = sizeof(float)*middleSize*resultCol;
    size_t resultDataSize = sizeof(float)*resultRow*resultCol;

    float* leftDataGpuPtr;
    float* rightDataGpuPtr;
    float* resultDataGpuPtr;

    cudaMalloc(&leftDataGpuPtr, leftDataSize);
    cudaMalloc(&rightDataGpuPtr, rightDataSize);
    cudaMalloc(&resultDataGpuPtr, resultDataSize);

    cudaMemcpy(leftDataGpuPtr, leftDataCpuPtr, leftDataSize,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightDataGpuPtr, rightDataCpuPtr, rightDataSize,
            cudaMemcpyHostToDevice);

    size_t numThreads;
    size_t numBlocks;
    if (resultSize<2048) {
        numThreads = resultSize;
        numBlocks = 1;
    }
    else {
        numBlocks = resultSize/2048+1;
        numThreads = 2048;
        // TODO optimize allocated thread size
    }

    multiplyKernel<<<numBlocks, numThreads>>>(leftDataGpuPtr, rightDataGpuPtr, resultDataGpuPtr, resultRow,
            middleSize, resultCol);

    cudaMemcpy(resultDataCpuPtr, resultDataGpuPtr, resultDataSize, cudaMemcpyDeviceToHost);
    cudaFree(leftDataGpuPtr);
    cudaFree(rightDataGpuPtr);
    cudaFree(resultDataGpuPtr);
}