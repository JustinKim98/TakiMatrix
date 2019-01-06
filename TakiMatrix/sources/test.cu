#include "../includes/test.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "thrust/transform.h"
#include <iostream>

__global__ void add_kernel(int *a, int *b, int size)
{
    if(threadIdx.x < size){
        b[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
    }
}


void add_with_cuda(){
    int a[5] = {1,2,3,4,5};
    int b[5] = {1,2,3,4,5};


    int *device_a;
    int *device_b;
    int *device_c;

    cudaMalloc((void**)&device_a, 5*sizeof(int));
    cudaMalloc((void**)&device_b, 5*sizeof(int));
    cudaMemcpy(device_a, a, 5*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, 5*sizeof(int), cudaMemcpyHostToDevice);
    add_kernel<<<1,100>>>(device_a, device_b, 5);
    cudaMemcpy(a, device_a, 5*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, device_b, 5*sizeof(int), cudaMemcpyDeviceToHost);

    for(int count = 0; count < 5; count++){
        std::cout<<b[count]<<" ";
    }
    cudaFree(device_a);
    cudaFree(device_b);
}