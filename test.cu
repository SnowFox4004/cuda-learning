#include <stdio.h>
// #include <shared_lib.cuh>
#include "./shared_lib.cu"

__global__ void hello()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    printf("Hello from %d-%d\n", bid, tid);
}

int main()
{
    int devId = 0;
    CHECK(cudaSetDevice(devId));

    printf("Hello From CPU\n");
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}