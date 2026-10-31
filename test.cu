#include <stdio.h>

__global__ void hello()
{
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;

    printf("Hello from %d-%d\n", bid, tid);
}

int main()
{
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}