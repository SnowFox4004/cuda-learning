#include <stdio.h>

__global__ void hello()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    printf("Hello from %d-%d\n", bid, tid);
}

int main()
{
    printf("Hello From CPU\n");
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}