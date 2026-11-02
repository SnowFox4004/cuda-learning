// #include "cuda_runtime_api.h"
// #include <__clang_cuda_runtime_wrapper.h>
// #include <__clang_cuda_builtin_vars.h>
#include <stdio.h>

__global__ void hello() {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  // 其实这些idx是三维的 有xyz
  // Grid dim
  // yz 方向最大 2^16-1

  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  printf("Hello from block %d thread %d, global id %d\n", bid, tid, id);
}

int main() {
  // could be
  dim3 grid_size(2, 2, 2);
  dim3 block_size(4, 4, 1);
  // blockDim
  // xy max=1024 ; z max=64
  // 但是 x+y+z max = 1024;
  hello<<<grid_size, block_size>>>();
  cudaDeviceSynchronize();

  return 0;
}
