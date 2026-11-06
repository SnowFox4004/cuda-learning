#include "./shared_lib.cu"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "shared_lib.cuh"
// #include "vector_functions.hpp"
// #include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

// static global variable
//
__device__ int d_x = 1;
__device__ int d_y[2] = {1};

__global__ void kernel() {
  d_y[0] += d_x;
  d_y[1] += d_x;

  printf("d_x = %d\nd_y[0] = %d ; d_y[1] = %d\n", d_x, d_y[0], d_y[1]);
}

int main() {
  cudaDeviceProp devProp;
  errorCheck(cudaGetDeviceProperties(&devProp, 0), __FILE__, __LINE__);
  std::cout << "running on: " << devProp.name << "\n";

  // 可以传host的变量到device
  int h_y[2] = {10, 20};

  // 要用这个 ToSymbol 和 FromSymbol函数
  CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(h_y)));

  kernel<<<1, 1>>>();
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(h_y)));

  std::cout << h_y[0] << " " << h_y[1] << std::endl;
  CHECK(cudaDeviceReset());

  return 0;
}
