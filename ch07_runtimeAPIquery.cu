#include "./shared_lib.cu"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include <cstdio>
#include <iostream>
#include <stdio.h>

template <typename T> void println(T value) { std::cout << value << "\n"; }

int main() {
  errorCheck(InitGPU(), __FILE__, __LINE__);

  cudaDeviceProp prop;
  errorCheck(cudaGetDeviceProperties(&prop, 0), __FILE__, __LINE__);

  println(prop.name);
  println(prop.major);
  println(prop.minor);
  println(prop.totalGlobalMem);
  println(prop.totalConstMem);
  // println(prop.maxGridSize[0]);
  //
  std::cout << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " "
            << prop.maxGridSize[2] << "\n";
  println(prop.multiProcessorCount);
  return 0;
}
