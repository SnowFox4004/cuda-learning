#include "cuda_device_runtime_api.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include <cstdio>
#include <stdio.h>

cudaError_t errorCheck(cudaError_t error_code, const char *filename,
                       int lineNo) {
  if (error_code != cudaSuccess) {
    printf(
        "CUDA "
        "Error:\n\rcode=%d\n\rname=%s\n\rdescription=%s\n\rfile=%s\n\rline=%d"
        "\n\r",
        error_code, cudaGetErrorName(error_code),
        cudaGetErrorString(error_code), filename, lineNo);
    return error_code;
  }
  return error_code;
}

__global__ void errorKernelFunc() { const int bid = blockIdx.x; }

int main() {
  float *fphost_a;
  fphost_a = (float *)malloc(4);

  float *fpdev_a;
  errorCheck(cudaMalloc((float **)&fpdev_a, 4), __FILE__, __LINE__);
  cudaMemset(fpdev_a, 0, 4);

  errorCheck(cudaMemcpy(fpdev_a, fphost_a, 4, cudaMemcpyHostToDevice), __FILE__,
             __LINE__);

  free(fphost_a);
  errorCheck(cudaFree(fpdev_a), __FILE__, __LINE__);

  errorCheck(cudaDeviceReset(), __FILE__, __LINE__);

  dim3 blocksize = 2048;

  // 核函数异步 所以要手动获取 LastError
  errorKernelFunc<<<1, blocksize>>>();
  errorCheck(cudaGetLastError(), __FILE__, __LINE__);

  printf("Done\n");

  return 0;
}
