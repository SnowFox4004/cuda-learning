#include "shared_lib.cuh"
#include <stdio.h>
#define CHECK(x) errorCheck(x, __FILE__, __LINE__);

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

cudaError_t InitGPU() {
  int iDeviceCnt;
  auto error = cudaGetDeviceCount(&iDeviceCnt);
  if (error != cudaSuccess || iDeviceCnt == 0) {
    printf("No CUDA device!\n");
    exit(-1);
  } else {
    printf("Found %d CUDA device\n", iDeviceCnt);
  }

  int iDev = 0;
  error = cudaSetDevice(iDev);
  if (error != cudaSuccess) {
    printf("Fail to select device!\n");
  } else {
    printf("Set %d for computing\n", iDev);
  }

  return error;
}
