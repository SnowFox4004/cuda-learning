// #include <__clang_cuda_runtime_wrapper.h>
#include "./shared_lib.cu"
#include <stdio.h>

__constant__ float PI = 3.14;
__constant__ float scale_factor;

__global__ void kernel(void) {
  printf("Constant data: %f\n", PI);
  printf("scale_factor: %.3f\n", scale_factor);
}

int main() {
  float host_data = 8.8f;
  CHECK(cudaMemcpyToSymbol(scale_factor, &host_data, sizeof(float)));

  dim3 block(1);
  dim3 grid(1);

  kernel<<<grid, block>>>();
  CHECK(cudaDeviceSynchronize());

  // 可以从host复制一个东西去设备的常量内存
  CHECK(cudaMemcpyFromSymbol(&host_data, PI, sizeof(float)));

  printf("Got Data host_data = %.2f\n", host_data);

  CHECK(cudaDeviceReset());

  return 0;
}
