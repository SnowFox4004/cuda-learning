// #include <__clang_cuda_runtime_wrapper.h>
#include "./shared_lib.cu"
#include "cuda_runtime_api.h"
#include "driver_types.h"
// #include <__clang_cuda_runtime_wrapper.h>
// #include <__clang_cuda_runtime_wrapper.h>

// #include <__clang_cuda_runtime_wrapper.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <stdio.h>
// 不能写长度
extern __shared__ float dyn_array[];

__global__ void kernel2(float *a, const int n) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int id = bid * blockDim.x + tid;

  // kernel2 使用动态共享内存
  if (id < n) {
    dyn_array[tid] = -a[id];
  }

  __syncthreads();

  if (tid == 0) {
    // 只打印一次
    for (int i = 0; i < 32; i++) {
      printf("dyn_array[%2d] = %.3f\t blockIdx: %d\n", i, dyn_array[i], bid);
    }
  }
}

__global__ void kernel(float *a, const int n) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int id = bid * blockDim.x + tid;

  // 每个线程块共享相同的 s_array 变量
  // 故一般大小设为线程数量
  //
  // 但是这里是静态的，需要固定长度
  // 动态的在上面
  __shared__ float s_array[256];
  if (id < n) {
    s_array[tid] = a[id];
  }

  __syncthreads();

  if (tid == 0) {
    // 只打印一次
    for (int i = 0; i < 32; i++) {
      printf("s_array[%2d] = %.3f\t blockIdx: %d\n", i, s_array[i], bid);
    }
  }
}

int main() {
  int devId = 0;
  cudaDeviceProp devProp;
  CHECK(cudaGetDeviceProperties(&devProp, devId));

  int nElem = 512;
  float *host_a = (float *)malloc(sizeof(float) * nElem);
  for (int i = 0; i < nElem; i++) {
    host_a[i] = (float)i;
  }

  float *dev_a;
  CHECK(cudaMalloc((float **)&dev_a, sizeof(float) * nElem));
  CHECK(
      cudaMemcpy(dev_a, host_a, sizeof(float) * nElem, cudaMemcpyHostToDevice));

  dim3 griddim(2);
  dim3 blockdim(nElem / 2);

  kernel<<<griddim, blockdim>>>(dev_a, nElem);

  CHECK(cudaDeviceSynchronize());

  // 这里第三个参数是sharedMem的字节数意思是跟线程块大小一样
  kernel2<<<griddim, blockdim, nElem * sizeof(float)>>>(dev_a, nElem);
  CHECK(cudaDeviceSynchronize());

  return 0;
}
