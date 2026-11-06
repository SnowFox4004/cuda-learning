#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include <cstdio>
#include <stdio.h>

void InitGPU() {
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
}

void initdata(float *addr, int size) {
  for (int i = 0; i < size; i++) {
    addr[i] = (float)(rand() & 0xff) / 10.f;
  }
  return;
}

__device__ float Add(float a, float b) { return a + b; }

__global__ void arrayAdd(float *a, float *b, float *c, int size) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const int ID = bid * blockDim.x + tid;
  if (ID > size)
    return;

  c[ID] = Add(a[ID], b[ID]);
}

void cpuAdd(float *a, float *b, float *c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  //    cudaEvent_t start, end;

  InitGPU();
  int elemSize = 512;
  size_t arraySize = elemSize * sizeof(float);

  float *fp_a, *fp_b, *fp_c;
  fp_a = (float *)malloc(arraySize);
  fp_b = (float *)malloc(arraySize);
  fp_c = (float *)malloc(arraySize);
  if (fp_a && fp_b && fp_c) {
    memset(fp_a, 0, arraySize);
    memset(fp_b, 0, arraySize);
    memset(fp_c, 0, arraySize);
  } else {
    printf("Failed malloc\n");
    exit(-1);
  }

  // 神秘双重指针
  float *fpdev_a, *fpdev_b, *fpdev_c;
  cudaMalloc((float **)&fpdev_a, arraySize);
  cudaMalloc((float **)&fpdev_b, arraySize);
  cudaMalloc((float **)&fpdev_c, arraySize);
  if (fpdev_a && fpdev_b && fpdev_c) {
    cudaMemset(fpdev_a, 0, arraySize);
    cudaMemset(fpdev_b, 0, arraySize);
    cudaMemset(fpdev_c, 0, arraySize);
  } else {
    printf("Failed gpu malloc\n");
    exit(-1);
  }

  // 随机初始化AB
  srand(42);
  initdata(fp_a, elemSize);
  initdata(fp_b, elemSize);

  cudaMemcpy(fpdev_a, fp_a, arraySize, cudaMemcpyHostToDevice);
  cudaMemcpy(fpdev_b, fp_b, arraySize, cudaMemcpyHostToDevice);
  cudaMemcpy(fpdev_c, fp_c, arraySize, cudaMemcpyHostToDevice);

  // 计时来
  float total_time = 0;

  for (int repeat = 0; repeat < 10; repeat++) {
    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);

    cudaEventRecord(begin);
    cudaEventQuery(begin);

    // 直接一一对应全并行了
    dim3 block_dim(32);
    dim3 grid_dim(elemSize / 32);
    arrayAdd<<<grid_dim, block_dim>>>(fpdev_a, fpdev_b, fpdev_c, elemSize);

    // cpuAdd(fp_a, fp_b, fp_c, elemSize);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float elasped_time;
    cudaEventElapsedTime(&elasped_time, begin, stop);
    ;
    if (repeat != 0)
      total_time += elasped_time;
    else
      printf("First Elasped Time: %.3f ms\n", elasped_time);
  }
  printf("总耗时 %.3f ms\n", total_time);

  cudaMemcpy(fp_c, fpdev_c, arraySize, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++) {
    printf("%.2f\t+ %.2f\t= %.2f\t; is_correct = %d\n", fp_a[i], fp_b[i],
           fp_c[i], (fp_c[i] == fp_a[i] + fp_b[i]));
  }
  free(fp_a);
  free(fp_b);
  free(fp_c);

  cudaFree(fp_a);
  cudaFree(fp_b);
  cudaFree(fp_c);

  //?
  cudaDeviceReset();

  return 0;
}
