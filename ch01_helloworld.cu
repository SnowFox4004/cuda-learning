#include <stdio.h>

__global__ void hello_from_gpu() { printf("Hello World Form GPU\n"); }

int main(void) {
  //              线程块数  每个线程块的线程数
  hello_from_gpu<<<1, 1>>>();

  // 异步核函数 这里等待执行完成 本质await
  cudaDeviceSynchronize();
  if (1 == 0) {
    return 1;
  }
  // std::cout<<"Dick\n";
  printf("Dick\n");

  return 0;
}
