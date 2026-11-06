#include <iostream>
#include <stdio.h>

// L1 cache 和 共享内存 共用一个总大小固定的mem
// 可以配置各自的大小但是似乎不一定会应用
// 共享 + 缓存 = 统一缓存
// compile arg: -dlcm=ca 强制打开L1缓存来缓存globalMem加载的数据
int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::cout << "Is Global L1 Cache Supported: " << prop.globalL1CacheSupported
            << std::endl;
  std::cout << "L2 cache size: " << prop.l2CacheSize / (1024 * 1024) << " M"
            << std::endl;

  return 0;
}
