#include <iostream>
#include <stdio.h>

using namespace std;

int main() {
  int devid = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  cout << prop.name << endl;
  cout << "SM count:    " << prop.multiProcessorCount << endl;
  cout << "L2 size:     " << prop.l2CacheSize / (1024 * 1024) << " M" << endl;
  cout << "max thread:  " << prop.maxThreadsPerMultiProcessor << endl;
  cout << "priori strm: " << prop.streamPrioritiesSupported << endl;
  cout << "global l1:   " << prop.globalL1CacheSupported << endl;
  cout << "local l1:    " << prop.localL1CacheSupported << endl;
  cout << "shrd mem max:" << prop.sharedMemPerMultiprocessor / (1024) << " KB"
       << endl;
  cout << "mreg per sm: " << prop.regsPerMultiprocessor / 1024 << " K" << endl;
  cout << "mblk per sm: " << prop.maxBlocksPerMultiProcessor << endl;
  cout << "mem bndwth:  " << prop.memoryBusWidth << endl;
  //  cout << "mem freq:    " << prop. << endl;
  return 0;
}
