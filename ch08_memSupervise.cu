/*
 * 并没有特别的代码
 * 编译的时候这样子写
 * 可以观察到内存占用
 * 但是这里的全局内存是静态global mem 用量 动态的看不到
 * nvcc --resource-usage ch04_matrixadd.cu -o .build/ch04_matrixadd -arch=sm_120
 * */
