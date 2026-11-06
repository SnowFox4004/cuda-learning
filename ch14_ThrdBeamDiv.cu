/*
线程束分化
指的是一个线程束内 32个 thread 遇到分支语句的时候，执行了不同的命令
就会互相等待，导致并行化不完全
尽可能让一个线程块内所有线程执行相同的操作
就可以避免分化
例如这个求和
如果是 2**x 号线程工作，就会出现 0 2 4 6 8 -> if; 1 3 5 7 9->else
这一个块内要互相等待

而如果是 前x个负责工作 那么最多就是在交界处有一个块出现分化
前面的块全部都要工作 -> if
后面的块超范围不工作 -> else
就可以使得一个线程束快速并行执行完毕
然后执行下一个线程束
就会快一些
*/

#include "./shared_lib.cu"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

class Timer
{
public:
    Timer() {}
    ~Timer() {}

    void c_start()
    {
        cstart = std::chrono::steady_clock::now();
    }
    double c_stop()
    {
        /*
            return: duration (us)
        */
        cend = std::chrono::steady_clock::now();
        auto duration = cend - cstart;

        std::cout << duration.count() / 1000.f / 1000.f << " ms on cpu\n";
        return duration.count();
    }

    void g_start()
    {
        cudaEventCreate(&gstart);
        cudaEventCreate(&gend);

        cudaEventRecord(gstart);
        cudaEventQuery(gstart);
    }

    double g_stop()
    {
        cudaEventRecord(gend);
        cudaDeviceSynchronize();
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, gstart, gend);

        std::cout << elapsed_time << " ms on GPU"
                  << std::endl;

        return elapsed_time;
    }

private:
    cudaEvent_t gstart, gend;
    std::chrono::time_point<std::chrono::steady_clock> cstart, cend;
};

__global__ void ReduceWithDivergence(float *data, float *out_data, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int identity = bid * blockDim.x + tid;

    auto idata = data + bid * blockDim.x;

    if (identity >= n)
        return;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (2 * stride) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        out_data[blockIdx.x] = idata[0];
    }
}

__global__ void ReduceWithoutDivergence(float *data, float *out_data, int n)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = bid * blockDim.x + tid;

    if (id >= n)
        return;

    float *idata = data + bid * blockDim.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out_data[bid] = idata[0];
    }
}
int main()
{
    int nblock = 1024;
    cudaDeviceProp devprop;
    CHECK(cudaGetDeviceProperties(&devprop, 0));
    std::cout << "Running on: " << devprop.name << std::endl;

    int nelem = 1048576 / 2;

    float *host_a = (float *)malloc(nelem * sizeof(float));

    float *dev_a;
    CHECK(cudaMalloc(&dev_a, nelem * sizeof(float)));

    float *data_out;
    CHECK(cudaMalloc(&data_out, nblock * sizeof(float)));

    // gen
    for (int i = 0; i < nelem; i++)
    {
        host_a[i] = rand() % 0xfff;
    }

    {
        // warmup
        // drop result
        CHECK(cudaMemcpy(dev_a, host_a, nelem * sizeof(float), cudaMemcpyHostToDevice));

        dim3 grid_size(nblock);
        dim3 block_size(nelem / nblock);

        // Timer t_gpu_0;
        // t_gpu_0.g_start();
        ReduceWithDivergence<<<grid_size, block_size>>>(dev_a, data_out, nelem);
        cudaDeviceSynchronize();
        // t_gpu_0.g_stop();

        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(dev_a, host_a, nelem * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid_size(nblock);
    dim3 block_size(nelem / nblock);

    Timer t_gpu_0;
    t_gpu_0.g_start();
    ReduceWithDivergence<<<grid_size, block_size>>>(dev_a, data_out, nelem);
    cudaDeviceSynchronize();
    t_gpu_0.g_stop();
    printf("With Thread Beam Divergence\n\n");
    CHECK(cudaGetLastError());

    float *result = (float *)malloc(nblock * sizeof(float));
    CHECK(cudaMemcpy(result, data_out, sizeof(float) * nblock, cudaMemcpyDeviceToHost));

    float answer_gpu_0 = 0;
    for (int i = 0; i < nblock; i++)
    {
        answer_gpu_0 += result[i];
    }

    {
        // warmup
        // drop result
        CHECK(cudaMemcpy(dev_a, host_a, nelem * sizeof(float), cudaMemcpyHostToDevice));

        dim3 grid_size(nblock);
        dim3 block_size(nelem / nblock);

        // Timer t_gpu_0;
        // t_gpu_0.g_start();
        ReduceWithoutDivergence<<<grid_size, block_size>>>(dev_a, data_out, nelem);
        cudaDeviceSynchronize();
        // t_gpu_0.g_stop();

        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(dev_a, host_a, nelem * sizeof(float), cudaMemcpyHostToDevice));

    // dim3 grid_size(nblock);
    // dim3 block_size(nelem / nblock);

    Timer t_gpu_1;
    t_gpu_1.g_start();
    ReduceWithoutDivergence<<<grid_size, block_size>>>(dev_a, data_out, nelem);
    cudaDeviceSynchronize();
    t_gpu_1.g_stop();
    CHECK(cudaGetLastError());
    std::cout << "Without Divergence\n\n";

    CHECK(cudaMemcpy(result, data_out, nblock * sizeof(float), cudaMemcpyDeviceToHost));

    float answer_gpu_1 = 0;
    for (int i = 0; i < nblock; i++)
    {
        answer_gpu_1 += result[i];
    }

    Timer t_cpu;
    t_cpu.c_start();
    float answer_cpu = 0;
    for (int i = 0; i < nelem; i++)
    {
        answer_cpu += host_a[i];
    }
    t_cpu.c_stop();

    std::cout << std::fixed << std::setprecision(2) << "Answer From GPU With Divergence:    " << answer_gpu_0 << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Answer From GPU Without Divergence: " << answer_gpu_1 << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Answer Form CPU:                    " << answer_cpu << std::endl;
    return 0;
}
