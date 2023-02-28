#include <iostream>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>

#include "cuax/CudaMatrix.h"


__global__ void cudaMatrixAdd(const float * A, const float * B, float * C, std::size_t kNX, std::size_t kNY)
{
    std::size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t iy = threadIdx.y + blockIdx.y * blockDim.y;
    std::size_t idx = iy * kNY + ix;

    if (ix < kNX && iy < kNY)
    {
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void cudaDebugBuiltIn()
{
    std::printf("block %u %u %u thread %u %u %u\n",
                blockIdx.x, blockIdx.y, blockIdx.z,
                threadIdx.x, threadIdx.y, threadIdx.z);
}


void cpuMatrixAdd(const float * A, const float * B, float * C, std::size_t kNX, std::size_t kNY)
{
    for (std::size_t idx = 0; idx != kNX * kNY; ++idx)
    {
        C[idx] = A[idx] + B[idx];
    }
}


double test()
{
    // Generate random data serially.
    thrust::default_random_engine rng(1337);
    thrust::uniform_real_distribution<double> dist(-50.0, 50.0);
    thrust::host_vector<double> h_vec(32 << 20);
    thrust::generate(h_vec.begin(), h_vec.end(), [&]
    {
        return dist(rng);
    });

    // Transfer to device and compute the sum.
    thrust::device_vector<double> d_vec = h_vec;
    return thrust::reduce(d_vec.begin(), d_vec.end(), 0.0, thrust::plus<int>());
}