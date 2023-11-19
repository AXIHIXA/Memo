#include <random>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "util/CudaUtil.h"


namespace cuax
{

/// Monte-Carlo intergration routine inside unit circle.
/// Estimates value of PI.
__global__ 
void iint(
        const float * __restrict__ sample,
        int * __restrict__ mask,
        int len
)
{
    // Well, ncu says that row-majored indexing has better overall throughput.
    // ncu -k regex:iint ./cmake-build-release/cumo
    auto i = static_cast<int>(blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    // Some says that column-majored indexing utilizes cache better, hum.
    // auto i = static_cast<int>(blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x * blockDim.y + threadIdx.y);

    if (i < len)
    {
        float x = sample[2 * i];
        float y = sample[2 * i + 1];
        mask[i] = x * x + y * y <= 1.0f;
    }
}


class UniformFloat2
{
public:
    UniformFloat2() = delete;

    UniformFloat2(unsigned int seed, float xMin, float xMax, float yMin, float yMax)
            : e(seed), dx(xMin, xMax), dy(yMin, yMax)
    {
        // Nothing needed here. 
    }

    __host__ __device__ 
    float2 operator()(unsigned long long i)
    {
        e.discard(i);
        return {dx(e), dy(e)};
    }

private:
    thrust::default_random_engine e;
    thrust::uniform_real_distribution<float> dx;
    thrust::uniform_real_distribution<float> dy;
};


int test(int argc, char * argv[])
{
    static constexpr int kNumSamples = 50000000;
    static constexpr dim3 kBlockDim = {32U, 32U, 1U};
    static constexpr int kBlockSize = static_cast<int>(kBlockDim.x * kBlockDim.y * kBlockDim.z);
    
    unsigned int seed = std::random_device()();
    std::printf("seed = %u\n", seed);

    thrust::device_vector<float2> dSample(kNumSamples);
    
    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(kNumSamples),
        dSample.begin(),
        UniformFloat2(seed, -1.0f, 1.0f, -1.0f, 1.0f)
    );

    thrust::device_vector<int> dMask(kNumSamples, 0);

    dim3 iintGridDim {static_cast<unsigned int>(kNumSamples / kBlockSize) + 1U, 1U, 1U};

    iint<<<iintGridDim, kBlockDim>>>(
            reinterpret_cast<float *>(dSample.data().get()),
            dMask.data().get(),
            static_cast<int>(dMask.size())
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    int numInside = thrust::reduce(thrust::device, dMask.begin(), dMask.end());
    std::printf("Monte-Carlo PI = %lf\n", static_cast<double>(numInside) / static_cast<double>(kNumSamples) * 4.0);

    return EXIT_SUCCESS;
}

}  // namespace cuax