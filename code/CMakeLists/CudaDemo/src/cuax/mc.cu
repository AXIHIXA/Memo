#include <random>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>


namespace cuax
{

/// Monte-Carlo intergration routine inside unit circle.
/// Estimates value of PI.
__global__ 
void iint(
        float * __restrict__ sample,
        int * __restrict__ mask,
        int len
)
{
    // Well, ncu says that row-majored indexing has better overall throughput.
    // ncu -k regex:iint ./cmake-build-release/cumo
    int i = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    // Some says that column-majored indexing utilizes cache better, hum.
    // int i = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x * blockDim.y + threadIdx.y;

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
    constexpr unsigned long long kNumSamples = 50000000ULL;
    
    unsigned int seed = std::random_device()();
    printf("seed = %u\n", seed);

    thrust::device_vector<float2> dPt(kNumSamples);
    thrust::device_vector<int> dInside(kNumSamples, 0);
    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator(0ULL),
        thrust::make_counting_iterator(kNumSamples),
        dPt.begin(),
        UniformFloat2(seed, -1.0f, 1.0f, -1.0f, 1.0f)
    );

    dim3 blockDim {32, 32, 1};
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
    unsigned int numGrids = kNumSamples / blockSize + 1;
    dim3 gridDim {numGrids, 1, 1};

    iint<<<gridDim, blockDim>>>(reinterpret_cast<float *>(dPt.data().get()), dInside.data().get(), dInside.size());
    cudaDeviceSynchronize();

    int numInside = thrust::reduce(thrust::device, dInside.begin(), dInside.end());
    printf("Monte-Carlo PI = %lf\n", static_cast<double>(numInside) / static_cast<double>(kNumSamples) * 4.0);

    return EXIT_SUCCESS;
}

}  // namespace cuax