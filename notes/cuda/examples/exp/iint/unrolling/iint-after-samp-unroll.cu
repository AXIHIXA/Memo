#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


constexpr dim3 kBlockDim {32U, 32U, 1U};

constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};

constexpr int kCenterUnrollFactor = 32;


__constant__
float2 center[kCenterUnrollFactor];


__global__
void f2(
        const float2 * __restrict__ sample,
        int sampleLen,
        int sampleLenPadded,
        int centerLen,
        float * __restrict__ res
)
{
    auto idx = static_cast<int>(blockIdx.x * (4 * kBlockDim.x * kBlockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (idx + 3 * kBlockDim.x * kBlockDim.y < sampleLen)
    {
        float2 r0 = sample[idx];
        float2 r1 = sample[idx +     kBlockDim.x * kBlockDim.y];
        float2 r2 = sample[idx + 2 * kBlockDim.x * kBlockDim.y];
        float2 r3 = sample[idx + 3 * kBlockDim.x * kBlockDim.y];

        auto resIdx = static_cast<int>(blockIdx.x * kBlockDim.x * kBlockDim.y + threadIdx.y * blockDim.x + threadIdx.x);

        #pragma unroll
        for (int ci = 0; ci != kCenterUnrollFactor; ++ci)
        {
            if (ci < centerLen)
            {
                float2 c = center[ci];
                float nr0c = (r0.x - c.x) * (r0.x - c.x) + (r0.y - c.y) * (r0.y - c.y);
                float nr1c = (r1.x - c.x) * (r1.x - c.x) + (r1.y - c.y) * (r1.y - c.y);
                float nr2c = (r2.x - c.x) * (r2.x - c.x) + (r2.y - c.y) * (r2.y - c.y);
                float nr3c = (r3.x - c.x) * (r3.x - c.x) + (r3.y - c.y) * (r3.y - c.y);
                res[sampleLenPadded * ci + resIdx] = nr0c + nr1c + nr2c + nr3c;
            }
        }
    }
}


std::pair<int, int> padTo32k(int a)
{
    static constexpr int k32 = 32;

    if (int b = a % k32; b == 0)
    {
        return {a, 0};
    }
    else
    {
        return {a + k32 - b, b};
    }
}


int main(int argc, char * argv[])
{
    int numDuplication = (argc == 2) ? std::stoi(argv[1]) : 1;

    static constexpr int kNumSamplesInit = 10240000;
    static constexpr int kNumCentersInit = 8196;

    int numSamples = kNumSamplesInit;
    int numCenters = kNumCentersInit;
    int numCentersLastBatch = numCenters % kCenterUnrollFactor;

    thrust::device_vector<float2> dSample(numSamples, {1.0f, 0.0f});
    thrust::device_vector<float2> dCenter(numCenters, {0.0f, 0.0f});
    dCenter[numCenters - 4] = {1.0f, 0.0f};
    dCenter[numCenters - 3] = {1.0f, 1.0f};
    dCenter[numCenters - 2] = {0.0f, 1.0f};
    dCenter[numCenters - 1] = {1.0f, 0.0f};

    // Pad numSamples to multiple of 32
    // (32 * sizeof(float) == 128, L1 cache line granularity)
    // for aligned & coalesced memory access pattern.
    auto [numSamplesPadded, remainder] = padTo32k(numSamples >> 2U);
    // int numSamplesPadded = numSamples;  // Test for non-aligned pattern. Slower!
    thrust::device_vector<float> dBuffer(numSamplesPadded * kCenterUnrollFactor, 0.0f);

    dim3 mGridDim {(numSamples + 2 * kBlockSize - 1) / (2 * kBlockSize), 1U, 1U};

    // Test
    auto ss = std::chrono::high_resolution_clock::now();

    thrust::device_vector<int> dBeginOffset(numCenters);
    thrust::device_vector<int> dEndOffset(numCenters);
    thrust::device_vector<float> dResult(numCenters);
    thrust::sequence(thrust::device, dBeginOffset.begin(), dBeginOffset.end(), 0, numSamplesPadded);
    thrust::sequence(thrust::device, dEndOffset.begin(), dEndOffset.end(), numSamples >> 2U, numSamplesPadded);

    std::size_t tempStorageBytes;
    cub::DeviceSegmentedReduce::Sum(
            nullptr,
            tempStorageBytes,
            dBuffer.data().get(),
            dResult.data().get(),
            numCenters,
            dBeginOffset.data().get(),
            dEndOffset.data().get()
    );
    cudaDeviceSynchronize();
    thrust::device_vector<unsigned char> dTempStorage(tempStorageBytes);

    cudaMemsetAsync(dBuffer.data().get(), 0, sizeof(float) * dBuffer.size(), nullptr);

    for (int _ = 0; _ != numDuplication; ++_)
    {
        for (int ci = 0; ci < numCenters; ci += kCenterUnrollFactor)
        {
            int numCentersThisBatch =
                    (numCentersLastBatch != 0 and numCenters <= ci + kCenterUnrollFactor) ?
                    numCentersLastBatch :
                    kCenterUnrollFactor;

            cudaMemcpyToSymbolAsync(
                    center,
                    dCenter.data().get() + ci,
                    numCentersThisBatch * sizeof(float2),
                    0UL,
                    cudaMemcpyDeviceToDevice,
                    nullptr
            );

            f2<<<mGridDim, kBlockDim, 0U, nullptr>>>(
                    dSample.data().get(),
                    numSamples,
                    numSamplesPadded,
                    numCentersThisBatch,
                    dBuffer.data().get()
            );

            cub::DeviceSegmentedReduce::Sum(
                    dTempStorage.data().get(),
                    tempStorageBytes,
                    dBuffer.data().get(),
                    dResult.data().get() + ci,
                    numCentersThisBatch,
                    dBeginOffset.data().get(),
                    dEndOffset.data().get(),
                    nullptr
            );

            cudaMemsetAsync(dBuffer.data().get(), 0, sizeof(float) * dBuffer.size(), nullptr);
        }
    }

    cudaDeviceSynchronize();
    auto tic = (std::chrono::high_resolution_clock::now() - ss).count();

    std::cout << "After Unrolling "
              << static_cast<double>(tic) * 1e-6 / static_cast<double>(numDuplication)
              << " ms\n\n";

    if (numDuplication == 1)
    {
        thrust::host_vector<float> hRes = dResult;

        for (int i = 0; i != numCenters; ++i)
        {
            std::cout << "sum @ center[" << i << "] = " << hRes[i] << '\n';
        }
    }

    return EXIT_SUCCESS;
}


/*
$ ./cmake-build-release/exe 40

After Unrolling 587.185 ms
After Unrolling 551.579 ms
*/
