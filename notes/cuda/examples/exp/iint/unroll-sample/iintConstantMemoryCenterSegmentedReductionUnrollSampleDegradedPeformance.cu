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
    auto idx = static_cast<int>(blockIdx.x * (kBlockDim.x * kBlockDim.y << 2U) + threadIdx.y * blockDim.x + threadIdx.x);

    if (idx < sampleLen)
    {
        float2 r = sample[idx];

        float2 r2;
        if (idx + kBlockSize < sampleLen)
            r2 = sample[idx + kBlockSize];
        float2 r3;
        if (idx + (kBlockSize << 1U) < sampleLen)
            r3 = sample[idx + (kBlockSize << 1U)];
        float2 r4;
        if (idx + (kBlockSize * 3) < sampleLen)
            r4 = sample[idx + (kBlockSize * 3)];

        #pragma unroll
        for (int ci = 0; ci != kCenterUnrollFactor; ++ci)
        {
            if (ci < centerLen)
            {
                float2 c = center[ci];
                res[sampleLenPadded * ci + idx] = (r.x - c.x) * (r.x - c.x) + (r.y - c.y) * (r.y - c.y);

                if (idx + kBlockSize < sampleLen)
                    res[sampleLenPadded * ci + idx + kBlockSize] =
                            (r2.x - c.x) * (r2.x - c.x) + (r2.y - c.y) * (r2.y - c.y);
                if (idx + (kBlockSize << 1U) < sampleLen)
                    res[sampleLenPadded * ci + idx + (kBlockSize << 1U)] =
                            (r3.x - c.x) * (r3.x - c.x) + (r3.y - c.y) * (r3.y - c.y);
                if (idx + (kBlockSize * 3) < sampleLen)
                    res[sampleLenPadded * ci + idx + (kBlockSize * 3)] =
                            (r4.x - c.x) * (r4.x - c.x) + (r4.y - c.y) * (r4.y - c.y);
            }
        }
    }
}


std::pair<int, int> padTo128k(int a)
{
    static constexpr int k128 = 128;

    if (int b = a % k128; b == 0)
    {
        return {a, 0};
    }
    else
    {
        return {a + k128 - b, b};
    }
}


int main(int argc, char * argv[])
{
    int numDuplication = (argc == 2) ? std::stoi(argv[1]) : 1;

    static constexpr int kNumSamplesInit = 10240010;
    static constexpr int kNumCentersInit = 8196;

    int numSamples = kNumSamplesInit;
    int numCenters = kNumCentersInit;
    int numCentersLastBatch = numCenters % kCenterUnrollFactor;

    thrust::device_vector<float2> dSample(numSamples, {1.0f, 0.0f});
    thrust::device_vector<float2> dCenter(numCenters, {0.0f, 0.0f});
    dCenter[numCenters - 4] = {1.0f, 0.0f};
    dCenter[numCenters - 3] = {1.0f, 0.0f};
    dCenter[numCenters - 2] = {0.0f, 1.0f};
    dCenter[numCenters - 1] = {1.0f, 0.0f};

    // Pad numSamples to multiple of 128 for aligned & coalesced memory access pattern.
    auto [numSamplesPadded, remainder] = padTo128k(numSamples);
    thrust::device_vector<float> dBuffer(numSamplesPadded * kCenterUnrollFactor, 0.0f);

    dim3 mGridDim {(numSamples + (kBlockSize << 2U) - 1) / (kBlockSize << 2U), 1U, 1U};

    // Warmup
    if (1 < numDuplication)
    {
        f2<<<mGridDim, kBlockDim>>>(
                dSample.data().get(),
                numSamples,
                numSamplesPadded,
                numCenters,
                dBuffer.data().get()
        );
        cudaDeviceSynchronize();
    }

    // atomicAdd
    // Dump global array and cub::DeviceReduce::Sum
    auto ss = std::chrono::high_resolution_clock::now();

    thrust::device_vector<int> dBeginOffset(numCenters);
    thrust::device_vector<int> dEndOffset(numCenters);
    thrust::device_vector<float> dResult(numCenters);
    thrust::sequence(thrust::device, dBeginOffset.begin(), dBeginOffset.end(), 0, numSamplesPadded);
    thrust::sequence(thrust::device, dEndOffset.begin(), dEndOffset.end(), numSamples, numSamplesPadded);

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

    for (int _ = 0; _ != numDuplication; ++_)
    {
        for (int ci = 0; ci < numCenters; ci += kCenterUnrollFactor)
        {
            int numCentersThisBatch =
                    (numCentersLastBatch != 0 and numCenters < ci + kCenterUnrollFactor) ?
                    numCentersLastBatch :
                    kCenterUnrollFactor;

            cudaMemcpyToSymbol(
                    center,
                    dCenter.data().get() + ci,
                    numCentersThisBatch * sizeof(float2),
                    0UL,
                    cudaMemcpyDeviceToDevice
            );

            f2<<<mGridDim, kBlockDim>>>(
                    dSample.data().get(),
                    numSamples,
                    numSamplesPadded,
                    numCentersThisBatch,
                    dBuffer.data().get()
            );
            cudaDeviceSynchronize();

            cub::DeviceSegmentedReduce::Sum(
                    dTempStorage.data().get(),
                    tempStorageBytes,
                    dBuffer.data().get(),
                    dResult.data().get() + ci,
                    numCentersThisBatch,
                    dBeginOffset.data().get(),
                    dEndOffset.data().get()
            );
            cudaDeviceSynchronize();
        }
    }

    auto tic = (std::chrono::high_resolution_clock::now() - ss).count();

    std::cout << "ConstantCenter + cub::DeviceSegmentedReduce::Sum "
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
$ ./cmake-build-release/exe 100
ConstantCenter + cub::DeviceSegmentedReduce::Sum 7.02152 ms

1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 0 2.048e+07 0
*/
