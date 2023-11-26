#include <fstream>
#include <iostream>

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
    auto idx = static_cast<int>(blockIdx.x * (1 * kBlockDim.x * kBlockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

//    if (idx + 2 * kBlockSize >= sampleLen)
//    {
//        printf("idx = %d, 3 * kBlockSize = %d, sampleLen = %d\n", idx, 3 * kBlockSize, sampleLen);
//    }

    if (idx + 1 * kBlockSize < sampleLen)
//    if (idx < sampleLen)
    {
        float2 r0 = sample[idx];
//        float2 r1 = sample[idx +     kBlockSize];
//        float2 r2 = sample[idx + 2 * kBlockSize];
//        float2 r3 = sample[idx + 3 * kBlockSize];

        #pragma unroll
        for (int ci = 0; ci != kCenterUnrollFactor; ++ci)
        {
            if (ci < centerLen)
            {
                float2 c = center[ci];

                res[sampleLenPadded * ci + idx                 ] =
                        (r0.x - c.x) * (r0.x - c.x) + (r0.y - c.y) * (r0.y - c.y);
//                res[sampleLenPadded * ci + idx +     kBlockSize] =
//                        (r1.x - c.x) * (r1.x - c.x) + (r1.y - c.y) * (r1.y - c.y);
//                res[sampleLenPadded * ci + idx + 2 * kBlockSize] =
//                        (r2.x - c.x) * (r2.x - c.x) + (r2.y - c.y) * (r2.y - c.y);
//                res[sampleLenPadded * ci + idx + 3 * kBlockSize] =
//                        (r3.x - c.x) * (r3.x - c.x) + (r3.y - c.y) * (r3.y - c.y);
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
    static constexpr int kNumCentersInit = 4096;

    int numSamples = kNumSamplesInit;
    int numCenters = kNumCentersInit;
    int numCentersLastBatch = numCenters % kCenterUnrollFactor;

    thrust::device_vector<float2> dSample(numSamples, {1.0f, 0.0f});
    thrust::device_vector<float2> dCenter(numCenters, {0.0f, 0.0f});
    dCenter[numCenters - 4] = {1.0f, 2.0f};
    dCenter[numCenters - 3] = {1.0f, 0.0f};
    dCenter[numCenters - 2] = {0.0f, 1.0f};
    dCenter[numCenters - 1] = {0.0f, 0.0f};

    // Pad numSamples to multiple of 32
    // (32 * sizeof(float) == 128, L1 cache line granularity)
    // for aligned & coalesced memory access pattern.
    auto [numSamplesPadded, remainder] = padTo32k(numSamples);
    // int numSamplesPadded = numSamples;  // Test for non-aligned pattern. Slower!
    thrust::device_vector<float> dBuffer(numSamplesPadded * kCenterUnrollFactor, 0.0f);

    dim3 mGridDim {(numSamples + 1 * kBlockSize - 1) / (1 * kBlockSize), 1U, 1U};

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

//            thrust::host_vector<float> hBuffer = dBuffer;
//
//            if (std::ofstream fout {"var/1.txt"})
//            {
//                for (int i = 0; i != hBuffer.size(); ++i)
//                {
//                    fout << i % numSamplesPadded << ' ' << i << ' ' << hBuffer[i] << '\n';
//                }
//            }
//            else
//            {
//                throw std::runtime_error("");
//            }

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
