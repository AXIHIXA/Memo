#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


constexpr dim3 kBlockDim {32U, 32U, 1U};

constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};

constexpr int kCenterUnrollFactor = 32;


__constant__
float centerX[kCenterUnrollFactor];


__constant__
float centerY[kCenterUnrollFactor];


__global__
void f2(
        const float * __restrict__ sampleX,
        const float * __restrict__ sampleY,
        int sampleLen,
        int sampleLenPadded,
        int centerLen,
        float * __restrict__ res
)
{
    auto idx = static_cast<int>(blockIdx.x * (kBlockDim.x * kBlockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (idx < sampleLen)
    {
        float2 r = {sampleX[idx], sampleY[idx]};

        #pragma unroll
        for (int ci = 0; ci != kCenterUnrollFactor; ++ci)
        {
            if (ci < centerLen)
            {
                float2 c = {centerX[ci], centerY[ci]};
                float dx = r.x - c.x;
                float dy = r.y - c.y;
                res[sampleLenPadded * ci + idx] = dx * dx + dy * dy;
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

    static constexpr int kNumSamplesInit = 20480010;
    static constexpr int kNumCentersInit = 32;

    int numSamples = kNumSamplesInit;
    int numCenters = kNumCentersInit;
    int numCentersLastBatch = numCenters % kCenterUnrollFactor;

    thrust::device_vector<float> dSampleX(numSamples, 1.0f);
    thrust::device_vector<float> dSampleY(numSamples, 0.0f);
    thrust::device_vector<float> dCenterX(numCenters, 0.0f);
    thrust::device_vector<float> dCenterY(numCenters, 0.0f);
    dCenterX[numCenters - 4] = 1.0f; dCenterY[numCenters - 4] = 0.0f;
    dCenterX[numCenters - 3] = 1.0f; dCenterY[numCenters - 3] = 0.0f;
    dCenterX[numCenters - 2] = 0.0f; dCenterY[numCenters - 2] = 1.0f;
    dCenterX[numCenters - 1] = 1.0f; dCenterY[numCenters - 1] = 0.0f;

    // Pad numSamples to multiple of 32
    // (32 * sizeof(float) == 128, L1 cache line granularity)
    // for aligned & coalesced memory access pattern.
    auto [numSamplesPadded, remainder] = padTo32k(numSamples);
    // int numSamplesPadded = numSamples;  // Test for non-aligned pattern. Slower!
    thrust::device_vector<float> dBuffer(numSamplesPadded * kCenterUnrollFactor, 0.0f);

    dim3 mGridDim {(numSamples + kBlockSize - 1) / kBlockSize, 1U, 1U};

    // Test
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

//    // We are not using shared memory, so max L1.
//    cudaFuncSetAttribute(
//            f2,
//            cudaFuncAttributePreferredSharedMemoryCarveout,
//            cudaSharedmemCarveoutMaxL1
//    );

    for (int _ = 0; _ != numDuplication; ++_)
    {
        for (int ci = 0; ci < numCenters; ci += kCenterUnrollFactor)
        {
            int numCentersThisBatch =
                    (numCentersLastBatch != 0 and numCenters <= ci + kCenterUnrollFactor) ?
                    numCentersLastBatch :
                    kCenterUnrollFactor;

            cudaMemcpyToSymbolAsync(
                    centerX,
                    dCenterX.data().get() + ci,
                    numCentersThisBatch * sizeof(float),
                    0UL,
                    cudaMemcpyDeviceToDevice
            );
            cudaMemcpyToSymbolAsync(
                    centerY,
                    dCenterY.data().get() + ci,
                    numCentersThisBatch * sizeof(float),
                    0UL,
                    cudaMemcpyDeviceToDevice
            );
            cudaDeviceSynchronize();

            f2<<<mGridDim, kBlockDim>>>(
                    dSampleX.data().get(),
                    dSampleY.data().get(),
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

    std::cout << "Structure of Array "
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
