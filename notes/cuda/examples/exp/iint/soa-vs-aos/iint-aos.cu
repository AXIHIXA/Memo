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
    auto idx = static_cast<int>(blockIdx.x * (kBlockDim.x * kBlockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (idx < sampleLen)
    {
        float2 r = sample[idx];

        #pragma unroll
        for (int ci = 0; ci != kCenterUnrollFactor; ++ci)
        {
            if (ci < centerLen)
            {
                float2 c = center[ci];
                res[sampleLenPadded * ci + idx] = (r.x - c.x) * (r.x - c.x) + (r.y - c.y) * (r.y - c.y);
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

    thrust::device_vector<float2> dSample(numSamples, {1.0f, 0.0f});
    thrust::device_vector<float2> dCenter(numCenters, {0.0f, 0.0f});
    dCenter[numCenters - 4] = {1.0f, 0.0f};
    dCenter[numCenters - 3] = {1.0f, 0.0f};
    dCenter[numCenters - 2] = {0.0f, 1.0f};
    dCenter[numCenters - 1] = {1.0f, 0.0f};

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
//    cudaError_t err_ = cudaFuncSetCacheConfig(
//            f2,
//            cudaFuncCachePreferL1
//    );
//
//    if (err_ != cudaSuccess)
//    {
//        std::cerr << "fsdajnklasdfjkladfsjklasdfj\n";
//        return EXIT_FAILURE;
//    }
//
//    err_ = cudaFuncSetAttribute(
//            f2,
//            cudaFuncAttributePreferredSharedMemoryCarveout,
//            cudaSharedmemCarveoutMaxL1
//    );
//
//    if (err_ != cudaSuccess)
//    {
//        std::cerr << "fsdajnklasdfjkladfsjklasdfj\n";
//        return EXIT_FAILURE;
//    }

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
                    cudaMemcpyDeviceToDevice
            );
            cudaDeviceSynchronize();

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

    std::cout << "Array of Structure "
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


//Section: Memory Workload Analysis
//    ---------------------------------------------------------------------- --------------- ------------------------------
//    Memory Throughput                                                         Gbyte/second                         466.57
//    Mem Busy                                                                             %                          26.41
//    Max Bandwidth                                                                        %                          73.41
//    L1/TEX Hit Rate                                                                      %                              0
//    L2 Hit Rate                                                                          %                          93.79
//    Mem Pipes Busy                                                                       %                           8.29
//    ---------------------------------------------------------------------- --------------- ------------------------------
//
//    WRN   The memory access pattern for global loads in L1TEX might not be optimal. On average, this kernel accesses
//          4.1 bytes per thread per memory request; but the address pattern, possibly caused by the stride between
//          threads, results in 8.0 sectors per request, or 8.0*32 = 256.0 bytes of cache data transfers per request.
//          The optimal thread address pattern for 4.1 byte accesses would result in 4.1*32 = 131.9 bytes of cache data
//          transfers per request, to maximize L1TEX cache performance. Check the Source Counters section for
//          uncoalesced global loads.
//
//
//Section: Memory Workload Analysis
//    ---------------------------------------------------------------------- --------------- ------------------------------
//    Memory Throughput                                                         Gbyte/second                         468.50
//    Mem Busy                                                                             %                          25.05
//    Max Bandwidth                                                                        %                          70.65
//    L1/TEX Hit Rate                                                                      %                              0
//    L2 Hit Rate                                                                          %                          93.91
//    Mem Pipes Busy                                                                       %                           8.19
//    ---------------------------------------------------------------------- --------------- ------------------------------
