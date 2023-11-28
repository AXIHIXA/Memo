#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


// CUDA API error checking
inline constexpr int kCudaUtilsBufferSize = 1024;

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            char checkBuf[kCudaUtilsBufferSize] {'\0'};                                            \
            std::sprintf(checkBuf, "%s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            throw std::runtime_error(checkBuf);                                                    \
        }                                                                                          \
    } while (false)

#define CUDA_CHECK_LAST_ERROR()                                                                    \
    do {                                                                                           \
        cudaError_t err_ = cudaGetLastError();                                                     \
        if (err_ != cudaSuccess) {                                                                 \
            char checkBuf[kCudaUtilsBufferSize] {'\0'};                                            \
            std::sprintf(checkBuf, "%s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            throw std::runtime_error(checkBuf);                                                    \
        }                                                                                          \
    } while (false)


constexpr dim3 kBlockDim {32U, 32U, 1U};

constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};

constexpr int kCenterUnrollFactor = 32;

constexpr int kBlockUnrollFactor = 16;


__constant__
float2 center[kCenterUnrollFactor];


__global__
void f2(
        const float2 * __restrict__ sample,
        int sampleLen,
        int segmentLenPadded,
        int centerLen,
        float * __restrict__ res
)
{
    auto blockSize = static_cast<int>(blockDim.x * blockDim.y);
    auto idx = static_cast<int>(blockIdx.x * (kBlockUnrollFactor * blockSize) + threadIdx.y * blockDim.x + threadIdx.x);

    if (idx < sampleLen)
    {
        float2 r[kBlockUnrollFactor] = {};
        int riMax = 0;

        for (int si = idx; riMax != kBlockUnrollFactor and si < sampleLen; ++riMax, si += blockSize)
        {
            r[riMax] = sample[si];
        }

        auto resIdx = static_cast<int>(blockIdx.x * blockSize + threadIdx.y * blockDim.x + threadIdx.x);

        #pragma unroll
        for (int ci = 0; ci != kCenterUnrollFactor; ++ci)
        {
            if (ci < centerLen)
            {
                float2 c = center[ci];

                float tmp = 0.0f;

                for (int ri = 0; ri != riMax; ++ri)
                {
                    tmp += (r[ri].x - c.x) * (r[ri].x - c.x) + (r[ri].y - c.y) * (r[ri].y - c.y);
                }

                res[segmentLenPadded * ci + resIdx] = tmp;
            }
        }
    }
}


inline std::pair<int, int> padTo32k(int a)
{
    static constexpr int k32 = 32;

    if (int r = a % k32; r == 0)
    {
        return {a, 0};
    }
    else
    {
        return {a + k32 - r, r};
    }
}


int main(int argc, char * argv[])
{
    int numDuplication = (argc == 2) ? std::stoi(argv[1]) : 1;

    static constexpr int kNumSamplesInit = 10244321;
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
    int numSamplesPerBlock = kBlockSize * kBlockUnrollFactor;
    unsigned int numBlocks = (numSamples + numSamplesPerBlock - 1) / numSamplesPerBlock;
    int segmentLen = static_cast<int>(numBlocks * kBlockSize);
    auto [segmentLenPadded, remainder] = padTo32k(segmentLen);
    // int numSamplesPadded = numSamples;  // Test for non-aligned pattern. Slower!
    thrust::device_vector<float> dBuffer(segmentLenPadded * kCenterUnrollFactor, 0.0f);
    dim3 mGridDim {numBlocks, 1U, 1U};

    // Test
    auto ss = std::chrono::high_resolution_clock::now();

    thrust::device_vector<int> dBeginOffset(numCenters);
    thrust::device_vector<int> dEndOffset(numCenters);
    thrust::device_vector<float> dResult(numCenters);
    thrust::sequence(thrust::device, dBeginOffset.begin(), dBeginOffset.end(), 0, segmentLenPadded);
    thrust::sequence(thrust::device, dEndOffset.begin(), dEndOffset.end(), segmentLen, segmentLenPadded);

    std::size_t tempStorageBytes;
    CUDA_CHECK(
            cub::DeviceSegmentedReduce::Sum(
                    nullptr,
                    tempStorageBytes,
                    dBuffer.data().get(),
                    dResult.data().get(),
                    numCenters,
                    dBeginOffset.data().get(),
                    dEndOffset.data().get()
            ));
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_vector<unsigned char> dTempStorage(tempStorageBytes);

    CUDA_CHECK(
            cudaMemsetAsync(
                    dBuffer.data().get(),
                    0,
                    sizeof(float) * dBuffer.size(),
                    nullptr
            )
    );

    for (int _ = 0; _ != numDuplication; ++_)
    {
        for (int ci = 0; ci < numCenters; ci += kCenterUnrollFactor)
        {
            int numCentersThisBatch =
                    (numCentersLastBatch != 0 and numCenters <= ci + kCenterUnrollFactor) ?
                    numCentersLastBatch :
                    kCenterUnrollFactor;

            CUDA_CHECK(
                    cudaMemcpyToSymbolAsync(
                            center,
                            dCenter.data().get() + ci,
                            numCentersThisBatch * sizeof(float2),
                            0UL,
                            cudaMemcpyDeviceToDevice,
                            nullptr
                    )
            );

            f2<<<mGridDim, kBlockDim, 0U, nullptr>>>(
                    dSample.data().get(),
                    numSamples,
                    segmentLenPadded,
                    numCentersThisBatch,
                    dBuffer.data().get()
            );
            CUDA_CHECK_LAST_ERROR();

//            cudaDeviceSynchronize();
//            thrust::host_vector<float> hBuffer = dBuffer;
//            for (float ff : hBuffer)
//            {
//                std::cout << ff << ' ';
//            }
//            std::printf("\n%10.2f\n", thrust::reduce(thrust::device, dBuffer.begin(), dBuffer.end()));

            CUDA_CHECK(
                    cub::DeviceSegmentedReduce::Sum(
                            dTempStorage.data().get(),
                            tempStorageBytes,
                            dBuffer.data().get(),
                            dResult.data().get() + ci,
                            numCentersThisBatch,
                            dBeginOffset.data().get(),
                            dEndOffset.data().get(),
                            nullptr
                    )
            );

            CUDA_CHECK(
                    cudaMemsetAsync(
                            dBuffer.data().get(),
                            0,
                            sizeof(float) * dBuffer.size(),
                            nullptr
                    )
            );
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto tic = (std::chrono::high_resolution_clock::now() - ss).count();

    std::cout << "After Unrolling "
              << static_cast<double>(tic) * 1e-6 / static_cast<double>(numDuplication)
              << " ms\n\n";

    if (numDuplication == 1)
    {
        thrust::host_vector<float> hRes = dResult;

        for (int i = 0; i != numCenters; ++i)
        {
            std::printf("sum @ center[%d] = %10.2f\n", i, hRes[i]);
        }
    }

    return EXIT_SUCCESS;
}


/*
$ ./cmake-build-release/exe 40

# Block Unroll 4
After Unrolling 587.185 ms
After Unrolling 551.579 ms

# Block Unroll 8
After Unrolling 284.009 ms
After Unrolling 284.418 ms

# Block Unroll 16
After Unrolling 206.704 ms
After Unrolling 207.629 ms
*/