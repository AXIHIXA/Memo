#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


constexpr dim3 kBlockDim {32U, 32U, 1U};
constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};
constexpr int kCenterUnrollLevel = 64;



//    iintGlobalCenter(const float2 *, int, const float2 *, int, int, float *)
//    Section: GPU Speed Of Light Throughput
//    ---------------------------------------------------------------------- --------------- ------------------------------
//    DRAM Frequency                                                           cycle/nsecond                           6.67
//    SM Frequency                                                             cycle/nsecond                           1.33
//    Elapsed Cycles                                                                   cycle                      2,327,826
//    Memory [%]                                                                           %                          58.31
//    DRAM Throughput                                                                      %                          16.69
//    Duration                                                                       msecond                           1.74
//    L1/TEX Cache Throughput                                                              %                          84.88
//    L2 Cache Throughput                                                                  %                          58.31
//    SM Active Cycles                                                                 cycle                   2,278,675.76
//    Compute (SM) [%]                                                                     %                          53.09
//    ---------------------------------------------------------------------- --------------- ------------------------------
//    WRN   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance
//          of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate
//          latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.
__global__
void iintGlobalCenter(
        const float2 * __restrict__ sample,
        int sampleLen,
        const float2 * __restrict__ center,
        int centerLen,
        int offset,
        float * __restrict__ res
)
{
    auto idx = static_cast<int>(blockIdx.x * kBlockSize + threadIdx.y * blockDim.x + threadIdx.x);


    if (idx < sampleLen)
    {
        float2 r = sample[idx];

#pragma unroll
        for (int b = 0; b != kCenterUnrollLevel; ++b)
        {
            if (int j = b + offset; j < centerLen)
            {
                float2 c = center[b + offset];
                res[sampleLen * b + idx] = (r.x - c.x) * (r.x - c.x) + (r.y - c.y) * (r.y - c.y);
            }
        }
    }
}


//    iintConstantCenter(const float2 *, int, float *)
//    Section: GPU Speed Of Light Throughput
//    ---------------------------------------------------------------------- --------------- ------------------------------
//    DRAM Frequency                                                           cycle/nsecond                           6.67
//    SM Frequency                                                             cycle/nsecond                           1.33
//    Elapsed Cycles                                                                   cycle                      1,139,587
//    Memory [%]                                                                           %                          66.53
//    DRAM Throughput                                                                      %                          34.25
//    Duration                                                                       usecond                         853.09
//    L1/TEX Cache Throughput                                                              %                          81.08
//    L2 Cache Throughput                                                                  %                          66.53
//    SM Active Cycles                                                                 cycle                   1,096,630.25
//    Compute (SM) [%]                                                                     %                          66.49
//    ---------------------------------------------------------------------- --------------- ------------------------------
//    INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced.
//          Check both the Compute Workload Analysis and Memory Workload Analysis sections.
__constant__
float2 center[kCenterUnrollLevel];


__global__
void iintConstantCenter(
        const float2 * __restrict__ sample,
        int sampleLen,
        int centerLen,
        float * __restrict__ res
)
{
    auto idx = static_cast<int>(blockIdx.x * kBlockSize + threadIdx.y * blockDim.x + threadIdx.x);

    if (idx < sampleLen)
    {
        float2 r = sample[idx];

#pragma unroll
        for (int b = 0; b != kCenterUnrollLevel; ++b)
        {
            if (b < centerLen)
            {
                float2 c = center[b];
                res[sampleLen * b + idx] = (r.x - c.x) * (r.x - c.x) + (r.y - c.y) * (r.y - c.y);
            }
        }
    }
}


int main(int argc, char * argv[])
{
    constexpr int kNumCenters = 8192;
    constexpr int kNumSamples = 10240000;
    int numDuplication = (argc == 2) ? std::stoi(argv[1]) : 1;

    // static_assert(kNumCenters % kCenterUnrollLevel == 0);
    // Now kNumCenters % kCenterUnrollLevel could be non-zero.
    // The last batch could have fewer than kCenterUnrollLevel center points.
    // Avoid index-out-of-bound errors!
    int centerLenLastBatch = kNumCenters % kCenterUnrollLevel;

    thrust::device_vector<float2> dSample(kNumSamples);
    thrust::device_vector<float2> dCenter(kNumCenters);
    thrust::device_vector<float> dBuffer(kNumSamples * kCenterUnrollLevel);
    thrust::device_vector<float> dRes(kNumCenters);

    dim3 mGridDim {kNumSamples / kBlockSize + 1U, 1U, 1U};

    thrust::device_vector<int> dOffset(kCenterUnrollLevel + 1);
    thrust::sequence(thrust::device, dOffset.begin(), dOffset.end(), 0, kNumSamples);

    std::size_t tempStorageBytes;
    cub::DeviceSegmentedReduce::Sum(
            nullptr,
            tempStorageBytes,
            dBuffer.begin(),
            dRes.begin(),
            kCenterUnrollLevel,
            dOffset.begin(),
            dOffset.begin() + 1
    );
    thrust::device_vector<unsigned char> dTempStorage(tempStorageBytes);
    std::cout << "tempStorageBytes = " << tempStorageBytes << '\n';

    // Warmup
    cudaMemset(dBuffer.begin().base().get(), 0, sizeof(float2) * dBuffer.size());
    iintGlobalCenter<<<mGridDim, kBlockDim>>>(
            dSample.begin().base().get(),
            kNumSamples,
            dCenter.begin().base().get(),
            kNumCenters,
            0,
            dBuffer.begin().base().get()
    );
    cudaDeviceSynchronize();
    cub::DeviceSegmentedReduce::Sum(
            dTempStorage.begin().base().get(),
            tempStorageBytes,
            dBuffer.begin(),
            dRes.begin(),
            kCenterUnrollLevel,
            dOffset.begin(),
            dOffset.begin() + 1
    );

    auto bb = std::chrono::high_resolution_clock::now();

    // Acutal Tests
    for (int _ = 0; _ != numDuplication; ++_)
    {
        for (int oo = 0; oo < kNumCenters; oo += kCenterUnrollLevel)
        {
            int centerLenThisBatch = (centerLenLastBatch != 0 and kNumCenters < oo + kCenterUnrollLevel) ?
                                     centerLenLastBatch :
                                     kCenterUnrollLevel;

            cudaMemset(dBuffer.begin().base().get(), 0, sizeof(float2) * dBuffer.size());

            iintGlobalCenter<<<mGridDim, kBlockDim>>>(
                    dSample.begin().base().get(),
                    kNumSamples,
                    dCenter.begin().base().get(),
                    kNumCenters,
                    oo,
                    dBuffer.begin().base().get()
            );
            cudaDeviceSynchronize();

            cub::DeviceSegmentedReduce::Sum(
                    dTempStorage.begin().base().get(),
                    tempStorageBytes,
                    dBuffer.begin(),
                    dRes.begin() + oo,
                    centerLenThisBatch,
                    dOffset.begin(),
                    dOffset.begin() + 1
            );
        }
    }

    auto tic = (std::chrono::high_resolution_clock::now() - bb).count();
    std::cout << "iintGlobalCenter "
              << static_cast<double>(tic) * 1e-6 / static_cast<double>(numDuplication)
              << " ms\n";

    bb = std::chrono::high_resolution_clock::now();

    for (int _ = 0; _ != numDuplication; ++_)
    {
        for (int oo = 0; oo < kNumCenters; oo += kCenterUnrollLevel)
        {
            int centerLenThisBatch = (centerLenLastBatch != 0 and kNumCenters < oo + kCenterUnrollLevel) ?
                                     centerLenLastBatch :
                                     kCenterUnrollLevel;

            cudaMemset(dBuffer.begin().base().get(), 0, sizeof(float2) * dBuffer.size());

            cudaMemcpyToSymbol(
                    center,
                    dCenter.begin().base().get() + oo,
                    sizeof(float2) * centerLenThisBatch,
                    0U,
                    cudaMemcpyDeviceToDevice
            );

            iintConstantCenter<<<mGridDim, kBlockDim>>>(
                    dSample.begin().base().get(),
                    kNumSamples,
                    centerLenThisBatch,
                    dBuffer.begin().base().get()
            );
            cudaDeviceSynchronize();

            cub::DeviceSegmentedReduce::Sum(
                    dTempStorage.begin().base().get(),
                    tempStorageBytes,
                    dBuffer.begin(),
                    dRes.begin() + oo,
                    centerLenThisBatch,
                    dOffset.begin(),
                    dOffset.begin() + 1
            );
        }
    }

    tic = (std::chrono::high_resolution_clock::now() - bb).count();
    std::cout << "iintConstantCenter "
              << static_cast<double>(tic) * 1e-6 / static_cast<double>(numDuplication)
              << " ms\n";

    return EXIT_SUCCESS;
}
