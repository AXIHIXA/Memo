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
                res[sampleLen * ci + idx] = (r.x - c.x) * (r.x - c.x) + (r.y - c.y) * (r.y - c.y);
            }
        }
    }
}


int main(int argc, char * argv[])
{
    constexpr int kNumSamples = 10240010;
    constexpr int kNumCenters = 34;
    int numCentersLastBatch = kNumCenters % kCenterUnrollFactor;

    thrust::device_vector<float2> dSample(kNumSamples, {1.0f, 0.0f});
    thrust::device_vector<float2> dCenter(kNumCenters, {0.0f, 0.0f});
    dCenter[kCenterUnrollFactor - 1] = {1.0f, 0.0f};
    dCenter[kNumCenters - 2] = {0.0f, 1.0f};
    dCenter[kNumCenters - 1] = {1.0f, 0.0f};
    thrust::device_vector<float> dBuffer(kNumSamples * kCenterUnrollFactor, 0.0f);
    thrust::device_vector<float> dRes(kNumCenters);

    dim3 mGridDim {(kNumSamples + kBlockSize - 1) / kBlockSize, 1U, 1U};

    int numDuplication = (argc == 2) ? std::stoi(argv[1]) : 1;

    // Warmup
    f2<<<mGridDim, kBlockDim>>>(
            dSample.data().get(),
            kNumSamples,
            kNumCenters,
            dBuffer.data().get()
    );
    cudaDeviceSynchronize();

    // atomicAdd
    // Dump global array and cub::DeviceReduce::Sum
    auto ss = std::chrono::high_resolution_clock::now();

    thrust::device_vector<int> dOffset(kNumCenters + 1);
    thrust::sequence(thrust::device, dOffset.begin(), dOffset.end(), 0, kNumSamples);

    std::size_t tempStorageBytes;
    cub::DeviceSegmentedReduce::Sum(
            nullptr,
            tempStorageBytes,
            dBuffer.data().get(),
            dRes.data().get(),
            kNumCenters,
            dOffset.data().get(),
            dOffset.data().get() + 1
    );
    cudaDeviceSynchronize();
    thrust::device_vector<unsigned char> dTempStorage(tempStorageBytes);

    for (int _ = 0; _ != numDuplication; ++_)
    {
        for (int ci = 0; ci < kNumCenters; ci += kCenterUnrollFactor)
        {
            int numCentersThisBatch =
                    (numCentersLastBatch != 0 and kNumCenters < ci + kCenterUnrollFactor) ?
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
                    kNumSamples,
                    numCentersThisBatch,
                    dBuffer.data().get()
            );
            cudaDeviceSynchronize();
            std::cerr << "fasdhjklasdfjklasdfjkldfas" << '\n';

            cub::DeviceSegmentedReduce::Sum(
                    dTempStorage.data().get(),
                    tempStorageBytes,
                    dBuffer.data().get(),
                    dRes.data().get() + ci,
                    numCentersThisBatch,
                    dOffset.data().get(),
                    dOffset.data().get() + 1
            );
            cudaDeviceSynchronize();
        }
    }

    auto tic = (std::chrono::high_resolution_clock::now() - ss).count();

    std::cout << "ConstantCenter + cub::DeviceSegmentedReduce::Sum "
              << static_cast<double>(tic) * 1e-6 / static_cast<double>(numDuplication)
              << " ms\n\n";

    thrust::host_vector<float> hRes = dRes;

    for (int i = 0; i != kNumCenters; ++i)
    {
        std::cout << hRes[i] << (i == kNumCenters - 1 ? '\n' : ' ');
    }

    return EXIT_SUCCESS;
}


/*
$ ./cmake-build-release/exe 100
ConstantCenter + cub::DeviceSegmentedReduce::Sum 7.02152 ms

1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 1.024e+07 0 2.048e+07 0
*/
