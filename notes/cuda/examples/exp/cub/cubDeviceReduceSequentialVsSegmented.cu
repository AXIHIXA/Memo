#include <random>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>

#include "util/TimerGuard.h"


namespace
{

constexpr int kNumDuplications = 50;

constexpr int kVecLen = 640000000;

constexpr int kNumSegments = 64;

}  // namespace anomynous


int dumpRandomVector()
{
    unsigned int seed = std::random_device()();
    std::cout << "seed = " << seed << '\n';
    std::default_random_engine e(seed);
    std::uniform_real_distribution<float> g(-1.0f, 1.0f);

    std::vector<float> hVec(kVecLen, 0.0f);

    for (int i = 0LL; i != kVecLen; ++i)
    {
        hVec[i] = g(e);
    }

    if (std::FILE * fp = std::fopen("var/vec.bin", "wb"); fp)
    {
        std::fwrite(hVec.begin().base(), sizeof(float) * hVec.size(), 0UL, fp);
        std::fclose(fp);
    }
    else
    {
        std::fclose(fp);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int testSequentialReduction()
{
    std::vector<float> hVec(kVecLen);

    if (std::FILE * fp = std::fopen("var/vec.bin", "rb"); fp)
    {
        [[maybe_unused]] std::size_t _ = std::fread(hVec.begin().base(), sizeof(float), kVecLen, fp);
        std::fclose(fp);
    }
    else
    {
        std::fclose(fp);
        return EXIT_FAILURE;
    }

    thrust::device_vector<float> dVec = hVec;

    int segmentSize = kVecLen / kNumSegments;
    std::vector<int> hOffset;
    for (int i = 0; i <= kVecLen; i += segmentSize)
    {
        hOffset.emplace_back(i);
    }

    thrust::device_vector<float> dRes(kNumSegments, 0.0f);

    // Temp storage allocation
    std::size_t tempStroageBytes = 0UL;
    cub::DeviceReduce::Sum(
            nullptr,
            tempStroageBytes,
            dVec.data().get(),
            dRes.data().get(),
            segmentSize
    );
    std::cout << "tempStorageBytes = " << tempStroageBytes << '\n';
    thrust::device_vector<unsigned char> dTempStorage(tempStroageBytes);

    // Warmup
    cub::DeviceReduce::Sum(
            dTempStorage.data().get(),
            tempStroageBytes,
            dVec.begin().base().get(),
            dRes.begin().base().get(),
            segmentSize
    );

    {
        xi::TimerGuard<> tg;

        for (int dup = 0; dup != kNumDuplications; ++dup)
        {
            for (int seg = 0; seg != kNumSegments; ++seg)
            {
                cub::DeviceReduce::Sum(
                        dTempStorage.data().get(),
                        tempStroageBytes,
                        dVec.begin().base().get() + seg * segmentSize,
                        dRes.begin().base().get() + seg,
                        segmentSize
                );
            }
        }

        cudaDeviceSynchronize();
    }

    return EXIT_SUCCESS;
}


int testSegmentedReduction()
{
    std::vector<float> hVec(kVecLen);

    if (std::FILE * fp = std::fopen("var/vec.bin", "rb"); fp)
    {
        [[maybe_unused]] std::size_t _ = std::fread(hVec.begin().base(), sizeof(float), kVecLen, fp);
        std::fclose(fp);
    }
    else
    {
        std::fclose(fp);
        return EXIT_FAILURE;
    }

    thrust::device_vector<float> dVec = hVec;

    int segmentSize = kVecLen / kNumSegments;
    std::vector<int> hOffsets;
    for (int i = 0; i <= kVecLen; i += segmentSize)
    {
        hOffsets.emplace_back(i);
    }
    thrust::device_vector<int> dOffsets = hOffsets;

    thrust::device_vector<float> dRes(kNumSegments, 0.0f);

    // Temp storage allocation
    std::size_t tempStroageBytes = 0UL;
    cub::DeviceSegmentedReduce::Sum(
            nullptr,
            tempStroageBytes,
            dVec.data().get(),
            dRes.data().get(),
            kNumSegments,
            dOffsets.begin().base().get(),
            dOffsets.begin().base().get() + 1
    );
    std::cout << "tempStorageBytes = " << tempStroageBytes << '\n';
    thrust::device_vector<unsigned char> dTempStorage(tempStroageBytes);

    // Warmup
    cub::DeviceSegmentedReduce::Sum(
            dTempStorage.data().get(),
            tempStroageBytes,
            dVec.data().get(),
            dRes.data().get(),
            1,
            dOffsets.begin().base().get(),
            dOffsets.begin().base().get() + 1
    );

    {
        xi::TimerGuard<> tg;

        for (int dup = 0; dup != kNumDuplications; ++dup)
        {
            cub::DeviceSegmentedReduce::Sum(
                    dTempStorage.data().get(),
                    tempStroageBytes,
                    dVec.data().get(),
                    dRes.data().get(),
                    kNumSegments,
                    dOffsets.begin().base().get(),
                    dOffsets.begin().base().get() + 1
            );
        }

        cudaDeviceSynchronize();
    }

    return EXIT_SUCCESS;
}


int main(int argc, char * argv[])
{
    if (argc != 2)
    {
        return EXIT_FAILURE;
    }

    if (std::string(argv[1]) == "seq")
    {
        testSequentialReduction();
    }
    else
    {
        testSegmentedReduction();
    }

    return EXIT_SUCCESS;
}


/*
$ ./cmake-build-release/cumo seq
tempStorageBytes = 5887
237.3ms

$ ./cmake-build-release/cumo seg
tempStorageBytes = 1
222.337ms
*/
