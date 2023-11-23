#include <random>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>


namespace
{

constexpr int kVecLen = 640000000;

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


struct ss
{
    float x;
    float y;
};


__global__
void arrayOfStructure(
    const ss * __restrict__ aos, 
    ss * __restrict__ res, 
    int len
)
{
    auto i = static_cast<int>(blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (i < len)
    {
        ss tmp = aos[i];
        tmp.x += 0.0001f;
        tmp.y += 0.0001f;
        res[i]= tmp;
    }
}


__global__
void structureOfArray(
        const float * __restrict__ x,
        const float * __restrict__ y,
        float * __restrict__ resX,
        float * __restrict__ resY,
        int len
)
{
    auto i = static_cast<int>(blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (i < len)
    {
        float tmpX = x[i];
        float tmpY = y[i];
        tmpX += 0.0001f;
        tmpY += 0.0001f;
        resX[i] = tmpX;
        resY[i] = tmpY;
    }
}


int testArrayStructure(int argc, char * argv[])
{
    std::vector<float> hVec(kVecLen);

    if (std::FILE * fp = std::fopen("var/vec.bin", "rb"))
    {
        [[maybe_unused]] std::size_t numObjectsRead = std::fread(hVec.begin().base(), sizeof(float), kVecLen, fp);
        std::fclose(fp);
    }
    else
    {
        std::fclose(fp);
        return EXIT_FAILURE;
    }

    thrust::device_vector<float> dVec = hVec;
    thrust::device_vector<float> dRes(kVecLen);

    auto dAos = reinterpret_cast<ss *>(dVec.begin().base().get());
    auto dAosRes = reinterpret_cast<ss *>(dRes.begin().base().get());

    auto dX = dVec.begin().base().get();
    auto dY = dX + (kVecLen >> 1U);
    auto dXRes = dRes.begin().base().get();
    auto dYRes = dXRes + (kVecLen >> 1U);

    if (argc != 2)
    {
        return EXIT_FAILURE;
    }

    if (std::string(argv[1]) == "aos")
    {
        arrayOfStructure<<<dim3((kVecLen >> 1U) / 1024 + 1, 1, 1), dim3(32, 32, 1)>>>(
            dAos, dAosRes, kVecLen >> 1U
        );
    }
    else
    {
        structureOfArray<<<dim3((kVecLen >> 1U) / 1024 + 1, 1, 1), dim3(32, 32, 1)>>>(
            dX, dY, dXRes, dYRes, kVecLen >> 1U
        );
    }

    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}


int main(int argc, char * argv[])
{
    testArrayStructure(argc, argv);
    return EXIT_SUCCESS;
}
