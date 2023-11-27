#include <random>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>


int dumpRandomVector(int len)
{
    unsigned int seed = std::random_device()();
    std::cout << "seed = " << seed << '\n';
    std::default_random_engine e(seed);
    std::uniform_real_distribution<float> g(-1.0f, 1.0f);

    std::vector<float> hVec(len, 0.0f);

    for (int i = 0LL; i != len; ++i)
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