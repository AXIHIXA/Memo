#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>


__global__ void readOffset(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
    {
        C[i] = A[k] + B[k];
    }
}


__global__ void readOffsetUnroll4(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int k = i + offset;

    if (k + 3 * blockDim.x < n)
    {
        C[i] = A[k] + B[k];  // NOTE: The textbook omitted + B[k], leading to fake results!
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
        C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
        C[i + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
    }
}


int main(int argc, char * argv[])
{
    int numDuplication = (argc == 2) ? std::stoi(argv[1]) : 1;

    int numSamples = 409600000;
    thrust::device_vector<float> dA(static_cast<int>(numSamples * 1.2), 0.0f);
    thrust::device_vector<float> dB(static_cast<int>(numSamples * 1.2), 0.0f);
    thrust::device_vector<float> dC(numSamples, 0.0f);

    static constexpr dim3 kBlockDim {1024U, 1U, 1U};
    static constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};

    // ReadOffset
    dim3 mGridDim {(numSamples + (1 * kBlockSize) - 1) / (1 * kBlockSize), 1U, 1U};

    readOffset<<<mGridDim, kBlockDim>>>(
            dA.data().get(),
            dB.data().get(),
            dC.data().get(),
            numSamples,
            11
    );
    cudaDeviceSynchronize();

    auto ss = std::chrono::high_resolution_clock::now();

    for (int _ = 0; _ != numDuplication; ++_)
    {
        readOffset<<<mGridDim, kBlockDim>>>(
                dA.data().get(),
                dB.data().get(),
                dC.data().get(),
                numSamples,
                11
        );
        cudaDeviceSynchronize();
    }

    auto tic = (std::chrono::high_resolution_clock::now() - ss).count();

    std::cout << "readOffset "
              << static_cast<double>(tic) * 1e-6 / static_cast<double>(numDuplication)
              << " ms\n\n";

    // readOffsetUnroll4
    mGridDim.x /= 4;

    readOffsetUnroll4<<<mGridDim, kBlockDim>>>(
            dA.data().get(),
            dB.data().get(),
            dC.data().get(),
            numSamples,
            11
    );
    cudaDeviceSynchronize();

    ss = std::chrono::high_resolution_clock::now();

    for (int _ = 0; _ != numDuplication; ++_)
    {
        readOffsetUnroll4<<<mGridDim, kBlockDim>>>(
                dA.data().get(),
                dB.data().get(),
                dC.data().get(),
                numSamples,
                11
        );
        cudaDeviceSynchronize();
    }

    tic = (std::chrono::high_resolution_clock::now() - ss).count();

    std::cout << "readOffsetUnroll4 "
              << static_cast<double>(tic) * 1e-6 / static_cast<double>(numDuplication)
              << " ms\n\n";

    return EXIT_SUCCESS;
}


/*
$ ncu -k regex:read --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct  ./cmake-build-release/exe

  readOffset(const float *, const float *, float *, int, int)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         367.24
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                         185.19
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                          80.00
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                         100.00
    ---------------------------------------------------------------------- --------------- ------------------------------

  readOffsetUnroll4(const float *, const float *, float *, int, int)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         411.00
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                         184.68
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                          80.00
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                         100.00
    ---------------------------------------------------------------------- --------------- ------------------------------

$ ./cmake-build-release/exe 100

  readOffset        9.08116 ms
  readOffsetUnroll4 9.13725 ms
*/
