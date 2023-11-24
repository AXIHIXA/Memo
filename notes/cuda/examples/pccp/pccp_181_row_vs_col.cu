#include <cuda_runtime.h>
#include <thrust/device_vector.h>


constexpr dim3 kBlockDim {32U, 32U, 1U};
constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};


__global__
void readColumnStoreRow(float * __restrict__ dst, const float * __restrict__ src, int n)
{
    auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto idy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (idx < n and idy < n)
    {
        dst[idy * n + idx] = src[idx * n + idy];
    }
}


__global__
void readRowStoreColumn(float * __restrict__ dst, const float * __restrict__ src, int n)
{
    auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto idy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (idx < n and idy < n)
    {
        dst[idx * n + idy] = src[idy * n + idx];
    }
}


__global__
void warmup(float * __restrict__ dst, const float * __restrict__ src, int n)
{
    auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto idy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (idx < n and idy < n)
    {
        dst[idy * n + idx] = src[idy * n + idx];
    }
}


int main(int argc, char * argv[])
{
    constexpr int kNumSamples = 20480000;
    int numDuplication = (argc == 2) ? 1000 : 1;
    int nn = 32 * 32;

    thrust::device_vector<float> dIn(kNumSamples);
    thrust::device_vector<float> dOut(kNumSamples);

    dim3 mGridDim {32U, 32U, 1U};

    // Warmup
    warmup<<<mGridDim, kBlockDim>>>(
            dOut.begin().base().get(),
            dIn.begin().base().get(),
            nn
    );

    cudaDeviceSynchronize();

    // Acutal Tests
    for (int _ = 0; _ != numDuplication; ++_)
    {
        readColumnStoreRow<<<mGridDim, kBlockDim>>>(
                dOut.begin().base().get(),
                dIn.begin().base().get(),
                nn
        );
    }

    cudaDeviceSynchronize();

    for (int _ = 0; _ != numDuplication; ++_)
    {
        readRowStoreColumn<<<mGridDim, kBlockDim>>>(
                dOut.begin().base().get(),
                dIn.begin().base().get(),
                nn
        );
    }

    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}


/*
$ ncu -k regex:read --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./cmake-build-release/exe

  readColumnStoreRow(float *, const float *, int)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         887.87
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                         110.98
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                          12.50
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                            100
    ---------------------------------------------------------------------- --------------- ------------------------------

  readRowStoreColumn(float *, const float *, int)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                          70.24
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                         561.94
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                            100
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                          12.50
    ---------------------------------------------------------------------- --------------- ------------------------------
*/

/*
$ nvprof ./cmake-build-release/exe adfhklahfklasd

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   59.56%  59.372ms      1000  59.372us  56.255us  800.12us  readRowStoreColumn(float*, float const *, int)
                   40.10%  39.970ms      1000  39.969us  35.936us  40.927us  readColumnStoreRow(float*, float const *, int)
*/
