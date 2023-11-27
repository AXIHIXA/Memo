#include <cuda_runtime.h>
#include <thrust/device_vector.h>


struct Float2
{
    float x;
    float y;
};


__global__
void testArrayOfStructure(
    const Float2 * __restrict__ sample,
    Float2 * __restrict__ res,
    int len
)
{
    auto i = static_cast<int>(blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (i < len)
    {
        Float2 tmp = sample[i];
        tmp.x += 0.0001f;
        tmp.y += 0.0002f;
        res[i]= tmp;
    }
}


__global__
void testArrayOfCudaFloat2(
    const float2 * __restrict__ sample,
    float2 * __restrict__ res,
    int len
)
{
    auto i = static_cast<int>(blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (i < len)
    {
        float2 tmp = sample[i];
        tmp.x += 0.0001f;
        tmp.y += 0.0002f;
        res[i]= tmp;
    }
}


__global__
void testStructureOfArray(
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
        tmpY += 0.0002f;
        resX[i] = tmpX;
        resY[i] = tmpY;
    }
}


int main(int argc, char * argv[])
{
    int numDuplication = (argc == 2) ? std::stoi(argv[1]) : 1;

    static constexpr dim3 kBlockDim {32U, 32U, 1U};
    static constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};
    static constexpr int kNumSamples = 81920010;

    int numSamples = kNumSamples;
    dim3 mGridDim {(numSamples + kBlockSize - 1U) / kBlockSize, 1U, 1U};

    // Test Array of User-defined Structure
    thrust::device_vector<Float2> dSample1(numSamples, {1.0f, 2.0f});
    thrust::device_vector<Float2> dResult1(numSamples, {0.0f, 0.0f});

    for (int _ = 0; _ != numDuplication; ++_)
    {
        testArrayOfStructure<<<mGridDim, kBlockDim>>>(
                dSample1.data().get(),
                dResult1.data().get(),
                numSamples
        );
        cudaDeviceSynchronize();
    }

    // Test Array of CUDA Built-in float2
    thrust::device_vector<float2> dSample2(numSamples, {1.0f, 2.0f});
    thrust::device_vector<float2> dResult2(numSamples, {0.0f, 0.0f});

    for (int _ = 0; _ != numDuplication; ++_)
    {
        testArrayOfCudaFloat2<<<mGridDim, kBlockDim>>>(
                dSample2.data().get(),
                dResult2.data().get(),
                numSamples
        );
        cudaDeviceSynchronize();
    }

    // Test Strucure of Array
    thrust::device_vector<float> dSampleX(numSamples, 1.0f);
    thrust::device_vector<float> dSampleY(numSamples, 2.0f);
    thrust::device_vector<float> dResultX(numSamples, 0.0f);
    thrust::device_vector<float> dResultY(numSamples, 0.0f);

    for (int _ = 0; _ != numDuplication; ++_)
    {
        testStructureOfArray<<<mGridDim, kBlockDim>>>(
                dSampleX.data().get(),
                dSampleY.data().get(),
                dResultX.data().get(),
                dResultY.data().get(),
                numSamples
        );
        cudaDeviceSynchronize();
    }

    return EXIT_SUCCESS;
}

/*
$ ncu -k regex:arrayOfStructure --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./cmake-build-release/exe

  testArrayOfStructure(const Float2 *, Float2 *, int)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         544.40
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                         544.40
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                          50.00
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                          50.00
    ---------------------------------------------------------------------- --------------- ------------------------------

  testArrayOfCudaFloat2(const float2 *, float2 *, int)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         273.14
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                         273.14
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                         100.00
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                         100.00
    ---------------------------------------------------------------------- --------------- ------------------------------

  testStructureOfArray(const float *, const float *, float *, float *, int)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         266.64
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                         266.64
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                         100.00
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                        %                         100.00
    ---------------------------------------------------------------------- --------------- ------------------------------

$ nvprof ./cmake-build-release/exe 1000
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.34%  2.48828s      1000  2.4883ms  2.4504ms  3.3896ms  testStructureOfArray(float const *, float const *, float*, float*, int)
                   33.33%  2.48748s      1000  2.4875ms  2.4511ms  3.7301ms  testArrayOfCudaFloat2(float2 const *, float2*, int)
                   33.23%  2.48024s      1000  2.4802ms  2.3999ms  3.2394ms  testArrayOfStructure(Float2 const *, Float2*, int)
*/
