#include <cuda_runtime.h>
#include <thrust/device_vector.h>


namespace
{

constexpr dim3 kBlockDim {32U, 32U, 1U};
constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};

}  // namespace anomynous


__global__
void setRowReadRow(int * __restrict__ out)
{
    __shared__ int tile[kBlockDim.x][kBlockDim.y];

    auto idx = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}


__global__
void setColReadCol(int * __restrict__ out)
{
    __shared__ int tile[kBlockDim.x][kBlockDim.y];

    auto idx = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}


int main(int argc, char * argv[])
{
    thrust::device_vector<int> dOut(kBlockSize);

    setRowReadRow<<<1, kBlockDim>>>(dOut.begin().base().get());
    cudaDeviceSynchronize();

    setColReadCol<<<1, kBlockDim>>>(dOut.begin().base().get());
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}


/*
$ nvprof ./cmake-build-release/cumo

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.52%  3.9680us         1  3.9680us  3.9680us  3.9680us  setColReadCol(int*)
                   29.08%  2.8480us         1  2.8480us  2.8480us  2.8480us  setRowReadRow(int*)


$ ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum ./cmake-build-release/cumo

  setRowReadRow(int *)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                                               32
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                                               32
    ---------------------------------------------------------------------- --------------- ------------------------------

  setColReadCol(int *)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                                            1,024
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                                            1,024
    ---------------------------------------------------------------------- --------------- ------------------------------
*/
