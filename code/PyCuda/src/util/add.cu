#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "util/add.h"


float sum(std::vector<float> & vec)
{
    auto num_items = static_cast<int>(vec.size());

    float * d_in;
    float * d_out;
    cudaMalloc(&d_in, sizeof(float) * num_items);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, vec.data(), sizeof(float) * num_items, cudaMemcpyHostToDevice);

    void * d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    cudaFree(d_temp_storage);

    float res;
    cudaMemcpy(&res, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return res;
}
