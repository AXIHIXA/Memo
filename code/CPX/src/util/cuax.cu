#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <pybind11/numpy.h>

#include "util/cuax.h"


namespace cuax
{

namespace
{

//float cu(std::vector<float> & vec)
float cu(pybind11::array_t<float> & arr)
{
    auto num_items = static_cast<int>(arr.size());

    float * d_in;
    float * d_out;
    cudaMalloc(&d_in, sizeof(float) * num_items);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, arr.data(), sizeof(float) * num_items, cudaMemcpyHostToDevice);

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

}  // namespace anonymous


void init_py_module(pybind11::module_ & m)
{
    namespace py = pybind11;

//    py::bind_vector<std::vector<float>>(m, "FloatVector");

    m.def("cu", cu, py::arg("arr"), "A CUDA function.");
}

}  // namespace cuax
