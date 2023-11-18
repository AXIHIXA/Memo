#include <cuda_runtime.h>
#include <torch/extension.h>


torch::Tensor add_one(const torch::Tensor & t)
{
    return t + 1;
}


/// @brief Batch-wise 2D cross product.
/// @param a Tensor of shape (batch, 2).
/// @param b Tensor of shape (batch, 2).
/// @return  Tensor of shape (batch, 1).
torch::Tensor cross2d(const torch::Tensor & a, const torch::Tensor & b)
{
    try
    {
        torch::Tensor c = torch::ones({1, 2}).to(a.device());
        c.index_put_({0, 1}, -1);
        return torch::sum(a * b.roll(1, 1) * c, -1);
    }
    catch (std::exception & e)
    {
        std::cerr << e.what() << '\n';
        throw e;
    }
    catch (...)
    {
        std::cerr << "Caught whatever ..." << '\n';
        throw;
    }
}


namespace
{

constexpr dim3 kBlockDim {32, 4, 1};

constexpr dim3 kGridDim {1, 1, 1};

}  // namespace anomynous


__global__ void cudaPrint(float * a, int sz)
{
//    printf(
//            "blockIdx %d %d %d\n"
//            "threadIdx %d %d %d\n"
//            "=====================\n",
//            blockIdx.x, blockIdx.y, blockIdx.z,
//            threadIdx.x, threadIdx.y, threadIdx.z
//    );

    if (int offset = threadIdx.y * blockDim.x + threadIdx.x; offset < sz)
    {
        printf("a[%d] = %f\n", offset, a[offset]);
    }
}


void print(torch::Tensor & t)
{
    cudaPrint<<<kGridDim, kBlockDim>>>(t.data_ptr<float>(), static_cast<int>(t.numel()));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using py::literals::operator""_a;
    m.def("add_one", add_one, "t"_a);
    m.def("cross2d", cross2d, "a"_a, "b"_a);
    m.def("print", print, "t"_a);
}
