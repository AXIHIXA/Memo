#include <torch/extension.h>


torch::Tensor add_one(const torch::Tensor & t)
{
    return t + 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_one", &add_one, py::arg("t"));
}
