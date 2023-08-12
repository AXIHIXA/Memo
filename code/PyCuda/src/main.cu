#include <sstream>
#include <iostream>

#include <fmt/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "util/add.h"



PYBIND11_MODULE(MODULE_NAME, m)
{
    m.doc() = "pybind11 example plugin";
    m.def("sum", &sum, "GPU summation routine with CUB");
}


//int main(int argc, char * argv[])
//{
//    std::vector<float> vec {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//    fmt::print("{}\n", test(vec));
//
//    return EXIT_SUCCESS;
//}