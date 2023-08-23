#include <execution>
#include <iostream>

#include <pybind11/numpy.h>

#include "util/fun.h"


namespace fun
{

namespace
{

void f(int a)
{
    std::cout << "int " << a << '\n';
}


void f(const std::string & s)
{
    std::cout << "std::string " << s << '\n';
}


double cu(pybind11::array_t<double, pybind11::array::c_style> & arr)
{
    const double * b = arr.data();
    const double * e = b + arr.size();

    return std::reduce(std::execution::par, b, e, 0.0);
}


void ff()
{
    std::vector<float> vec {0.0f, 0.1f, 0.2f, 0.3f};
    std::reduce(std::execution::par, vec.cbegin(), vec.cend(), 0.0f);
}


}  // namespace anonymous


void init_py_module(pybind11::module_ & m)
{
    namespace py = pybind11;

    m.def("cu", cu, py::arg("arr"));
    m.def("ff", ff);

//    // Overloading
//    m.def("f11", static_cast<void (*)(int)>(f), py::arg("a"), "void f(int)");
//    m.def("f11", static_cast<void (*)(const std::string &)>(f), py::arg("s"), "void f(const std::string & s)");
//
//    // C++14-style overloading
//    m.def("f14", py::overload_cast<int>(f), py::arg("a"), "void f(int)");
//    m.def("f14", py::overload_cast<const std::string &>(f), py::arg("s"), "void f(const std::string & s)");
}

}  // namespace fun
