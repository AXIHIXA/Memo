#include <iostream>

#include "util/fun.h"


namespace fun
{

void f(int a)
{
    std::cout << "int " << a << '\n';
}


void f(const std::string & s)
{
    std::cout << "std::string " << s << '\n';
}


void init_py_module(pybind11::module_ & m)
{
    namespace py = pybind11;

    // Overloading
    m.def("f11", static_cast<void (*)(int)>(f), py::arg("a"), "void f(int)");
    m.def("f11", static_cast<void (*)(const std::string &)>(f), py::arg("s"), "void f(const std::string & s)");

    // C++14-style overloading
    m.def("f14", py::overload_cast<int>(f), py::arg("a"), "void f(int)");
    m.def("f14", py::overload_cast<const std::string &>(f), py::arg("s"), "void f(const std::string & s)");
}

}  // namespace fun
