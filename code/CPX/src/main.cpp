#include <sstream>
#include <iostream>

#include <fmt/core.h>
#include <pybind11/pybind11.h>

#include "util/fun.h"
#include "util/pb.h"


PYBIND11_MODULE(MODULE_NAME, m)
{
    namespace py = pybind11;

    m.doc() = "pybind11 example plugin";

    // Export variables
    m.attr("int_variable") = 42;
    m.attr("string_variable") = py::cast("a string variable");
    m.attr("string_variable_2") = "another string variable";

    // Overloading
    // Separate initialization code to other files to accelerate compilation
    fun::init_py_module(m);

    // Class
    // Separate initialization code to other files to accelerate compilation
    Pb::init_py_module(m);
}


//int main(int argc, char * argv[])
//{
//    std::vector<float> vec {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//    fmt::print("{}\n", test(vec));
//
//    return EXIT_SUCCESS;
//}
