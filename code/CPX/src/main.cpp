#include <sstream>
#include <iostream>

#include <fmt/core.h>

#include "util/pybind11_opaque.h"
//#include "util/cuax.h"
#include "util/fun.h"
//#include "util/pb.h"


PYBIND11_MODULE(MODULE_NAME, m)
{
    fun::init_py_module(m);
}


//int main(int argc, char * argv[])
//{
//    std::vector<float> vec {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//    fmt::print("{}\n", test(vec));
//
//    return EXIT_SUCCESS;
//}
