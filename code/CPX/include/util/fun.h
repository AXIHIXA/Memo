#ifndef FUN_H
#define FUN_H

#include "util/pybind11_opaque.h"


namespace fun
{

void init_py_module(pybind11::module_ & m);

}  // namespace fun


#endif  // FUN_H
