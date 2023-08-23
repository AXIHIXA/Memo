#ifndef CUAX_H
#define CUAX_H

#include "util/pybind11_opaque.h"


namespace cuax
{

void init_py_module(pybind11::module_ & m);

}  // namespace cuax


#endif  // CUAX_H
