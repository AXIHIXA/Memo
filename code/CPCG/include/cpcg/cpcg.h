#ifndef CPCG_H
#define CPCG_H

#include <pybind11/pybind11.h>


namespace cpcg
{

void init_py_module(pybind11::module_ & m);

}  // namespace cpcg


#endif  // CPCG_H
