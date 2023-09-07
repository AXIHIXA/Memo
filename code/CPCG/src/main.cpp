#include <pybind11/pybind11.h>

#include "cpcg/cpcg.h"


PYBIND11_MODULE(MODULE_NAME, m)
{
    cpcg::init_py_module(m);
}
