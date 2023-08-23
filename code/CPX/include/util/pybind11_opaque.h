#ifndef PYBIND11_OPAQUE_H
#define PYBIND11_OPAQUE_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


PYBIND11_MAKE_OPAQUE(std::vector<float>);

PYBIND11_MAKE_OPAQUE(std::vector<int>);


#endif  // PYBIND11_OPAQUE_H
