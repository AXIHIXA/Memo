#ifndef FUN_H
#define FUN_H

#include <pybind11/pybind11.h>


namespace fun
{

void f(int);

void f(const std::string & s);

void init_py_module(pybind11::module_ & m);

}  // namespace fun


#endif  // FUN_H
