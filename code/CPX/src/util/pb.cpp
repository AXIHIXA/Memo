#include <iostream>

#include "util/pb.h"


void Pb::init_py_module(pybind11::module_ & m)
{
    namespace py = pybind11;

    py::class_<Pb> pb(m, "Pb");
    pb.def_static("random", &Pb::random, py::arg("s1") = "s1", py::arg("s2") = "s2");
    pb.def(py::init<>());
    pb.def(py::init<std::string>(), py::arg("name_"));
    pb.def("__repr__",
           [](const Pb & a)
           {
               return "<Pb, name = '" + a.name + "'>";
           }
    );
    pb.def_readwrite("name", &Pb::name);  // expose as public data member
//    pb.def_property("name", &Pb::getName, &Pb::setName);  // expose like public data member via getter/setter
//    pb.def("getName", &Pb::getName);  // Expose getter/setter
//    pb.def("setName", &Pb::setName, py::arg("name_"));  // Expose getter/setter
}


std::string Pb::random(const std::string & s1, const std::string & s2)
{
    return '[' + s1 + "] [" + s2 + ']';
}


void Pb::setName(std::string name_)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
    name = std::move(name_);
}


[[nodiscard]] const std::string & Pb::getName() const
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
    return name;
}

