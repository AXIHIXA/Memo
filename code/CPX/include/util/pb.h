#ifndef PB_H
#define PB_H

#include <string>
#include <utility>

#include <pybind11/pybind11.h>


class Pb
{
public:
    static void init_py_module(pybind11::module_ & m);

    static std::string random(const std::string & s1, const std::string & s2);

    Pb() = default;
    explicit Pb(std::string name_) : name(std::move(name_)) {}

    Pb(const Pb &) = default;
    Pb(Pb &&) = default;
    Pb & operator=(const Pb &) = default;
    Pb & operator=(Pb &&) = default;

    ~Pb() = default;

    void setName(std::string name_);

    [[nodiscard]] const std::string & getName() const;

    std::string name;
};


#endif  // PB_H
