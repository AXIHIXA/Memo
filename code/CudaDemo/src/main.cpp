#include <iostream>

#include <fmt/core.h>

#include "cuax/CudaMatrix.h"
#include "util/TimerGuard.h"


int main(int argc, char * argv[])
{
    {
        XH::TimerGuard<> tg;
        fmt::print("{}\n", test());
    }

    {
        XH::TimerGuard<> tg;
        fmt::print("{}\n", test());
    }

    {
        XH::TimerGuard<> tg;
        fmt::print("{}\n", test());
    }

    return EXIT_SUCCESS;
}
