#include <cstdlib>
#include <exception>

#include "app/VulkanApp.h"


int main(int argc, char * argv[])
{
    try
    {
        VulkanApp app {"lib/shader/vert.spv", "lib/shader/frag.spv"};
        app.run();
    }
    catch (const std::exception & e)
    {
        throw;
    }

    return EXIT_SUCCESS;
}
