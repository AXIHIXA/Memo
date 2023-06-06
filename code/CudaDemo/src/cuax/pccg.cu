/// CLion clangd bug, YouTrack Issue #CPP-25855.
/// Incorrect Clangd error for partial template specialization with default parameters (happens within thrust headers).
/// A workaround by @Justen Di Ruscio:
/// Go to the `Settings | Languages & Frameworks | C/C++ | Clangd`,
/// there will be a field for additional flags which are added to the every compilation command in the project.
/// Add there `-fno-relaxed-template-template-args`.

#include <fmt/core.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>


void test()
{
    // Generate random data serially.
    thrust::default_random_engine rng(1337);
    thrust::uniform_real_distribution<double> dist(-50.0, 50.0);
    thrust::host_vector<double> h_vec(32 << 20);
    thrust::generate(h_vec.begin(), h_vec.end(), [&]
    {
        return dist(rng);
    });

    // Transfer to device and compute the sum.
    thrust::device_vector<double> d_vec = h_vec;
    fmt::print("{}\n", thrust::reduce(d_vec.begin(), d_vec.end(), 0.0, thrust::plus<int>()));
}

