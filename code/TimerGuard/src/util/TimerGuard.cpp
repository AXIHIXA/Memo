#include "util/TimerGuard.h"


namespace xi
{

template class TimerGuard<std::chrono::high_resolution_clock, FloatMilliseconds>;

}  // namespace xi