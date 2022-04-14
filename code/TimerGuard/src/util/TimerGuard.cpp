#include "util/TimerGuard.h"


namespace XH
{

template class TimerGuard<std::chrono::high_resolution_clock, std::chrono::milliseconds>;

}  // namespace XH