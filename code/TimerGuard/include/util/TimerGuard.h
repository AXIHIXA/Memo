#ifndef XI_TIMER_GUARD_H
#define XI_TIMER_GUARD_H

#include <fmt/chrono.h>


namespace xi
{

using FloatMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;


template <typename Clock = std::chrono::high_resolution_clock, typename Duration = FloatMilliseconds>
class TimerGuard
{
public:
    using TimePoint = typename Clock::time_point;

    explicit TimerGuard(std::FILE * stream = stdout) : startTime(Clock::now()), stream(stream) {}

    ~TimerGuard()
    {
        fmt::print(stream, "{}\n", std::chrono::duration_cast<Duration>(Clock::now() - startTime));
    }

private:
    TimePoint startTime;
    std::FILE * stream;
};


extern template class TimerGuard<std::chrono::high_resolution_clock, FloatMilliseconds>;

}  // namespace xi


#endif  // XI_TIMER_GUARD_H