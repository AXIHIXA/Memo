#ifndef XH_TIMER_GUARD_H
#define XH_TIMER_GUARD_H

#include <fmt/chrono.h>


namespace XH
{

template <typename Clock = std::chrono::high_resolution_clock, typename Duration = std::chrono::milliseconds>
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


extern template class TimerGuard<std::chrono::high_resolution_clock, std::chrono::milliseconds>;

}  // namespace XH


#endif  // XH_TIMER_GUARD_H
