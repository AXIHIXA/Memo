#ifndef XH_TIMER_GUARD_H
#define XH_TIMER_GUARD_H

#include <chrono>

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>


namespace XH
{

template <typename Clock = std::chrono::high_resolution_clock, typename Duration = std::chrono::milliseconds>
class TimerGuard
{
public:
    explicit TimerGuard(std::FILE * stream_ = stdout) : startTime(Clock::now()), stream(stream_) {}

    ~TimerGuard()
    {
        fmt::print(stream, FMT_STRING("{}\n"), std::chrono::duration_cast<Duration>(Clock::now() - startTime));
    }

private:
    typename Clock::time_point startTime;
    std::FILE * stream;
};


extern template class TimerGuard<std::chrono::high_resolution_clock, std::chrono::milliseconds>;

}  // namespace XH


#endif  // XH_TIMER_GUARD_H
