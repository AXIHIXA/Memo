#ifndef XH_TIMER_GUARD_H
#define XH_TIMER_GUARD_H

#include <chrono>
#include <iostream>


namespace XH
{

template <typename Clock = std::chrono::high_resolution_clock, typename Duration = std::chrono::milliseconds>
class TimerGuard
{
public:
    using TimePoint = typename Clock::time_point;

    explicit TimerGuard(std::FILE * stream_ = stdout) : startTime(Clock::now()), stream(stream_) {}

    ~TimerGuard()
    {
        std::cout << std::chrono::duration_cast<Duration>(Clock::now() - startTime) << '\n';
    }

private:
    TimePoint startTime;
    std::FILE * stream;
};


extern template class TimerGuard<std::chrono::high_resolution_clock, std::chrono::milliseconds>;

}  // namespace XH


#endif  // XH_TIMER_GUARD_H
