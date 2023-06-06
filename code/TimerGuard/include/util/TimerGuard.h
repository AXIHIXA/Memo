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
    using Rep = typename Duration::rep;
    using TimePoint = typename Clock::time_point;

    /// @brief RAII-style timer guard.
    /// @param t      A number of counts of the Duration (time elapsed from construction to destruction).
    /// @param stream Print time elapsed to stream.
    explicit TimerGuard(Rep * t = nullptr, std::FILE * stream = stdout) : t(t), startTime(Clock::now()), stream(stream)
    {
        // Do nothing.
    }

    ~TimerGuard()
    {
        auto timeElapsed = std::chrono::duration_cast<Duration>(Clock::now() - startTime);

        if (t)
        {
            *t = timeElapsed.count();
        }

        fmt::print(stream, "{}\n", timeElapsed);
    }

private:
    Rep * t;
    TimePoint startTime;
    std::FILE * stream;
};


extern template class TimerGuard<std::chrono::high_resolution_clock, FloatMilliseconds>;

}  // namespace xi


#endif  // XI_TIMER_GUARD_H