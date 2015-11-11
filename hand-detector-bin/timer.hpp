#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>

template<typename TimeResolution, typename Functor>
TimeResolution timeToRunT(Functor doFunc) {
    std::chrono::system_clock clock;
    std::chrono::system_clock::time_point before = clock.now();
    doFunc();
    std::chrono::system_clock::time_point after = clock.now();
    TimeResolution duration =
        std::chrono::duration_cast<TimeResolution>(after - before);
    return duration;
}

#define timeToRunNs timeToRunT<std::chrono::nanoseconds>
#define timeToRunMicro timeToRunT<std::chrono::microseconds>
#define timeToRunMs timeToRunT<std::chrono::milliseconds>

#endif  // TIMER_H_
