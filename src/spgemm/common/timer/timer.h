#ifndef COMMON_TIMER_H
#define COMMON_TIMER_H

#include <chrono>

using namespace std::chrono;

struct Timer {
  time_point<high_resolution_clock> time_start;
  time_point<high_resolution_clock> time_stop;
  duration<long, std::micro> time_use;
  void start() { time_start = high_resolution_clock::now(); }
  void stop() {
    time_stop = high_resolution_clock::now();
    time_use = duration_cast<microseconds>(time_stop - time_start);
  }
};

#endif // COMMON_TIMER_H
