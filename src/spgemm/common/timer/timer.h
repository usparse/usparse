#ifndef COMMON_TIMER_H
#define COMMON_TIMER_H

#include <chrono>

using namespace std::chrono;

namespace common {
  struct Timer {
    time_point<high_resolution_clock> time_start;
    time_point<high_resolution_clock> time_stop;
    duration<long, std::micro> time_use;
    void start() { time_start = high_resolution_clock::now(); }
    void stop() {
      time_stop = high_resolution_clock::now();
      time_use = duration_cast<microseconds>(time_stop - time_start);
    }
  } timer[8];
} // namespace common

#define ustb_spgemm_warmup_timer 0
#define ustb_spgemm_timer 1
#define ustb_spgemm_symbolic_timer 2
#define ustb_spgemm_numeric_timer 3
#define rocsparse_spgemm_warmup_timer 4
#define rocsparse_spgemm_timer 5
#define rocsparse_spgemm_symbolic_timer 6
#define rocsparse_spgemm_numeric_timer 7

#endif // COMMON_TIMER_H
