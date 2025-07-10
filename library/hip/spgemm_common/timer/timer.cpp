#include <chrono>
#include "spgemm_common/timer/timer.hpp"

using namespace std::chrono;

namespace common {
  void Timer::start() { time_start = high_resolution_clock::now(); }
  void Timer::stop() {
    time_stop = high_resolution_clock::now();
    time_use = duration_cast<microseconds>(time_stop - time_start);
  }

  // definition
  Timer timer[8];
} // namespace common
