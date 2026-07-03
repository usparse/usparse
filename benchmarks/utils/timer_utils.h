//
// Created by chu genshen on 2025/5/7.
//

#ifndef SPMV_ACC_HIP_TIMER_UTILS_H
#define SPMV_ACC_HIP_TIMER_UTILS_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "benchmark_utils_config.h"
#include "common/usparse_utils.hpp"

// another timer implementation using hip Event.
namespace hip {
    namespace timer {
      struct event_timer {
        hipEvent_t _start, _stop;
        double time_use; // us
        static constexpr bool use_event_sync = true;

        event_timer() {
          USPARSE_HIP_API_CHECK(hipEventCreate(&_start));
          USPARSE_HIP_API_CHECK(hipEventCreate(&_stop));
        }

        ~event_timer() {
          USPARSE_HIP_API_CHECK(hipEventDestroy(_start));
          USPARSE_HIP_API_CHECK(hipEventDestroy(_stop));
        }

        inline void start() {
          USPARSE_HIP_API_CHECK(hipEventRecord(_start, NULL));
        }

        /// @brief lazy device sync will try to not sync device unless user the sync flag (from global config or local config) has been specificed.
        /// @param sync_flag the local sync flag.
        inline void stop(bool sync_flag = false) {
          USPARSE_HIP_API_CHECK(hipEventRecord(_stop, NULL));
          if(sync_flag || BENCHMARK_FORCE_KERNEL_SYNC) {
            if (use_event_sync) {
              USPARSE_HIP_API_CHECK(hipEventSynchronize(_stop));
            } else {
              USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
            }
          }

          float delta_t = 0.0f;
          USPARSE_HIP_API_CHECK(hipEventElapsedTime(&delta_t, _start, _stop));
          time_use = delta_t * 1000.0;
        }
      };
    }
}

#endif // SPMV_ACC_HIP_TIMER_UTILS_H
