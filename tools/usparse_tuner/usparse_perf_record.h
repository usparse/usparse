//
// Created by genshen to 2025-08-29.
//

#ifndef USPARSE_TUNER_USPARSE_PERF_RECORD_H
#define USPARSE_TUNER_USPARSE_PERF_RECORD_H

#include "usparse_tuner_states.hpp"

#include <map>
#include <vector>

namespace usparse {
  namespace tuner {
    namespace internal {

      // state time info under a given arg and parameter.
      struct CurrentParamArgStatePerf {
        const std::string tune_param_list_name;
        const int iterations = 0;
        std::map<std::string, Counter> counters;
        const double time_duration_us = 0.0;

        CurrentParamArgStatePerf(const std::string &_tune_param_list_name, const int _iterations,
                                 const double _time_duration_us, const std::map<std::string, Counter> _counters)
            : tune_param_list_name(_tune_param_list_name), iterations(_iterations), time_duration_us(_time_duration_us),
              counters(_counters) {}
        inline double avarage_time_per_itl() { return time_duration_us / iterations; };
      };

      // the time for different args and different tuning parameters.
      struct StatesPerfList {
        // key is "Tune case name" + "arguments name", value is a vector of the perfermance under different tune
        // parameters.
        std::map<std::string, std::vector<CurrentParamArgStatePerf>> state_time_lists;

      public:
        void emplace_back(const std::string &arg_list_name, const CurrentParamArgStatePerf &performance) {
          // it can handle the case when key does not exist.
          state_time_lists[arg_list_name].emplace_back(performance);
        }
      };

      extern StatesPerfList tuner_state_perf_list;

      void print_header();
      void print_summary();

    } // namespace internal
  } // namespace tuner
} // namespace usparse

#endif // USPARSE_TUNER_USPARSE_PERF_RECORD_H
