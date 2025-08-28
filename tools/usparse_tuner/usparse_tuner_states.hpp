//
// Created by genshen to 2025-08-28.
//

#ifndef USPARSE_TUNER_STATES_HPP
#define USPARSE_TUNER_STATES_HPP

#include <chrono>
#include <map>
#include <string>

#include "usparse_tuner_building_config.h"

namespace usparse {
  namespace tuner {
    struct CurrentParamStateInfo;
    class Counter;

    // States is used for performing iterations in performance testing.
    class States {
    private:
      int _iterations = 0;

    public:
      States(const std::string &args_key, const int iterations) : _iterations(iterations) {}

      class iterator {
      private:
        States *_parent = nullptr;
        int current = 0;

      public:
        iterator(States *parent, int c) : _parent(parent), current(c) {}

        int operator*() const { return current; }

        iterator &operator++() {
          --current;
          if (current == 0) {
            _parent->finish_timing();
          }
          return *this;
        }

        bool operator!=(const iterator &other) const { return current != other.current; }
      };

      iterator begin() { return iterator(this, _iterations); }

      iterator end() {
        start_timing();
        return iterator(this, 0);
      }

    private:
      std::chrono::high_resolution_clock::time_point start_time;

      double time_duration_us = 0.0;
      void start_timing() {
        if (TUNE_DEBUG) {
          printf("state start\n");
        }
        start_time = std::chrono::high_resolution_clock::now();
      }

      void finish_timing() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time);
        time_duration_us = duration.count();
        if (TUNE_DEBUG) {
          printf("state end with: %g μs\n", time_duration_us);
        }
      }

    public:
      std::map<std::string, Counter> counters;

      int iterations() const { return _iterations; };

      double get_time_duration_us() { return time_duration_us / _iterations; }
    };

    // tuner Counter's API keeps the same as google benchmark's Counter in https://github.com/google/benchmark
    class Counter {
    public:
      enum Flags {
        kDefaults = 0,
        // Mark the counter as a rate.
        // It will be presented divided by the duration in the tuner.
        kIsRate = 1 << 0,
      };

      enum KIs {
        // 1'000 items per 1k
        kIs1000 = 1000,
        // 1'024 items per 1k
        kIs1024 = 1024
      };

      constexpr static long K1000 = 1000;
      // constexpr static long M1000 = 1000 * K1000;
      // constexpr static long G1000 = 1000 * M1000;
      // constexpr static long T1000 = 1000 * G1000;

      constexpr static long K1024 = 1024;
      // constexpr static long M1024 = 1024 * K1024;
      // constexpr static long G1024 = 1024 * M1024;
      // constexpr static long T1024 = 1024 * G1024;

      double value;
      Flags flags;
      KIs oneK;

      Counter(double val = 0., Flags flag = kDefaults, KIs k = kIs1000) : value(val), flags(flag), oneK(k) {}

      // calculate the counter (to rate) via the given config.
      // time is the total time in all iterations.
      std::string as_str(const int iterations, const double duration_time_us) {
        if (flags == Flags::kDefaults) {
          return std::to_string(value / iterations);
        } else if (flags == Flags::kIsRate) {
          const double rate = value * 1000 * 1000 / duration_time_us;
          // convert rate to a unit.
          if (oneK == KIs::kIs1000) {
            constexpr int MAX_UNITS = 5;
            std::array<std::string, MAX_UNITS> units = {"/s", "K/s", "M/s", "G/s", "T/s"};
            double new_rate = rate;
            for (int i = 0; i < MAX_UNITS; i++) {
              if (new_rate < K1000) {
                return std::to_string(new_rate) + units[i];
              }
              new_rate = new_rate / KIs::kIs1000; // rate too large
            }
            return std::to_string(new_rate) + "/s";
          } else if (oneK == KIs::kIs1024) {
            constexpr int MAX_UNITS = 5;
            std::array<std::string, MAX_UNITS> units = {"/s", "Ki/s", "Mi/s", "Gi/s", "Ti/s"};
            double new_rate = rate;
            for (int i = 0; i < MAX_UNITS; i++) {
              if (new_rate < K1024) {
                return std::to_string(new_rate) + units[i];
              }
              new_rate = new_rate / KIs::kIs1024;
            }
            return std::to_string(new_rate) + "/s"; // rate too large
          } else {
            return "";
          }
        } else {
          return "";
        }
      }
    };

  }; // namespace tuner
} // namespace usparse

#endif // USPARSE_TUNER_STATES_HPP
