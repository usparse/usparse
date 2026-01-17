//
// Created by genshen to 2025-08-28.
//

#ifndef USPARSE_TUNER_MULTI_LEVELS_HPP
#define USPARSE_TUNER_MULTI_LEVELS_HPP

#include <array>
#include <cassert>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "usparse_perf_record.h"
#include "usparse_tuner_config.hpp"
#include "usparse_tuner_states.hpp"
#include "usparse_tuner_utils.hpp"

namespace usparse {
  namespace tuner {
    template <typename T, T... Values> struct ParamRange {
      static constexpr std::array<T, sizeof...(Values)> values = {Values...};
      static constexpr size_t size = sizeof...(Values);
      using Type = T;
      static void print_rangs() {
        for (int i = 0; i < size; i++) {
          std::cout << values[i] << std::endl;
        }
        std::cout << "------" << std::endl;
      }
    };

    namespace internal {
      template <typename Func, typename ConfigType, int CurrentLevel, std::size_t ParamGroups, typename T1 = int,
                T1 P1 = 0, typename T2 = int, T2 P2 = 0, typename T3 = int, T3 P3 = 0, typename T4 = int, T4 P4 = 0,
                typename T5 = int, T5 P5 = 0, typename T6 = int, T6 P6 = 0, typename T7 = int, T7 P7 = 0,
                typename T8 = int, T8 P8 = 0>
      void perform_tune_imp(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        const ConfigType *cur_config = (config == nullptr) ? (ConfigType::NewDefaultConfig()) : config;

        cur_config->loop_over_args_list([&](const int _iterations, typename ConfigType::TypeArgTuple &args) {
          const std::string arg_list_name = cur_config->getTuneName() + usparse::tuner::utils::tupleToString(args);
          auto states = usparse::tuner::States(arg_list_name, _iterations);
          std::string tune_list_name = "";
          std::stringstream ss;
          if constexpr (CurrentLevel == 1) {
            Func::template utuner<P1>(states, args);
            ss << usparse::tuner::utils::paramFormat(param_name, P1);
          } else if constexpr (CurrentLevel == 2) {
            Func::template utuner<P1, P2>(states, args);
            ss << usparse::tuner::utils::paramFormat(param_name, P1, P2);
          } else if constexpr (CurrentLevel == 3) {
            Func::template utuner<P1, P2, P3>(states, args);
            ss << usparse::tuner::utils::paramFormat(param_name, P1, P2, P3);
          } else if constexpr (CurrentLevel == 4) {
            Func::template utuner<P1, P2, P3, P4>(states, args);
            ss << usparse::tuner::utils::paramFormat(param_name, P1, P2, P3, P4);
          } else if constexpr (CurrentLevel == 5) {
            Func::template utuner<P1, P2, P3, P4, P5>(states, args);
            ss << usparse::tuner::utils::paramFormat(param_name, P1, P2, P3, P4, P5);
          } else if constexpr (CurrentLevel == 6) {
            Func::template utuner<P1, P2, P3, P4, P5, P6>(states, args);
            ss << usparse::tuner::utils::paramFormat(param_name, P1, P2, P3, P4, P5, P6);
          } else if constexpr (CurrentLevel == 7) {
            Func::template utuner<P1, P2, P3, P4, P5, P6, P7>(states, args);
            ss << usparse::tuner::utils::paramFormat(param_name, P1, P2, P3, P4, P5, P6, P7);
          } else if constexpr (CurrentLevel == 8) {
            Func::template utuner<P1, P2, P3, P4, P5, P6, P7, P8>(states, args);
            ss << usparse::tuner::utils::paramFormat(param_name, P1, P2, P3, P4, P5, P6, P7, P8);
          } else {
            // todo: clang15 compile not pass
            // static_assert(false, "level larger than 8 is currently not support.");
          }
          tune_list_name = ss.str();

          if (!states.IsTestCaseSkipped()) {
            usparse::tuner::internal::CurrentParamArgStatePerf performance(
                cur_config->getTuneName(), usparse::tuner::utils::tupleToString(args), tune_list_name, _iterations,
                states.get_time_duration_us(), states.counters);

            usparse::tuner::internal::tuner_state_perf_list.emplace_back(arg_list_name, performance);
          }
        });
      }

      template <typename Func, typename ConfigType, std::size_t ParamGroups, typename T1, typename T2, typename T3,
                typename T4, typename T5, typename T6, typename T7, T1 P1, T2 P2, T3 P3, T4 P4, T5 P5, T6 P6, T7 P7,
                typename ParamRange8 = ParamRange<int>>
      void level7(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        using CurrentRange = ParamRange8;
        using CurrentType = typename CurrentRange::Type;
        constexpr int CurrentLevel = 8;

        [&]<std::size_t... I>(std::index_sequence<I...>) {
          if constexpr (ParamGroups == CurrentLevel) {
            (perform_tune_imp<Func, ConfigType, CurrentLevel, ParamGroups, T1, P1, T2, P2, T3, P3, T4, P4, T5, P5, T6,
                              P6, T7, P7, CurrentType, CurrentRange::values[I]>(param_name, config),
             ...);
          } else {
            // todo: clang15 compile not pass
            // static_assert(false, "level larger than 8 is currently not support.");
          }
        }(std::make_index_sequence<CurrentRange::size>{});
      }

      template <typename Func, typename ConfigType, std::size_t ParamGroups, typename T1, typename T2, typename T3,
                typename T4, typename T5, typename T6, T1 P1, T2 P2, T3 P3, T4 P4, T5 P5, T6 P6,
                typename ParamRange7 = ParamRange<int>, typename ParamRange8 = ParamRange<int>>
      void level6(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        using CurrentRange = ParamRange7;
        using CurrentType = typename CurrentRange::Type;
        constexpr int CurrentLevel = 7;

        [&]<std::size_t... I>(std::index_sequence<I...>) {
          if (ParamGroups == CurrentLevel) {
            (perform_tune_imp<Func, ConfigType, CurrentLevel, ParamGroups, T1, P1, T2, P2, T3, P3, T4, P4, T5, P5, T6,
                              P6, CurrentType, CurrentRange::values[I]>(param_name, config),
             ...);
          } else {
            (level7<Func, ConfigType, ParamGroups, T1, T2, T3, T4, T5, T6, CurrentType, P1, P2, P3, P4, P5, P6,
                    (CurrentRange::values[I]), ParamRange8>(param_name, config),
             ...);
          }
        }(std::make_index_sequence<CurrentRange::size>{});
      }

      template <typename Func, typename ConfigType, std::size_t ParamGroups, typename T1, typename T2, typename T3,
                typename T4, typename T5, T1 P1, T2 P2, T3 P3, T4 P4, T5 P5, typename ParamRange6 = ParamRange<int>,
                typename ParamRange7 = ParamRange<int>, typename ParamRange8 = ParamRange<int>>
      void level5(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        using CurrentRange = ParamRange6;
        using CurrentType = typename CurrentRange::Type;
        constexpr int CurrentLevel = 6;

        [&]<std::size_t... I>(std::index_sequence<I...>) {
          if (ParamGroups == CurrentLevel) {
            (perform_tune_imp<Func, ConfigType, CurrentLevel, ParamGroups, T1, P1, T2, P2, T3, P3, T4, P4, T5, P5,
                              CurrentType, CurrentRange::values[I]>(param_name, config),
             ...);
          } else {
            (level6<Func, ConfigType, ParamGroups, T1, T2, T3, T4, T5, CurrentType, P1, P2, P3, P4, P5,
                    (CurrentRange::values[I]), ParamRange6, ParamRange7, ParamRange8>(param_name, config),
             ...);
          }
        }(std::make_index_sequence<CurrentRange::size>{});
      }

      template <typename Func, typename ConfigType, std::size_t ParamGroups, typename T1, typename T2, typename T3,
                typename T4, T1 P1, T2 P2, T3 P3, T4 P4, typename ParamRange5 = ParamRange<int>,
                typename ParamRange6 = ParamRange<int>, typename ParamRange7 = ParamRange<int>,
                typename ParamRange8 = ParamRange<int>>
      void level4(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        using CurrentRange = ParamRange5;
        using CurrentType = typename CurrentRange::Type;
        constexpr int CurrentLevel = 5;

        [&]<std::size_t... I>(std::index_sequence<I...>) {
          if (ParamGroups == CurrentLevel) {
            (perform_tune_imp<Func, ConfigType, CurrentLevel, ParamGroups, T1, P1, T2, P2, T3, P3, T4, P4, CurrentType,
                              CurrentRange::values[I]>(param_name, config),
             ...);
          } else {
            (level5<Func, ConfigType, ParamGroups, T1, T2, T3, T4, CurrentType, P1, P2, P3, P4,
                    (CurrentRange::values[I]), ParamRange6, ParamRange7, ParamRange8>(param_name, config),
             ...);
          }
        }(std::make_index_sequence<CurrentRange::size>{});
      }

      template <typename Func, typename ConfigType, std::size_t ParamGroups, typename T1, typename T2, typename T3,
                T1 P1, T2 P2, T3 P3, typename ParamRange4 = ParamRange<int>, typename ParamRange5 = ParamRange<int>,
                typename ParamRange6 = ParamRange<int>, typename ParamRange7 = ParamRange<int>,
                typename ParamRange8 = ParamRange<int>>
      void level3(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        using CurrentRange = ParamRange4;
        using CurrentType = typename CurrentRange::Type;
        constexpr int CurrentLevel = 4;

        [&]<std::size_t... I>(std::index_sequence<I...>) {
          if (ParamGroups == CurrentLevel) {
            (perform_tune_imp<Func, ConfigType, CurrentLevel, ParamGroups, T1, P1, T2, P2, T3, P3, CurrentType,
                              CurrentRange::values[I]>(param_name, config),
             ...);
          } else {
            (level4<Func, ConfigType, ParamGroups, T1, T2, T3, CurrentType, P1, P2, P3, (CurrentRange::values[I]),
                    ParamRange5, ParamRange6, ParamRange7, ParamRange8>(param_name, config),
             ...);
          }
        }(std::make_index_sequence<CurrentRange::size>{});
      }

      template <typename Func, typename ConfigType, std::size_t ParamGroups, typename T1, typename T2, T1 P1, T2 P2,
                typename ParamRange3 = ParamRange<int>, typename ParamRange4 = ParamRange<int>,
                typename ParamRange5 = ParamRange<int>, typename ParamRange6 = ParamRange<int>,
                typename ParamRange7 = ParamRange<int>, typename ParamRange8 = ParamRange<int>>
      void level2(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        using CurrentRange = ParamRange3;
        using CurrentType = typename CurrentRange::Type;
        constexpr int CurrentLevel = 3;

        [&]<std::size_t... I>(std::index_sequence<I...>) {
          if (ParamGroups == CurrentLevel) {
            (perform_tune_imp<Func, ConfigType, CurrentLevel, ParamGroups, T1, P1, T2, P2, CurrentType,
                              CurrentRange::values[I]>(param_name, config),
             ...);
          } else {
            (level3<Func, ConfigType, ParamGroups, T1, T2, CurrentType, P1, P2, (CurrentRange::values[I]), ParamRange4,
                    ParamRange5, ParamRange6, ParamRange7, ParamRange8>(param_name, config),
             ...);
          }
        }(std::make_index_sequence<CurrentRange::size>{});
      }

      template <typename Func, typename ConfigType, std::size_t ParamGroups, typename T1, T1 P1,
                typename ParamRange2 = ParamRange<int>, typename ParamRange3 = ParamRange<int>,
                typename ParamRange4 = ParamRange<int>, typename ParamRange5 = ParamRange<int>,
                typename ParamRange6 = ParamRange<int>, typename ParamRange7 = ParamRange<int>,
                typename ParamRange8 = ParamRange<int>>
      void level1(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        using CurrentRange = ParamRange2;
        using CurrentType = typename CurrentRange::Type;
        constexpr int CurrentLevel = 2;

        [&]<std::size_t... I>(std::index_sequence<I...>) {
          if (ParamGroups == CurrentLevel) {
            (perform_tune_imp<Func, ConfigType, CurrentLevel, ParamGroups, T1, P1, CurrentType,
                              CurrentRange::values[I]>(param_name, config),
             ...);
          } else {
            (level2<Func, ConfigType, ParamGroups, T1, CurrentType, P1, (CurrentRange::values[I]), ParamRange3,
                    ParamRange4, ParamRange5, ParamRange6, ParamRange7, ParamRange8>(param_name, config),
             ...);
          }
        }(std::make_index_sequence<CurrentRange::size>{});
      }

      template <typename Func, typename ConfigType, std::size_t ParamGroups, typename ParamRange1 = ParamRange<int>,
                typename ParamRange2 = ParamRange<int>, typename ParamRange3 = ParamRange<int>,
                typename ParamRange4 = ParamRange<int>, typename ParamRange5 = ParamRange<int>,
                typename ParamRange6 = ParamRange<int>, typename ParamRange7 = ParamRange<int>,
                typename ParamRange8 = ParamRange<int>>
      void optimize_imp(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        using CurrentRange = ParamRange1;
        using CurrentType = typename CurrentRange::Type;
        constexpr int CurrentLevel = 1;

        [&]<std::size_t... I>(std::index_sequence<I...>) {
          if (ParamGroups == CurrentLevel) {
            (perform_tune_imp<Func, ConfigType, CurrentLevel, ParamGroups, CurrentType, CurrentRange::values[I]>(
                 param_name, config),
             ...);
          } else {
            (level1<Func, ConfigType, ParamGroups, CurrentType, (CurrentRange::values[I]), ParamRange2, ParamRange3,
                    ParamRange4, ParamRange5, ParamRange6, ParamRange7, ParamRange8>(param_name, config),
             ...);
          }
        }(std::make_index_sequence<CurrentRange::size>{});
      }

      template <typename Func, typename ConfigType, std::size_t ParamGroups, typename... ParamRanges>
      void optimize(const std::array<std::string, ParamGroups> param_name, const ConfigType *config = nullptr) {
        optimize_imp<Func, ConfigType, ParamGroups, ParamRanges...>(param_name, config);
      }
    } // namespace internal
  } // namespace tuner
} // namespace usparse

#endif // USPARSE_TUNER_MULTI_LEVELS_HPP
