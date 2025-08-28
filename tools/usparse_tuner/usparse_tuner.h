//
// Created by genshen to 2025-08-28.
//

#ifndef USPARSE_TUNER_USPARSE_TUNER_HP
#define USPARSE_TUNER_USPARSE_TUNER_HP

#include <functional>
#include <iostream>
#include <tuple>
#include <utility>

#include "usparse_tuner_multi_levels.hpp"

namespace usparse {
  namespace tuner {
    namespace internal {
      extern std::vector<std::function<void()>> *tuner_entrypointers;

      struct private_tuner {
        // unused class. just for global register.
      };

      private_tuner *register_tuner(const std::string &tuner_name, std::function<void()> func);

      void run_all_tuners();

    } // namespace internal
  } // namespace tuner
} // namespace usparse

#define USPARSE_TUNER_DECLARE_Func(n) __usparse_tuner_##n
#define USPARSE_TUNER_PRIVITE_REGISTER_DECLARE(n) __usparse_tuner_reg_##n
#define USPARSE_TUNER_DECLARE_Class(n) __class_usparse_tuner_##n

#define USPARSE_TUNER_BENCH(n) struct USPARSE_TUNER_DECLARE_Class(n)

// n: tune name
// ...: the paramter ranges for tuning.
#define USPARSE_TUNER_DEFINE_TUNE_RANGE(n)                                                                             \
  void USPARSE_TUNER_DECLARE_Func(n)();                                                                                \
  const static usparse::tuner::internal::private_tuner *USPARSE_TUNER_PRIVITE_REGISTER_DECLARE(n) =                    \
      usparse::tuner::internal::register_tuner(#n, USPARSE_TUNER_DECLARE_Func(n));                                     \
  void USPARSE_TUNER_DECLARE_Func(n)()

#define USPARSE_TUNER_LAUNCH(n, ConfigType, pConfig, ...)                                                              \
  pConfig->setTuneName(#n);                                                                                            \
  usparse::tuner::internal::optimize<USPARSE_TUNER_DECLARE_Class(n), ConfigType, __VA_ARGS__>(pConfig)

#define USPARSE_TUNER_MAIN(n)                                                                                          \
  int main() {                                                                                                         \
    usparse::tuner::internal::run_all_tuners();                                                                        \
    usparse::tuner::internal::print_header();                                                                          \
    usparse::tuner::internal::print_summary();                                                                         \
    return 0;                                                                                                          \
  }

#endif // USPARSE_TUNER_USPARSE_TUNER_HP
