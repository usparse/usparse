//
// Created by genshen to 2025-08-28.
//

#include <cmath>

#include "usparse_tuner.h"

// OFFSET is the parameter to be tuned.
template <int OFFSET> int Sqrt(int n) {
  for (int i = 0; i < 1000 * OFFSET * n; i++) {
    sqrt(i + 0.0);
  }
  return 0;
}

USPARSE_TUNER_BENCH(TuneSqrt) {
  template <int A, bool B = false, int C = 0, int D = 0, int E = 0>
  static void utuner(usparse::tuner::States & states, std::tuple<int, double> args) {
    int i = 0;
    auto [a, b] = args;
    for (auto state : states) {
      Sqrt<A + C>(a + b);
    }

    const int mem_size = 1024 * (A + C + a + b);
    states.counters["Bandwidth"] = usparse::tuner::Counter(
        mem_size * states.iterations(), usparse::tuner::Counter::kIsRate, usparse::tuner::Counter::kIs1024);
  };
};

USPARSE_TUNER_DEFINE_TUNE_RANGE(TuneSqrt) {
  using BlockSizes = usparse::tuner::ParamRange<int, 8, 16>;
  using TransposeOptions = usparse::tuner::ParamRange<bool, true, false>;
  using UnrollFactors = usparse::tuner::ParamRange<int, 6, 12>;
  using Qw = usparse::tuner::ParamRange<int, 4, 5>;
  using PP = usparse::tuner::ParamRange<int, 9, 10>;
  constexpr int ParamGroups = 5;
  const std::array<std::string, ParamGroups> ParameterNames = {"BLOCK_SIZE", "TRANS", "UNROLL", "QW", "PP"};

  using ConfigType = usparse::tuner::Config<int, double>;
  auto config = new ConfigType();
  config->setIterations(10);
  config->setArgs(std::tuple{1, 3})->setArgs(std::tuple{2, 6});

  USPARSE_TUNER_LAUNCH(TuneSqrt, ConfigType, config, ParamGroups, ParameterNames, BlockSizes, TransposeOptions,
                       UnrollFactors, Qw, PP);
}

USPARSE_TUNER_MAIN();
