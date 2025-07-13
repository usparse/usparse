//
// Created by genshen to 2025-07-14.
//

#include <benchmark/benchmark.h>

static void BM_Level1DotProd(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}

BENCHMARK(BM_Level1DotProd);

BENCHMARK_MAIN();
