//
// Created by genshen to 2026-01-13.
//

#include "csrmv_tuner_input.h"
#include "hip/level2/csrmv_kernels/line-enhance/csrmv_line_enhance.h"
#include "usparse_tuner.h"

template std::string usparse::tuner::utils::tupleToString(const std::tuple<utuner::csrmv_tuner_input<int, double>> &t);

// THREADS_PER_BLOCK, R, ROWS_PER_BLOCK,  VEC_SIZE are the parameters to be tuned.
template <int THREADS_PER_BLOCK, int R, int ROWS_PER_BLOCK, int VEC_SIZE>
inline void line_enhance_wrapper(utuner::csrmv_tuner_input<int, double> csrmv_input) {
  // set parameters
  enum usparse::usparse_operation operation = usparse::usparse_operation::operation_none;
  double alpha = 1.0;
  double beta = 1.0;

  auto d_csr_mat_descr = usparse::csr_mat_descr<double, int, int, int>(csrmv_input.csr_mat);
  line_enhance_sparse_spmv<double, int, int, int, double, double, double, double, R, ROWS_PER_BLOCK, VEC_SIZE,
                           THREADS_PER_BLOCK>(0, operation, alpha, beta, d_csr_mat_descr, csrmv_input.x,
                                              csrmv_input.y);

  hipDeviceSynchronize();
  return;
}

USPARSE_TUNER_BENCH(uTuneLineEnhanceFP64) {
  template <int THREADS_PER_BLOCK, int R = 0, int ROWS_PER_BLOCK = 0, int VEC_SIZE = 0>
  static void utuner(usparse::tuner::States & states, std::tuple<utuner::csrmv_tuner_input<int, double>> args) {
    // skip invalid parameter group
    if constexpr (VEC_SIZE * ROWS_PER_BLOCK > THREADS_PER_BLOCK) {
      states.SkipThisTestCase();
      return;
    }

    auto [csrmv_input] = args;

    // warm up
    for (int i = 0; i < 10; i++) {
      line_enhance_wrapper<THREADS_PER_BLOCK, R, ROWS_PER_BLOCK, VEC_SIZE>(csrmv_input);
    }

    // run loops.
    for (auto state : states) {
      line_enhance_wrapper<THREADS_PER_BLOCK, R, ROWS_PER_BLOCK, VEC_SIZE>(csrmv_input);
      // todo: reset vector y after each iteration.
    }

    const size_t nnz = csrmv_input.csr_mat.nnz;
    const size_t rows = csrmv_input.csr_mat.rows;
    const size_t cols = csrmv_input.csr_mat.cols;
    const size_t mem_size = nnz * sizeof(double) + nnz * sizeof(int) + (rows + 1) * sizeof(int) +
                            (cols) * sizeof(double) + (cols) * sizeof(double) * 2;

    states.counters["Bandwidth"] = usparse::tuner::Counter(
        mem_size * states.iterations(), usparse::tuner::Counter::kIsRate, usparse::tuner::Counter::kIs1024);
  };
};

USPARSE_TUNER_DEFINE_TUNE_RANGE(uTuneLineEnhanceFP64) {
  using HIP_THREADS_PER_BLOCK = usparse::tuner::ParamRange<int, 256, 512>;
  using LINE_ENHANCE_R = usparse::tuner::ParamRange<int, 1, 2, 3, 4>;
  using LINE_ENHANCE_ROWS_PER_BLOCK = usparse::tuner::ParamRange<int, 16, 32, 64, 128>;
  using LINE_ENHANCE_VEC_SIZE =
      usparse::tuner::ParamRange<int, 1, 2, 4, 8, 16, 32, 64>; // todo: 64 is only for AMD GPUs and Hygon DCUs
  constexpr int ParamGroups = 4;
  const std::array<std::string, ParamGroups> ParameterNames = {"THREADS", "R", "ROWS_PER_BLOCK", "VEC_SIZE"};

  using ConfigType = usparse::tuner::Config<utuner::csrmv_tuner_input<int, double>>;
  auto config = new ConfigType();
  config->setIterations(10);
  config->setArgs({*global_csrmv_tuner_input_ptr});

  // call uTuneLineEnhanceFP64::utuner
  USPARSE_TUNER_LAUNCH(uTuneLineEnhanceFP64, ConfigType, config, ParamGroups, ParameterNames, HIP_THREADS_PER_BLOCK,
                       LINE_ENHANCE_R, LINE_ENHANCE_ROWS_PER_BLOCK, LINE_ENHANCE_VEC_SIZE);
}
