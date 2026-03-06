//
// Created by you chen to 2026-03-06.
//

#include "../common/csrmv_tuner_input.h"
#include "common/usparse_utils.hpp"
#include "hip/level2/csrmv_kernels/flat/csrmv_flat.cpp"
#include "hip/level2/csrmv_kernels/flat/csrmv_flat.h"
#include "usparse_tuner.h"

template std::string usparse::tuner::utils::tupleToString(const std::tuple<utuner::csrmv_tuner_input<int, double>> &t);

// THREADS_PER_BLOCK, R, ROWS_PER_BLOCK,  VEC_SIZE are the parameters to be tuned.
template <int STAGE, int THREADS_PER_BLOCK, int R, int VEC_SIZE>
inline void flat_wrapper(utuner::csrmv_tuner_input<int, double> csrmv_input, csrmv_flat_mtx_info<int, double> *info) {
  // set parameters
  enum usparse::usparse_operation operation = usparse::usparse_operation::operation_none;
  double alpha = 1.0;
  double beta = 0.0;

  auto d_csr_mat_descr = usparse::csr_mat_descr<double, int, int, int>(csrmv_input.csr_mat);

  flat_run_stage_template<double, int, int, int, int, double, double, double, double, STAGE,
                          FLAT_STORE_Y_POLICY_SEGMENTED_SUM, true, R, THREADS_PER_BLOCK, VEC_SIZE>(
      hipStreamDefault, d_csr_mat_descr, info, 0, alpha, beta, csrmv_input.x, csrmv_input.y);

  if constexpr (STAGE == FLAT_STAGE_COMPUTE) {
    USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
  }
}

USPARSE_TUNER_BENCH(uTuneFlatFP64) {
  template <int THREADS_PER_BLOCK, int R = 1, int VEC_SIZE = 1>
  static void utuner(usparse::tuner::States & states, std::tuple<utuner::csrmv_tuner_input<int, double>> args) {
    if constexpr (VEC_SIZE > THREADS_PER_BLOCK || VEC_SIZE == 0) {
      states.SkipThisTestCase();
      return;
    }

    auto [csrmv_input] = args;
    csrmv_flat_mtx_info<int, double> flat_info;
    // analyse
    flat_wrapper<FLAT_STAGE_ANALYSE, THREADS_PER_BLOCK, R, VEC_SIZE>(csrmv_input, &flat_info);
    // compute: warm up
    for (int i = 0; i < 10; i++) {
      flat_wrapper<FLAT_STAGE_COMPUTE, THREADS_PER_BLOCK, R, VEC_SIZE>(csrmv_input, &flat_info);
    }

    // compute:
    for (auto state : states) {
      flat_wrapper<FLAT_STAGE_COMPUTE, THREADS_PER_BLOCK, R, VEC_SIZE>(csrmv_input, &flat_info);
    }
    // destroy.
    flat_wrapper<FLAT_STAGE_DESTROY, THREADS_PER_BLOCK, R, VEC_SIZE>(csrmv_input, &flat_info);

    const size_t nnz = csrmv_input.csr_mat.nnz;
    const size_t rows = csrmv_input.csr_mat.rows;
    const size_t cols = csrmv_input.csr_mat.cols;
    const size_t mem_size = nnz * sizeof(double) + nnz * sizeof(int) + (rows + 1) * sizeof(int) +
                            (cols) * sizeof(double) + (cols) * sizeof(double) * 2;

    states.counters["Bandwidth"] = usparse::tuner::Counter(
        mem_size * states.iterations(), usparse::tuner::Counter::kIsRate, usparse::tuner::Counter::kIs1024);
  };
};

USPARSE_TUNER_DEFINE_TUNE_RANGE(uTuneFlatFP64) {
  using HIP_THREADS_PER_BLOCK = usparse::tuner::ParamRange<int, 256, 512>;
  using FLAT_R = usparse::tuner::ParamRange<int, 1, 2, 3, 4>;

  using FLAT_VEC_SIZE = usparse::tuner::ParamRange<int, 1, 2, 4, 8, 16, 32, 64>; // 64 is only for AMD GPUs

  constexpr int ParamGroups = 3;
  const std::array<std::string, ParamGroups> ParameterNames = {"THREADS", "R", "VEC_SIZE"};

  using ConfigType = usparse::tuner::Config<utuner::csrmv_tuner_input<int, double>>;
  auto config = new ConfigType();
  config->setIterations(10);
  config->setArgs({*global_csrmv_tuner_input_ptr});

  USPARSE_TUNER_LAUNCH(uTuneFlatFP64, ConfigType, config, ParamGroups, ParameterNames, HIP_THREADS_PER_BLOCK, FLAT_R,
                       FLAT_VEC_SIZE);
}
