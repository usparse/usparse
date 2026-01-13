//
// Created by genshen to 2026-01-13.
//

#ifndef CSRMV_TUNER_LINE_ENHAHCE_CSRMV_TUNER_INPUT
#define CSRMV_TUNER_LINE_ENHAHCE_CSRMV_TUNER_INPUT

#include "usparse.h"

namespace utuner {
  template <typename INDEX_TYPE, typename DATA_TYPE> struct csrmv_tuner_input {
    std::string mtx_path;
    const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> csr_mat;
    const DATA_TYPE *x;
    DATA_TYPE *y;
  };
} // namespace utuner

namespace usparse {
  namespace tuner {
    namespace utils {
      // convert csrmv_tuner_input to string (the string is used in the summary report)
      template <typename INDEX_TYPE, typename DATA_TYPE>
      std::string tupleToString(const std::tuple<utuner::csrmv_tuner_input<INDEX_TYPE, DATA_TYPE>> &t) {
        auto [csrmv_input] = t;
        return csrmv_input.mtx_path;
      }
    } // namespace utils
  } // namespace tuner
} // namespace usparse

extern utuner::csrmv_tuner_input<int, double> *global_csrmv_tuner_input_ptr; // for passing data to utuner

#endif // CSRMV_TUNER_LINE_ENHAHCE_CSRMV_TUNER_INPUT
