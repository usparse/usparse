//
// Created by genshen on 2022/5/25.
//

#include "api/spgemm_hip_api.h"

int main(int argc, char **argv) {
  // todo:
  sparse_gemm_hip<double, int>(operation_none, operation_none, 1.0, 0.0, 100, 100, 100, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr);
  return 0;
}
