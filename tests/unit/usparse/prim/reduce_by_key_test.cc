//
// Created by genshen to 2025-07-13.
//

#include <cmath>
#include <random>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "tests_utils.hpp"
#include "primitives/reduce_by_key/reduce_by_key.hpp"

TEST(prim_test, reduce_by_key) {
  hipStream_t stream = hipStreamDefault;

  using kv = usparse::primitives::reduce_by_key::KV<int, double>;

  const int HIP_BLOCKS = 8194;
  double *y;
  kv *kvs;
  hipMalloc(&y, HIP_BLOCKS * sizeof(double));
  hipMalloc(&kvs, 2 * HIP_BLOCKS * sizeof(kv));  // size is doubled as temp buffer

  for (int i = 0; i < HIP_BLOCKS; i++) {
    double val = static_cast<double>(i);
    y[i] = 0.0;
    kvs[i].key = 1;
    kvs[i].value = 1.0;
  }

  usparse::primitives::reduce_by_key::device::flat_reduce_carry_out<int, double, int, 512>(stream, y, kvs, HIP_BLOCKS);
  hipStreamSynchronize(stream);
  printf("result is: %f\n", y[1]);
  EXPECT_EQ(y[1], HIP_BLOCKS);
}
