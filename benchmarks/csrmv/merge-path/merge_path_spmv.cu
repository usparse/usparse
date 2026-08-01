#include "../utils/benchmark_time.h"
#include "merge_path_config.h"
#include "merge_path_partition.h"
#include "merge_path_reduction.h"
#include "merge_path_spmv.h"
#include "merge_path_update.h"
#include "merge_path_utils.h"
#include "usparse_matrix.h"
#include <common/platforms/cuda/cuda_utils.hpp>
#include <cuda_runtime.h>
#include <utility>

#include "../utils/timer_utils.h"
#include "benchmark_config.h"

template <typename DATA_TYPE, int REDUCTION_ALGORITHM, int UPDATE_ALGORITHM>
void merge_path_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta, const csr_desc<int, DATA_TYPE> h_csr_desc,
                     const csr_desc<int, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y, BenchmarkTime *bmt) {
  hip::timer::event_timer pre_timer, calc_timer, calc2_timer, destroy_timer;
  pre_timer.start();
  cudaMemset(y, 0, h_csr_desc.rows * sizeof(DATA_TYPE));

  constexpr int BLOCK_THREAD_NUM = 512;
  constexpr int ITEMS_PER_THREAD = 4;
  constexpr int ITEMS_PER_BLOCK = BLOCK_THREAD_NUM * ITEMS_PER_THREAD;
  const int GlobalBlockNum = (h_csr_desc.nnz + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
  using I = int;
  using T = DATA_TYPE;

  void *temp_storage = nullptr;
  char *remain_storage = nullptr;
  int temp_storage_bytes = 0;
  temp_storage_bytes += ALIGN_256_BYTES((GlobalBlockNum + 1) * sizeof(int));
  temp_storage_bytes += ALIGN_256_BYTES(GlobalBlockNum * sizeof(KeyValuePair<int, T>));
  temp_storage_bytes += update_temp_storage_bytes<256>(GlobalBlockNum, typename UpdateTrait<UPDATE_ALGORITHM>::type{});
  cudaMalloc(&temp_storage, temp_storage_bytes);
  remain_storage = reinterpret_cast<char *>(temp_storage);
  int *S = reinterpret_cast<int *>(remain_storage);
  remain_storage += ALIGN_256_BYTES((GlobalBlockNum + 1) * sizeof(int));
  KeyValuePair<int, T> *r = reinterpret_cast<KeyValuePair<int, T> *>(remain_storage);
  remain_storage += ALIGN_256_BYTES(GlobalBlockNum * sizeof(KeyValuePair<int, T>));

  // step 1: partition
  partition<int, 256, ITEMS_PER_BLOCK><<<512, 256>>>(d_csr_desc.row_ptr, h_csr_desc.rows, GlobalBlockNum + 1, S);
  pre_timer.stop();

  // step 2: reduction
  calc_timer.start();
  reduction<T, I, BLOCK_THREAD_NUM, ITEMS_PER_THREAD><<<GlobalBlockNum, BLOCK_THREAD_NUM>>>(
      alpha, h_csr_desc.nnz, S, r, d_csr_desc.row_ptr, d_csr_desc.col_index, d_csr_desc.values, x, y,
      typename ReductionTrait<REDUCTION_ALGORITHM>::type{});
  calc_timer.stop();

  // step 3: update
  calc2_timer.start();
  update<T, 256>(r, h_csr_desc.rows, GlobalBlockNum, y, reinterpret_cast<void *>(remain_storage),
                 typename UpdateTrait<UPDATE_ALGORITHM>::type{});
  remain_storage += update_temp_storage_bytes<256>(GlobalBlockNum, typename UpdateTrait<UPDATE_ALGORITHM>::type{});
  calc2_timer.stop(true);

  destroy_timer.start();
  cudaFree(temp_storage);
  destroy_timer.stop();
  if (bmt != nullptr) {
    bmt->set_time(pre_timer.time_use, calc_timer.time_use, calc2_timer.time_use, destroy_timer.time_use);
  }
}

// template instantiation
#define MERGE_PATH_INSTANTIATION(DATA_TYPE, REDUCE_ALGO, UPDATE_ALGO)                                                  \
  template void merge_path_spmv<DATA_TYPE, REDUCE_ALGO, UPDATE_ALGO>(                                                  \
      int trans, const DATA_TYPE alpha, const DATA_TYPE beta, const csr_desc<int, DATA_TYPE> h_csr_desc,               \
      const csr_desc<int, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y, BenchmarkTime *bmt);

MERGE_PATH_INSTANTIATION(float, Linear, SingleBlock);
MERGE_PATH_INSTANTIATION(float, Linear, LookBack);
MERGE_PATH_INSTANTIATION(float, Binary, SingleBlock);
MERGE_PATH_INSTANTIATION(float, Binary, LookBack);

MERGE_PATH_INSTANTIATION(double, Linear, SingleBlock);
MERGE_PATH_INSTANTIATION(double, Linear, LookBack);
MERGE_PATH_INSTANTIATION(double, Binary, SingleBlock);
MERGE_PATH_INSTANTIATION(double, Binary, LookBack);

#undef MERGE_PATH_INSTANTIATION
