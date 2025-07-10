#ifndef USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_FLAG_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_FLAG_HPP

struct FlashSortSingleScanTag {};
struct FlashSortMultipleScanTag {};

template <bool IS_SINGLE_SCAN> struct FlashSortDispatchTrait;

template <> struct FlashSortDispatchTrait<true> { using Tag = FlashSortSingleScanTag; };

template <> struct FlashSortDispatchTrait<false> { using Tag = FlashSortMultipleScanTag; };

static constexpr bool is_single_vec_scan(size_t n, size_t items_per_bucket) { return n <= items_per_bucket; }

static constexpr bool is_single_wf_scan(size_t n, size_t items_per_bucket, size_t threads) {
  return threads * n <= items_per_bucket * __WF_SIZE__;
}

#endif // USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_FLAG_HPP
