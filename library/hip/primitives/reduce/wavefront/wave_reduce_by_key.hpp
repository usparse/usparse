/**
 * Created by genshen on 2025/9/2.
 *
 */

#include "building_config.h"
#include "common/platforms/cuda/cuda_utils.hpp"

#ifdef __USPARSE_BUILD_BACKEND_ROCM__
#define __ballot_sync(mask, pred) __ballot(pred)
#define __any_sync(mask, pred) __any(pred)
#endif

namespace usparse {
  namespace primitives {
    namespace reduce {
      /**
       * reduce by key. One thread contributes one value.
       * Currently, mask is unused.
       * And finally store the reduced result to @prarm result in the device memory.
       */
      template <typename INDEX_TYPE, typename DATA_TYPE, typename KEY_TYPE, int WAVE_SIZE>
      __device__ __forceinline__ void wavefront_reduce_by_key(const int lane_id, DATA_TYPE value, const KEY_TYPE key,
                                                              DATA_TYPE *result, unsigned mask) {
        const KEY_TYPE left_lane_key = __shfl_up(key, 1);
        const bool is_root = (left_lane_key != key) || (lane_id == 0);
        // https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#warp-vote-and-ballot-functions
        // we use 60-bit __ballot in HIP:
        using WAVE_BIT_FLAG_TYPE = unsigned long long;
        WAVE_BIT_FLAG_TYPE roots_flag =
            __ballot_sync(0xffffffff, is_root); // on CUDA, it converts uint32 to uint64 (it is ok).

        // calculate the distance of current lane to the root lane of the next row.
        int next_row_root_dist = 0;
#ifdef USPARSE_HIP_CLANG
// TODO:
// next_row_root_dist = __lastbit_u32_u64(roots_flag >> (lane_id+1)); // for WAVE 64. index is based on -1.
#endif
#ifndef USPARSE_HIP_CLANG
        next_row_root_dist = __ffsll((roots_flag >> (lane_id + 1)));
        if (next_row_root_dist == 0) { // for the last lanes. They have no next row.
          next_row_root_dist = WAVE_SIZE - lane_id;
        }
#endif

        // Now, we divide the wave into several pieces by lane of `is_root = true`.
        // we can reduce values in each pieces into root lanes.

        // calculate `pd = powerOf(next_row_root_dist - 1)`, which is the number of reductions performed on each lane.
        // if `pd` is zero, this lane can not perform reduction (the data in `temp` should be discard).
        int pd = next_row_root_dist - 1;
        for (int shift = 1; __any_sync(0xffffffff, pd); shift *= 2) {
          const DATA_TYPE temp = __shfl_down(value, shift); // send to lower lane
          // which is the same as:
          // shift <= next_row_root_dist - 1
          // aka. (next_row_root_dist - 1)/(2^k) >= 1;
          // aka. pd >= 1
          // aka. pd > 0 (aka. pd != 0)
          if (pd) {
            value += temp; // receive from higher lane
          }
          pd /= 2;
        }

        if (is_root) {
          atomicAdd(result + key, value); // a row may span multiple blocks.
        }
        return;
      }
    } // namespace reduce
  } // namespace primitives
} // namespace usparse
