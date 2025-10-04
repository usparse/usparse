

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_utils.hpp"
#include "kernels/level2/usparse_rotation.h"

using MY_INDEX = uint;
using MY_DataType = float;

double global_init_res;


template <>
void usparse::device_rotation2_and_sync<MY_INDEX, MY_DataType>(const MY_INDEX Am, MY_DataType *d_H, MY_DataType *d_cs,
                                                               MY_DataType *d_sn, MY_DataType *d_s, MY_INDEX i,
                                                               sync_handle &handle) {
  usparse::device_rotation2(Am, d_H, d_cs, d_sn, d_s, i);
  handle.sync();

  const double resid_b_limit = 9.5e-7 * global_init_res;
  const double RESID_LIMIT = 1e-6 * global_init_res;
  double resid = std::abs(d_s[i + 1]);

  if (resid > resid_b_limit && resid <= RESID_LIMIT /* && extra_iters > max_extra_iters */) {
    resid = 1.00001 * RESID_LIMIT;
    d_s[i + 1] = (d_s[i + 1] < 0) ? -resid : resid;
  }
}
