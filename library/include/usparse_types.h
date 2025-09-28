//
// Created by genshen on 2025/05/08.
//

#ifndef USPARSE_HIP_USPARSE_TYPES_H
#define USPARSE_HIP_USPARSE_TYPES_H

namespace usparse {
  enum usparse_operation { operation_none = 0, operation_transpose = 1 };

  enum usparse_status {
    usparse_status_success = 0,
    usparse_status_not_implemented = 1,
    usparse_status_invalid_pointer = 2,
    usparse_status_invalid_size = 3,
  };

  using usparse_int = int;
  using usparse_uint = uint;
  using usparse_fp32 = float;
  using usparse_fp64 = double;
} // namespace usparse

#endif // USPARSE_HIP_USPARSE_TYPES_H
