//
// Created by genshen on 2025/7/17.
//

#ifndef USPARSE_KERNELS__USPARSE_CONFIG_H
#define USPARSE_KERNELS__USPARSE_CONFIG_H

#include "building_config.h"

namespace usparse {
  constexpr int USPARSE_WAVEFRONT_SIZE = __WRAP_SIZE__;
}

#endif // USPARSE_KERNELS__USPARSE_CONFIG_H
