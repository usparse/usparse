#ifndef GMRES_UTILS_HPP
#define GMRES_UTILS_HPP

#include <iomanip>

#define HIP_CHECK(stat)                                                                                                \
  {                                                                                                                    \
    hipError_t err = stat;                                                                                             \
    if (err != hipSuccess) {                                                                                           \
      std::cerr << "Error: HIP reports " << hipGetErrorString(err) << std::endl;                                       \
      std::cerr << "Raised in file " << __FILE__ << "#" << __func__ << "#line " << __LINE__ << std::endl;              \
      std::abort();                                                                                                    \
    }                                                                                                                  \
  }

#endif // GMRES_UTILS_HPP
