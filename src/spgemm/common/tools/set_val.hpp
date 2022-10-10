#ifndef SET_VAL_KERNEL_HPP
#define SET_VAL_KERNEL_HPP

namespace common {
template <typename T, T VALUE> __global__ void set_val_kernel(T *val_ptr) { *val_ptr = VALUE; }

template <typename T, T VALUE> void set_val(T *val_ptr) { (set_val_kernel<T, VALUE>)<<<1, 1>>>(val_ptr); }
} // namespace common

#endif // SET_VAL_KERNEL_HPP