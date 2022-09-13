#ifndef SPGEMM_KOKKOS_ALIGN_H
#define SPGEMM_KOKKOS_ALIGN_H

#define ALIGN_256_BYTES(n) ((n / 256 + (n % 256 == 0 ? 0 : 1)) * 256)

#endif // SPGEMM_KOKKOS_ALIGN_H