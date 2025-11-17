//
// Created by genshen to 2025-08-13.
//

#ifndef USPARSE_GMRES_OPTIMIZATION_CONFIG_H
#define USPARSE_GMRES_OPTIMIZATION_CONFIG_H

namespace usparse {
  constexpr int GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_NONE = 0;
  constexpr int GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1 = 1; // only fusion inner loop
  constexpr int GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2 = 2; // make full fusion.

  constexpr int GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION = GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2;

  constexpr int GMRES_OPT_MGS_FUSION_PARAMETER_STRATEGY_CLASSIC = 1;
  constexpr int GMRES_OPT_MGS_FUSION_PARAMETER_STRATEGY_COMPLEX = 2;
  constexpr int GMRES_OPT_MGS_FUSION_PARAMETER_STRATEGY = GMRES_OPT_MGS_FUSION_PARAMETER_STRATEGY_COMPLEX;

  // orthogonalization Methodss
  constexpr int GMRES_OPT_USE_MGS = 0; // Modified Gram-Schmidt
  constexpr int GMRES_OPT_USE_GS = 1;  // Gram-Schmidt

  constexpr int GMRES_ORTHO_USE = GMRES_OPT_USE_MGS;

  constexpr bool MGS_ALIGN_MEM = true; // malloc alligned memory for r0 and V.

  // precision verification
  constexpr bool GMRES_OPT_CLOSE_VERIFICATION = false;
  constexpr bool GMRES_OPT_OPEN_VERIFICATION = true;

  constexpr bool GMRES_OPT_VERIFY_SWITCH = GMRES_OPT_OPEN_VERIFICATION;

  constexpr bool GMRES_USE_HOST_TRI_SOLVER = false;
  constexpr bool GMRES_USE_HOST_ROTATION = false;

  constexpr inline bool check_mem_align_enabled() {
    // only enabled in leve2 and MGS method.
    return (GMRES_ORTHO_USE == GMRES_OPT_USE_MGS && (usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION ==
            usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2))? (MGS_ALIGN_MEM): false;
  }

  // use aligned memory for r0 and matrix V.
  inline std::size_t get_r0_v_mem_len(const std::size_t N) {
    constexpr int mem_align_unit = 512 * 8 * 8;
    const int r0_v_mem_len_if_align = (N / mem_align_unit + 1) * mem_align_unit;
    const int R0_V_MEM_LEN = check_mem_align_enabled() ? (r0_v_mem_len_if_align) : N;
    return R0_V_MEM_LEN;
  }
} // namespace usparse

#endif // USPARSE_GMRES_OPTIMIZATION_CONFIG_H
