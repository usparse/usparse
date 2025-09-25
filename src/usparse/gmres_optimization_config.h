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

  // orthogonalization Methodss
  constexpr int GMRES_OPT_USE_MGS = 0; // Modified Gram-Schmidt
  constexpr int GMRES_OPT_USE_GS = 1; // Gram-Schmidt

  constexpr int GMRES_ORTHO_USE = GMRES_OPT_USE_GS;

  // precision verification
  constexpr bool GMRES_OPT_CLOSE_VERIFICATION = false;
  constexpr bool GMRES_OPT_OPEN_VERIFICATION = true;

  constexpr bool GMRES_OPT_VERIFY_SWITCH = GMRES_OPT_OPEN_VERIFICATION;
}

#endif // USPARSE_GMRES_OPTIMIZATION_CONFIG_H
