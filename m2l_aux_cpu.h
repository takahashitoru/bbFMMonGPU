#ifndef M2L_AUX_CPU_H
#define M2L_AUX_CPU_H

#include "m2l_aux.h"

#ifdef __cplusplus
extern "C" {
#endif
  void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1_CPU(int r, real3 *center, real L0, int level, int B, real *Lanother, real *L);
  void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1_CPU2(int r, real3 *center, real L0, int level, int B, real *Lanother, real *L);
  void m2l_aux_convert_M_to_Manother_for_ij_blocking_col1_CPU(int r, int level_start, int level_end, real3 *center, real L0, real *M, real *Manother);
#ifdef __cplusplus
}
#endif

#endif /* M2L_AUX_CPU_H */
