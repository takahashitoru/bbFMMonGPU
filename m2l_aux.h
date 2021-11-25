#ifndef M2L_AUX_H
#define M2L_AUX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "debugmacros.h"

#ifndef NULL_KINDEX
#define NULL_KINDEX (- 1)
#endif

#ifndef NULLCELL
#define NULLCELL (- 1)
#endif

#ifndef NULL_CELL
#define NULL_CELL NULLCELL
#endif

#ifndef GET_PARENT_INDEX
#define GET_PARENT_INDEX(level, A) ( (A) % POW8((level) - 1) )
#endif

#ifndef GET_CHILD_INDEX
#define GET_CHILD_INDEX(A, i) ( 8 * (A) + 1 + (i) )
#endif

#ifndef POW2
#define POW2(n) (1 << (n))
#endif

#ifndef POW8
#define POW8(i) (1 << (3 * (i)))
#endif

#include "vec234.h"
#include "real.h"

#ifdef __cplusplus
extern "C" {
#endif
  void m2l_aux_comp_Ktable(int *Ktable);
  void m2l_aux_convert_K_to_Kanother_for_cluster_blocking(int r, int P, int Q, real *K, real *Kanother);
  void m2l_aux_convert_K_to_Kanother_for_ij_blocking(int r, int *Ktable, real *K, real4 *Kanother);
  int m2l_aux_get_number_of_real_and_ghost_cells_for_ij_blocking(int level_start, int level_end);
  int m2l_aux_get_starting_index_of_Manother_for_ij_blocking(int r, int level_start, int level);
  void m2l_aux_convert_M_to_Manother_for_ij_blocking(int r, int level_start, int level_end, real3 *center, real L0, real *M, real *Manother);
  void m2l_aux_convert_Lanother_to_L_for_ij_blocking(int r, real3 *center, real L0, int level, int B, real4 *Lanother, real *L);
  void m2l_aux_comp_cellsta_cellend(int *cellsta, int *cellend, int level);
  double m2l_aux_comp_kernel_performance_in_Gflops(int r, int level_start, int level_end, int *cellsta, int *cellend, int *iinter, double time);
#if defined(ENABLE_USE_PARENT_LEAVES_ARRAYS)
  void m2l_aux_comp_sourceclusters(int minlev, int maxlev, real L0, real3 *center, int *parent, int *leaves, int *ineigh, int pitch_neighbors, int *neighbors, int pitch_sourceclusters, int *sourceclusters);
#else
  void m2l_aux_comp_sourceclusters(int minlev, int maxlev, real L0, real3 *center, int *ineigh, int pitch_neighbors, int *neighbors, int pitch_sourceclusters, int *sourceclusters);
#endif
  int m2l_aux_get_a_Kindex(int *Ktable, int vx, int vy, int vz);
  void m2l_aux_comp_Kindex(int maxlev, real L0, real3 *center, int *iinter, int pitch_interaction, int *interaction, int *Ktable, int *Kindex);

  void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row4_col1(int r, int *Ktable, real *K, real4 *Kanother);
  void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row1_col2(int r, int *Ktable, real *K, real2 *Kanother);
  void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row8_col1(int r, int *Ktable, real *K, real8 *Kanother);
  void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row16_col1(int r, int *Ktable, real *K, real16 *Kanother);
  void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row8_col2(int r, int *Ktable, real *K, real8x2 *Kanother);
  void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row4_col4(int r, int *Ktable, real *K, real4x4 *Kanother);
  void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row4_col2(int r, int *Ktable, real *K, real4x2 *Kanother);
  void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row1_col1(int r, int *Ktable, real *K, real *Kanother);

  void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1(int r, real3 *center, real L0, int level, int B, real *Lanother, real *L);
  void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row4(int r, real3 *center, real L0, int level, int B, real4 *Lanother, real *L);
  void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row8(int r, real3 *center, real L0, int level, int B, real8 *Lanother, real *L);
  void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row16(int r, real3 *center, real L0, int level, int B, real16 *Lanother, real *L);
  void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1_CPU(int r, real3 *center, real L0, int level, int B, real *Lanother, real *L);

  void m2l_aux_convert_M_to_Manother_for_ij_blocking_col1(int r, int level_start, int level_end, real3 *center, real L0, real *M, real *Manother);
  void m2l_aux_convert_M_to_Manother_for_ij_blocking_col2(int r, int level_start, int level_end, real3 *center, real L0, real *M, real2 *Manother);
  void m2l_aux_convert_M_to_Manother_for_ij_blocking_col4(int r, int level_start, int level_end, real3 *center, real L0, real *M, real4 *Manother);
  void m2l_aux_convert_M_to_Manother_for_ij_blocking_col1_CPU(int r, int level_start, int level_end, real3 *center, real L0, real *M, real *Manother);



#ifdef __cplusplus
}
#endif

#endif /* M2L_AUX_H */
