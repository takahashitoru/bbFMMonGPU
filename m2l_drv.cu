#ifndef M2L_DRV_CU
#define M2L_DRV_CU

#include "m2l_drv.h"

#if defined(ENABLE_USE_PARENT_LEAVES_ARRAYS)
__host__ int m2l_drv(int r, real *L, real *K, real *M,
                     int minlev, int maxlev,
                     real L0, real3 *center, int *parent, int *leaves,
                     int *ineigh, int pitch_neighbors, int *neighbors,
                     int *iinter, int pitch_interaction, int *interaction,
                     int scheme)
#else
__host__ int m2l_drv(int r, real *L, real *K, real *M,
                     int minlev, int maxlev,
                     real L0, real3 *center,
                     int *ineigh, int pitch_neighbors, int *neighbors,
                     int *iinter, int pitch_interaction, int *interaction,
                     int scheme)
#endif
{
  int nallcell = (POW8(maxlev + 1) - 1) / 7;
  int *Ktable = (int *)malloc(343 * sizeof(int));
  m2l_aux_comp_Ktable(Ktable);


  if (scheme == M2L_SCHEME_BA) {

    int *Kindex = (int *)malloc(pitch_interaction * nallcell * sizeof(int));
    m2l_aux_comp_Kindex(maxlev, L0, (real3 *)center, iinter, pitch_interaction, interaction, Ktable, Kindex);

    m2l_host_basic(minlev, maxlev, r, L, K, M, pitch_interaction, interaction, iinter, Kindex);

    free(Kindex);

  } else {

    int pitch_sourceclusters = PITCH_SOURCECLUSTERS;
    const int nallcluster = (POW8(maxlev) - 8) / 7; // minlev=2 is supposed for m2l_host_{sibling,cluster}_blocking.cu
    int *sourceclusters = (int *)malloc(pitch_sourceclusters * nallcluster * sizeof(int));
#if defined(ENABLE_USE_PARENT_LEAVES_ARRAYS)
    m2l_aux_comp_sourceclusters(minlev, maxlev, L0, (real3 *)center, parent, leaves,
				ineigh, pitch_neighbors, neighbors,
				pitch_sourceclusters, sourceclusters);
#else
    m2l_aux_comp_sourceclusters(minlev, maxlev, L0, (real3 *)center,
				ineigh, pitch_neighbors, neighbors,
				pitch_sourceclusters, sourceclusters);
#endif

    if (scheme == M2L_SCHEME_SI) {

      m2l_host_sibling_blocking(minlev, maxlev, r, L, K, M,
				pitch_sourceclusters, sourceclusters, Ktable);
      
    } else if (scheme == M2L_SCHEME_CL) {
      
      m2l_host_cluster_blocking(minlev, maxlev, r, L, K, M,
				pitch_sourceclusters, sourceclusters, Ktable);
      
    } else if (scheme == M2L_SCHEME_IJ) {
      
      m2l_host_sibling_blocking(minlev, LEVEL_SWITCH_IJ, r, L, K, M,
				pitch_sourceclusters, sourceclusters, Ktable);
      
      if (maxlev > LEVEL_SWITCH_IJ) { // early exit

	double tmp = kernel_exec_time; // save the execution time for lower levels

	m2l_host_ij_blocking(LEVEL_SWITCH_IJ + 1, maxlev, r, L, K, M,
			     Ktable, center, L0);
      
	kernel_exec_time += tmp; // save the exectution time for both lower and higher levels

      }
      
    } else {
      
      free(Ktable);
      free(sourceclusters);

      return M2L_EXIT_FAIL;

    }

    free(sourceclusters);
    
  }
  
  free(Ktable);

  return M2L_EXIT_SUCCESS;
}

#endif /* M2L_DRV_CU */
