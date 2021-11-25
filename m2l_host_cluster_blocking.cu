#ifndef M2L_HOST_CLUSTER_BLOCKING_CU
#define M2L_HOST_CLUSTER_BLOCKING_CU

/* Include header file */
#include "m2l_host.h"

/* Parameters for kernel */
#ifndef CHUNK_SIZE_CL
#define CHUNK_SIZE_CL 2 // Currently, only 2
#endif
#ifndef TILE_SIZE_ROW_CL
#define TILE_SIZE_ROW_CL 32 // Currently, only 32
#endif
#ifndef TILE_SIZE_COLUMN_CL
#define TILE_SIZE_COLUMN_CL 32 // Currently, only 32
#endif

/* Include kernel */
#include "m2l_kern_cluster_blocking.cu"

/* Parameters for host */
#ifndef GROUP_SIZE_CL       // Maximum number of cells per cell group
#define GROUP_SIZE_CL 32768 // equal to the number of cells at level 5
#endif
#if (GROUP_SIZE_CL > 65535) // for CC 1.2-1.3; Max# is given by deviceProp.maxGridSize[0]
#error GROUP_SIZE_CL is too large.
#endif

__host__ void m2l_host_cluster_blocking(int level_start, int level_end, int r, real *L, real *K, real *M,
					int pitch_sourceclusters, int *sourceclusters, int *Ktable)
{
  /* Initialize timers */
  eventTimerType timer_m2l_all;
  initEventTimer(&timer_m2l_all);
  startEventTimer(&timer_m2l_all);
  
  eventTimerType timer_m2l_kernel, timer_m2l_set, timer_m2l_get;
  initEventTimer(&timer_m2l_kernel);
  initEventTimer(&timer_m2l_set);
  initEventTimer(&timer_m2l_get);
  
  /* Compute the indices of starting and final cells for eac level */
  int *cellsta = (int *)malloc((level_end + 1) * sizeof(int)); // cellsta[0:level_end]
  int *cellend = (int *)malloc((level_end + 1) * sizeof(int)); // cellend[0:level_end]
  m2l_aux_comp_cellsta_cellend(cellsta, cellend, level_end);

  /* Number of all the cells in level 0 to level_end, i.e. sum_{i=0}^level_end 8^i */
  int nallcell = (POW8(level_end + 1) - 1) / 7;

  /* Number of all the clusters in level 2 to level_end, i.e. sum_{i=2}^level_end 8^{i-1} */
  int nallcluster = (POW8(level_end) - 8) / 7;

  /* Constants */
  int B = CHUNK_SIZE_CL;
  int BBB8 = B * B * B * 8; // number of cells per chunk of size B

  if (GROUP_SIZE_CL % BBB8 != 0) {
    INFO("GROUP_SIZE_CL (%d) must be a multiple of 8*B^3 (%d) so that any chunks of size B are not chopped.\n", GROUP_SIZE_CL, BBB8);
    abort();
  }

  int P = r / TILE_SIZE_ROW_CL;    // number of row tiles
  int Q = r / TILE_SIZE_COLUMN_CL; // number of column tiles

  /* Convert K-matrix to another one */
  real *Kanother = (real *)malloc(316 * r * r * sizeof(real));
  m2l_aux_convert_K_to_Kanother_for_cluster_blocking(r, P, Q, K, Kanother);

  /* Allocate data on device */
  real *d_Kanother, *d_M;
  int *d_sourceclusters, *d_Ktable;
  CSC(cudaMalloc((void **)&d_Kanother, 316 * r * r * sizeof(real)));
  CSC(cudaMalloc((void **)&d_M, nallcell * r * sizeof(real)));
  CSC(cudaMalloc((void **)&d_sourceclusters, nallcluster * pitch_sourceclusters * sizeof(int)));
  CSC(cudaMalloc((void **)&d_Ktable, 343 * sizeof(int)));
    
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Kanother, Kanother, 316 * r * r * sizeof(real), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_M, M, nallcell * r * sizeof(real), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_sourceclusters, sourceclusters, nallcluster * pitch_sourceclusters * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_Ktable, Ktable, 343 * sizeof(int), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);

  /* Loop over levels */
  for (int level = level_start; level <= level_end; level ++) {
      
    /* Number of cells in this level */
    int nc = cellend[level] - cellsta[level] + 1;
      
    /* Number of cell groups in this level */
    int ngrp = (nc / GROUP_SIZE_CL) + (nc % GROUP_SIZE_CL == 0 ? 0 : 1);
      
    /* Loop over cell groups */
    for (int grp = 0; grp < ngrp; grp ++) {
	
      /* Indices of the first and last cells in this cell group */
      int Fsta = cellsta[level] + GROUP_SIZE_CL * grp;
      int Fend = min(Fsta + GROUP_SIZE_CL - 1, cellend[level]);

      /* Number of cells in this cell group */
      int ncg = Fend - Fsta + 1;

      /* Index of the first chunk in this cell group */
      int FCHsta = (Fsta - cellsta[2]) / BBB8;

      /* Allocate L-vector for this cell group on device */
      real *d_L;
      CSC(cudaMalloc((void **)&d_L, ncg * r * sizeof(real)));

      /* Initialise L-vector by zero (necessary) */
      CSC(cudaMemset(d_L, 0, ncg * r * sizeof(real)));

      /* Setup grid and thread-blocks */
      dim3 Db(r / P, 8);
      dim3 Dg(ncg / BBB8); // number of chunks of size B in this group
      CHECK_CONFIGURATION(Dg, Db);
      
      /* Invoke M2L kernel */
      if (r == 32 && B == 2 && P == 1 && Q == 1) {
	startEventTimer(&timer_m2l_kernel);
	m2l_kern_cluster_blocking_r32b2p1q1<<<Dg, Db>>>(d_L, d_Kanother, d_M, d_sourceclusters, d_Ktable, FCHsta);
	stopEventTimer(&timer_m2l_kernel);
      } else if (r == 256 && B == 2 && P == 8 && Q == 8) {
	startEventTimer(&timer_m2l_kernel);
	m2l_kern_cluster_blocking_r256b2p8q8<<<Dg, Db>>>(d_L, d_Kanother, d_M, d_sourceclusters, d_Ktable, FCHsta);
	stopEventTimer(&timer_m2l_kernel);
      } else {
	abort();
      }
	
      /* Copy field values from device */
      startEventTimer(&timer_m2l_get);
      CSC(cudaMemcpy(&(L[r * Fsta]), d_L, ncg * r * sizeof(real), cudaMemcpyDeviceToHost));
      stopEventTimer(&timer_m2l_get);
	
      /* Free */
      cudaFree(d_L);

    }
  }

  /* Clean up */
  CSC(cudaFree(d_Kanother));
  CSC(cudaFree(d_M));
  CSC(cudaFree(d_sourceclusters));
  CSC(cudaFree(d_Ktable));

  free(Kanother);
  free(cellsta);
  free(cellend);

  /* Save the kernel execution time in the global variable */
  kernel_exec_time = getEventTimer(&timer_m2l_kernel);

  /* Finalize timers */
  INFO("timer_m2l_kernel = %14.7e\n", getEventTimer(&timer_m2l_kernel));
  INFO("timer_m2l_set = %14.7e\n", getEventTimer(&timer_m2l_set));
  INFO("timer_m2l_get = %14.7e\n", getEventTimer(&timer_m2l_get));
  finalizeEventTimer(&timer_m2l_kernel);
  finalizeEventTimer(&timer_m2l_set);
  finalizeEventTimer(&timer_m2l_get);

  stopEventTimer(&timer_m2l_all);
  INFO("timer_m2l_all = %14.7e\n", getEventTimer(&timer_m2l_all));
  finalizeEventTimer(&timer_m2l_all);
}

#endif /* M2L_HOST_CLUSTER_BLOCKING_CU */

