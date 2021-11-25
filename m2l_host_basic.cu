#ifndef M2L_HOST_BASIC_CU
#define M2L_HOST_BASIC_CU

/* Include header file */
#include "m2l_host.h"

/* Include kernel */
#include "m2l_kern_basic.cu"

/* Parameters for host */
#ifndef GROUP_SIZE_BA       // Maximum number of cells per cell group
#define GROUP_SIZE_BA 32768 // equal to the number of cells at level 5
#endif
#if (GROUP_SIZE_BA > 65535) // for CC 1.2-1.3; Max# is given by deviceProp.maxGridSize[0]
#error GROUP_SIZE_BA is too large.
#endif

__host__ void m2l_host_basic(int level_start, int level_end, int r, real *L, real *K, real *M,
			     int pitch_interaction, int *interaction, int *iinter, int *Kindex)
{
  /* Initialize timers */
  eventTimerType timer_m2l_all;
  initEventTimer(&timer_m2l_all);
  startEventTimer(&timer_m2l_all);
  
  eventTimerType timer_m2l_kernel, timer_m2l_set, timer_m2l_get;
  initEventTimer(&timer_m2l_kernel);
  initEventTimer(&timer_m2l_set);
  initEventTimer(&timer_m2l_get);

  /* Compute the indices of starting and final cells for each level */
  int *cellsta = (int *)malloc((level_end + 1) * sizeof(int)); // cellsta[0:level_end]
  int *cellend = (int *)malloc((level_end + 1) * sizeof(int)); // cellend[0:level_end]
  m2l_aux_comp_cellsta_cellend(cellsta, cellend, level_end);
  
  /* Number of all the cells in level 0 to level_end */
  int nallcell = (POW8(level_end + 1) - 1) / 7;

  /* Allocate K-matrix, M-vector, and interaction-list on device and
     copy them to device */
  real *d_K, *d_M;
  int *d_interaction, *d_iinter, *d_Kindex;
  CSC(cudaMalloc((void **)&d_K, 316 * r * r * sizeof(real)));
  CSC(cudaMalloc((void **)&d_M, nallcell * r * sizeof(real)));
  CSC(cudaMalloc((void **)&d_interaction, nallcell * pitch_interaction * sizeof(int)));
  CSC(cudaMalloc((void **)&d_iinter, nallcell * sizeof(int)));
  CSC(cudaMalloc((void **)&d_Kindex, nallcell * pitch_interaction * sizeof(int)));

  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_K, K, 316 * r * r * sizeof(real), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_M, M, nallcell * r * sizeof(real), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_interaction, interaction, nallcell * pitch_interaction * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_iinter, iinter, nallcell * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_Kindex, Kindex, nallcell * pitch_interaction * sizeof(int), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);


  /* Loop over levels */
  for (int level = level_start; level <= level_end; level ++) {

    /* Number of cells in this level */
    int nc = cellend[level] - cellsta[level] + 1;
    
    /* Number of cell groups in this level */
    int ngrp = (nc / GROUP_SIZE_BA) + (nc % GROUP_SIZE_BA == 0 ? 0 : 1);

    /* Loop over cell groups */
    for (int grp = 0; grp < ngrp; grp ++) {

      /* Indices of the first and last cells in this cell group */
      int Fsta = cellsta[level] + GROUP_SIZE_BA * grp;
      int Fend = min(Fsta + GROUP_SIZE_BA - 1, cellend[level]);

      /* Number of cells in this cell group */
      int ncg = Fend - Fsta + 1;

      /* Allocate L-vector for this cell group on device */
      real *d_L;
      CSC(cudaMalloc((void **)&d_L, ncg * r * sizeof(real)));

      /* Setup grid and thread-blocks */
      dim3 Db(r);
      dim3 Dg(ncg);
      CHECK_CONFIGURATION(Dg, Db);

      /* Invoke M2L kernel */
      if (r == 32) {
	startEventTimer(&timer_m2l_kernel);
	m2l_kern_basic_r32<<<Dg, Db>>>(d_L, d_K, d_M, Fsta, pitch_interaction, d_interaction, d_iinter, d_Kindex);
	stopEventTimer(&timer_m2l_kernel);
      } else if (r == 256) {
	startEventTimer(&timer_m2l_kernel);
	m2l_kern_basic_r256<<<Dg, Db>>>(d_L, d_K, d_M, Fsta, pitch_interaction, d_interaction, d_iinter, d_Kindex);
	stopEventTimer(&timer_m2l_kernel);
      } else {
	abort();
      }

      /* Copy L-vector from device */
      startEventTimer(&timer_m2l_get);
      CSC(cudaMemcpy(&(L[r * Fsta]), d_L, ncg * r * sizeof(real), cudaMemcpyDeviceToHost));
      stopEventTimer(&timer_m2l_get);
      
      /* Free L-vector for this cell group */
      CSC(cudaFree(d_L));
      
    }
  }

  /* Clean up */
  CSC(cudaFree(d_K));
  CSC(cudaFree(d_M));
  CSC(cudaFree(d_interaction));
  CSC(cudaFree(d_iinter));
  CSC(cudaFree(d_Kindex));

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

#endif /* M2L_HOST_BASIC_CU */
