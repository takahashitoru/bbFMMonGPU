#ifndef M2L_HOST_SIBLING_BLOCKING_CU
#define M2L_HOST_SIBLING_BLOCKING_CU

/* Include header file */
#include "m2l_host.h"

/* Include kernel */
#include "m2l_kern_sibling_blocking.cu"

/* Parameters for host */
#ifndef GROUP_SIZE_SI       // Maximum number of cells per cell group
#define GROUP_SIZE_SI 32768 // equal to the number of cells at level 5
#endif
#if (GROUP_SIZE_SI > 65535) // for CC 1.2-1.3; Max# is given by deviceProp.maxGridSize[0]
#error GROUP_SIZE_SI is too large.
#endif

#if (GROUP_SIZE_SI % 8 != 0)
#error GROUP_SIZE_SI must be a multiple of eight so that any clusters are not chopped.
#endif

__host__ void m2l_host_sibling_blocking(int level_start, int level_end, int r, real *L, real *K, real *M,
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


  /* Allocate data on device */
  real *d_K, *d_M;
  int *d_sourceclusters, *d_Ktable;
  CSC(cudaMalloc((void **)&d_K, 316 * r * r * sizeof(real)));
  CSC(cudaMalloc((void **)&d_M, nallcell * r * sizeof(real)));
  CSC(cudaMalloc((void **)&d_sourceclusters, nallcluster * pitch_sourceclusters * sizeof(int)));
  CSC(cudaMalloc((void **)&d_Ktable, 343 * sizeof(int)));
    
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_K, K, 316 * r * r * sizeof(real), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_M, M, nallcell * r * sizeof(real), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_sourceclusters, sourceclusters, nallcluster * pitch_sourceclusters * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_Ktable, Ktable, 343 * sizeof(int), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);

  /* Loop over levels */
  for (int level = level_start; level <= level_end; level ++) {
      
    /* Number of cells in this level */
    int nc = cellend[level] - cellsta[level] + 1;
      
    /* Number of cell groups in this level */
    int ngrp = (nc / GROUP_SIZE_SI) + (nc % GROUP_SIZE_SI == 0 ? 0 : 1);
      
    ////////////////////////////////////////////////////////////////////////////////
    DBG("level=%d nc=%d ngrp=%d\n", level, nc, ngrp);
    ////////////////////////////////////////////////////////////////////////////////

    /* Loop over cell groups */
    for (int grp = 0; grp < ngrp; grp ++) {
	
      /* Indices of the first and last cells in this cell group */
      int Fsta = cellsta[level] + GROUP_SIZE_SI * grp;
      int Fend = min(Fsta + GROUP_SIZE_SI - 1, cellend[level]);

      /* Number of cells in this cell group */
      int ncg = Fend - Fsta + 1;
	
      /* Index of the first cluster in this cell group */
      int FCsta = (Fsta - cellsta[2]) / 8;
	
      ////////////////////////////////////////////////////////////////////////////////
      DBG("grp=%d Fsta=%d Fend=%d ncg=%d FCsta=%d\n", grp, Fsta, Fend, ncg, FCsta);
      ////////////////////////////////////////////////////////////////////////////////

      /* Allocate L-vector for this cell group on device */
      real *d_L;
      CSC(cudaMalloc((void **)&d_L, ncg * r * sizeof(real)));
	
      /* Initialise L-vector by zero */
#if(0) // same
      ////////////////////////////////////////////////////////////////////////////////
      real *zero = (real *)calloc(ncg * r, sizeof(real));
      startEventTimer(&timer_m2l_set);
      CSC(cudaMemcpy(d_L, zero, ncg * r * sizeof(real), cudaMemcpyHostToDevice));
      stopEventTimer(&timer_m2l_set);
      free(zero);
      ////////////////////////////////////////////////////////////////////////////////
#else
      CSC(cudaMemset(d_L, 0, ncg * r * sizeof(real)));
#endif
	
      /* Setup grid and thread-blocks */
      dim3 Db(r);
      dim3 Dg(ncg / 8); // number of clusters in this group
      CHECK_CONFIGURATION(Dg, Db);

      /* Invoke M2L kernel */
      if (r == 32) {
	startEventTimer(&timer_m2l_kernel);
	m2l_kern_sibling_blocking_r32<<<Dg, Db>>>(d_L, d_K, d_M, d_sourceclusters, d_Ktable, FCsta);
	stopEventTimer(&timer_m2l_kernel);
      } else if (r == 256) {
	startEventTimer(&timer_m2l_kernel);
	m2l_kern_sibling_blocking_r256<<<Dg, Db>>>(d_L, d_K, d_M, d_sourceclusters, d_Ktable, FCsta);
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
  CSC(cudaFree(d_sourceclusters));
  CSC(cudaFree(d_Ktable));

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

#endif /* M2L_HOST_SIBLING_BLOCKING_CU */


