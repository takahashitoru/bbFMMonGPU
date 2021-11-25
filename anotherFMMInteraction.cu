#ifndef ANOTHERFMMINTERACTION_CU
#define ANOTHERFMMINTERACTION_CU

#include "bbfmm.h"
#include <cutil.h>
#include <cublas.h>
#include "eventTimer.h"
#include "debugmacros_cuda.h"

/* Alias for VER46, which is different from VER45 with respect to direct.cu */
#if defined(CUDA_VER46)
#define CUDA_VER45
#define CUDA_VER45G
#endif

/**************************************************************************/
#if defined(CUDA_VER45)
/**************************************************************************/
/* based on VER44 */

#include "m2l_drv.cu"

#include "auxAnotherFMMInteraction.h"

#if !defined(SCHEME)
#error SCHEME is undefined
#endif

void anotherFMMInteraction(anotherTree **atree, real *E, int *Ktable, 
			   real *U, real *Kweights, int n, int dof,
			   int cutoff, real homogen)
{
  eventTimerType time_all;
  initEventTimer(&time_all);
  startEventTimer(&time_all);

  int n3 = n * n * n;
  int dofn3 = dof * n3;

  int ncell = (*atree)->ncell;
  int minlev = (*atree)->minlev;
  int maxlev = (*atree)->maxlev;
  int *levsta = (*atree)->levsta;
  int *levend = (*atree)->levend;
  real *celeng = (*atree)->celeng;
  cell *c = (*atree)->c;

  /* Allocate and initialise field values of all the real cells */
  (*atree)->fieldval = (real *)calloc(ncell * dofn3, sizeof(real));

  /* Shortcut for proxy source value (moment) */
  real *PS = (*atree)->proxysval;

  /* Allocate and initialise proxy field values of all the real cells */
  real *PF = (real *)calloc(ncell * dofn3, sizeof(real));

  /* Keep M2L transfer matrices as column-major */
  real *K = E;
    
  /* Execute M2L driver */
#if defined(ENABLE_USE_PARENT_LEAVES_ARRAYS)
  m2l_drv(cutoff, PF, K, PS, minlev, maxlev,
	  celeng[0], (real3 *)c->center, c->parent, c->leaves,
	  c->ineigh, c->pitch_neighbors, c->neighbors,
	  c->iinter, c->pitch_interaction, c->interaction,
	  SCHEME);
#else
  m2l_drv(cutoff, PF, K, PS, minlev, maxlev,
	  celeng[0], (real3 *)c->center,
	  c->ineigh, c->pitch_neighbors, c->neighbors,
	  c->iinter, c->pitch_interaction, c->interaction,
	  SCHEME);
#endif

  /* Free proxy source value */
  free((*atree)->proxysval);

  /* Check kernel performance */
  double perf = m2l_aux_comp_kernel_performance_in_Gflops(cutoff, minlev, maxlev, levsta, levend, c->iinter, kernel_exec_time);
  INFO("calc_performance: kernel = %f [Gflop/s]\n", perf);

  /*
    Translate proxy field value to field value (post M2L)
  */
  
  eventTimerType time_kernel2;
  initEventTimer(&time_kernel2);

  /* Convert U from column-major to row-major */
  real *Ur = (real *)malloc(dofn3 * cutoff * sizeof(real));
  postm2l_convert_U_from_column_major_to_row_major(U, Ur, dofn3, cutoff);
  
  /* Precompute adjusting vector */
  real *adjust = (real *)malloc(dofn3 * sizeof(real));
  postm2l_compute_adjusting_vector(Kweights, adjust, dof, n3);
  
  /* Loop over levels */
  for (int level = minlev; level <= maxlev; level ++) {
    
    /* Length of cell */
    real L = celeng[level];
    
    /* Inverse-length */
    real iL = ONE / L;
    
    /* Scaling factor for SVD */
    real scale = POW(iL, homogen);
    
    /* Translate proxy field values to field values on host */
    startEventTimer(&time_kernel2);
    postm2l(levsta[level], levend[level], dofn3, cutoff, Ur, &(PF[cutoff * levsta[level]]), scale, adjust, (*atree)->fieldval);
    stopEventTimer(&time_kernel2);
  }

  free(Ur);
  free(adjust);
  free(PF);

  /* Finalise timers */
  printEventTimer(stderr, "time_kernel2", &time_kernel2);
  finalizeEventTimer(&time_kernel2);
  
  stopEventTimer(&time_all);
  printEventTimer(stderr, "time_all", &time_all);
  finalizeEventTimer(&time_all);
}
/**************************************************************************/
#else
/**************************************************************************/
#error Undefined version.
/**************************************************************************/
#endif
/**************************************************************************/
#endif /* ANOTHERFMMINTERACTION_CU */
