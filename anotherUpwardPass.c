#include "bbfmm.h"

#if(1)
#include "xdot.h"
#endif
#if(1)
#include "atransx.h"
#endif
#if(1)
#include "fast_memset.h"
#endif

#if defined(ENABLE_INTEL_MKL)
#if defined(__ICC)
#define INTEL_MKL_BLAS
#include <mkl_blas.h>
#endif
#endif

void anotherUpwardPass(anotherTree **atree, real3 *source, real *Cweights, real *Tkz,
		       real *q, real *V, real *Kweights, int cutoff, int n, int dof)
{
  real prefac = TWO / (real)n; // Prefactor for Sn
  real prefac3  = prefac*prefac*prefac;   // prefac3 = prefac^3
  cell *c = (*atree)->c;
  real3 *center = c->center;  // Center of source cell
  int *sourcesta = c->sourcesta;
  int *sourceend = c->sourceend;
  int *sourcelist = (*atree)->sourcelist;
  
  int n2 = n*n;                       // n2 = n^2
  int n3 = n*n*n;                     // n3 = n^3
  int dofn  = dof*n;
  int dofn2 = dof*n2;
  int dofn3 = dof*n3;
  
  char trans[]="t";
  real alpha = ONE, beta = ZERO;
  int incr=1;
  
  int ncell = (*atree)->ncell;
  int minlev = (*atree)->minlev;
  int maxlev = (*atree)->maxlev;
  int *levsta = (*atree)->levsta;
  int *levend = (*atree)->levend;
  real *celeng = (*atree)->celeng;
  
  /* Allocate sourceval and proxysval */
  (*atree)->sourceval = (real *)malloc(ncell * dofn3 * sizeof(real));
  (*atree)->proxysval = (real *)malloc(ncell * cutoff * sizeof(real));
  real *sourceval = (*atree)->sourceval;
  real *proxysval = (*atree)->proxysval;
  
#ifndef DISABLE_TIMING
  timerType *timer_upward_p2m;
  allocTimer(&timer_upward_p2m);
  initTimer(timer_upward_p2m);
  startTimer(timer_upward_p2m);
#endif

  {
    const int level = maxlev; // the level where P2M is performed
    
    const real L = celeng[level];    // Length of cell
    const real halfL = L / TWO;      // Length of child cell
    const real ihalfL = ONE / halfL; // Inverse of half-length
    
    /* Loop over cells in this level */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      
      /* Weighted source values */
      real *Sw = (real *)malloc(dofn3 * sizeof(real));
      
      /* Compute # of source points in this cell */
      const int Ns = sourceend[A] - sourcesta[A] + 1;
      
      /* Map all of the source points to the box ([-1 1])^3 */
      real3 *sourcet = (real3 *)malloc(Ns * sizeof(real3));

      for (int j = 0; j < Ns; j ++) {
	const int k = sourcelist[sourcesta[A] + j];
	sourcet[j].x = ihalfL * (source[k].x - center[A].x);
	sourcet[j].y = ihalfL * (source[k].y - center[A].y);
	sourcet[j].z = ihalfL * (source[k].z - center[A].z);
      }
      
      /* Compute Ss, the mapping function for the sources */
#if !defined(SLOW)
      real *Ssx = (real *)malloc(n * Ns * sizeof(real));
      real *Ssy = (real *)malloc(n * Ns * sizeof(real));
      real *Ssz = (real *)malloc(n * Ns * sizeof(real));
      ComputeSn2(sourcet, Tkz, n, Ns, Ssx, Ssy, Ssz);
#else
      real3 *Ss = (real3 *)malloc(n * Ns * sizeof(real3));
      ComputeSn(sourcet, Tkz, n, Ns, Ss);
#endif
      
      free(sourcet);
      
#if !defined(SLOW)
#if (DOF == 1)
      real *qtmp = (real *)malloc(Ns * sizeof(real));
      for (int j = 0; j < Ns; j ++) {
	const int k = sourcelist[sourcesta[A] + j];
	qtmp[j] = q[k];
      }
#else
      real *qtmp = (real *)malloc(dof * Ns * sizeof(real));
      for (int l4 = 0; l4 < dof; l4 ++) {
	for (int j = 0; j < Ns; j ++) {
	  const int k = dof * sourcelist[sourcesta[A] + j] + l4;
	  qtmp[dof * l4 + j] = q[k];
	}
      }
#endif
#endif
      
      /* Compute the source values */
      real *S = &(sourceval[dofn3 * A]);
#if !defined(SLOW)
#if (DOF == 1)
      int l = 0;
      for (int l1 = 0; l1 < n; l1 ++) {
	const real *Ssx1 = &(Ssx[l1 * Ns]);
	for (int l2 = 0; l2 < n; l2 ++) {
	  const real *Ssy2 = &(Ssy[l2 * Ns]);
#if(1)
#if(1) // unrolling x 4
	  for (int l3 = 0; l3 < n - 3; l3 += 4) {
	    const real *Ssz30 = &(Ssz[l3 * Ns]);
	    const real *Ssz31 = &(Ssz[(l3 + 1) * Ns]);
	    const real *Ssz32 = &(Ssz[(l3 + 2) * Ns]);
	    const real *Ssz33 = &(Ssz[(l3 + 3) * Ns]);
	    real sum0 = ZERO;
	    real sum1 = ZERO;
	    real sum2 = ZERO;
	    real sum3 = ZERO;
	    for (int j = 0; j < Ns; j ++) { // LOOP WAS VECTORIZED.
	      const real w = qtmp[j] * Ssx1[j] * Ssy2[j];
	      sum0 += w * Ssz30[j];
	      sum1 += w * Ssz31[j];
	      sum2 += w * Ssz32[j];
	      sum3 += w * Ssz33[j];
	    }
	    S[l] = prefac3 * sum0;
	    Sw[l] = S[l] * Kweights[l];
	    l ++;
	    S[l] = prefac3 * sum1;
	    Sw[l] = S[l] * Kweights[l];
	    l ++;
	    S[l] = prefac3 * sum2;
	    Sw[l] = S[l] * Kweights[l];
	    l ++;
	    S[l] = prefac3 * sum3;
	    Sw[l] = S[l] * Kweights[l];
	    l ++;
	  }
	  if (n % 4) { // n%4!=0
	    if (n % 4 == 1) {
	      const real *Ssz30 = &(Ssz[(n - 1) * Ns]);
	      real sum0 = ZERO;
	      for (int j = 0; j < Ns; j ++) { // LOOP WAS VECTORIZED.
		sum0 += qtmp[j] * Ssx1[j] * Ssy2[j] * Ssz30[j];
	      }
	      S[l] = prefac3 * sum0;
	      Sw[l] = S[l] * Kweights[l];
	      l ++;
	    } else if (n % 4 == 2) {
	      const real *Ssz30 = &(Ssz[(n - 2) * Ns]);
	      const real *Ssz31 = &(Ssz[(n - 1) * Ns]);
	      real sum0 = ZERO;
	      real sum1 = ZERO;
	      for (int j = 0; j < Ns; j ++) { // LOOP WAS VECTORIZED.
		const real w = qtmp[j] * Ssx1[j] * Ssy2[j];
		sum0 += w * Ssz30[j];
		sum1 += w * Ssz31[j];
	      }
	      S[l] = prefac3 * sum0;
	      Sw[l] = S[l] * Kweights[l];
	      l ++;
	      S[l] = prefac3 * sum1;
	      Sw[l] = S[l] * Kweights[l];
	      l ++;
	    } else if (n % 4 == 3) {
	      const real *Ssz30 = &(Ssz[(n - 3) * Ns]);
	      const real *Ssz31 = &(Ssz[(n - 2) * Ns]);
	      const real *Ssz32 = &(Ssz[(n - 1) * Ns]);
	      real sum0 = ZERO;
	      real sum1 = ZERO;
	      real sum2 = ZERO;
	      for (int j = 0; j < Ns; j ++) { // LOOP WAS VECTORIZED.
		const real w = qtmp[j] * Ssx1[j] * Ssy2[j];
		sum0 += w * Ssz30[j];
		sum1 += w * Ssz31[j];
		sum2 += w * Ssz32[j];
	      }
	      S[l] = prefac3 * sum0;
	      Sw[l] = S[l] * Kweights[l];
	      l ++;
	      S[l] = prefac3 * sum1;
	      Sw[l] = S[l] * Kweights[l];
	      l ++;
	      S[l] = prefac3 * sum2;
	      Sw[l] = S[l] * Kweights[l];
	      l ++;
	    }
	  }
#else // unrolling x 2
	  for (int l3 = 0; l3 < n - 1; l3 += 2) {
	    const real *Ssz30 = &(Ssz[l3 * Ns]);
	    const real *Ssz31 = &(Ssz[(l3 + 1) * Ns]);
	    real sum0 = ZERO;
	    real sum1 = ZERO;
	    for (int j = 0; j < Ns; j ++) { // LOOP WAS VECTORIZED.
	      const real w = qtmp[j] * Ssx1[j] * Ssy2[j];
	      sum0 += w * Ssz30[j];
	      sum1 += w * Ssz31[j];
	    }
	    S[l] = prefac3 * sum0;
	    Sw[l] = S[l] * Kweights[l];
	    l ++;
	    S[l] = prefac3 * sum1;
	    Sw[l] = S[l] * Kweights[l];
	    l ++;
	  }
	  if (n % 2) { // if n is odd
	    const real *Ssz30 = &(Ssz[(n - 1) * Ns]);
	    real sum0 = ZERO;
	    for (int j = 0; j < Ns; j ++) { // LOOP WAS VECTORIZED.
	      sum0 += qtmp[j] * Ssx1[j] * Ssy2[j] * Ssz30[j];
	    }
	    S[l] = prefac3 * sum0;
	    Sw[l] = S[l] * Kweights[l];
	    l ++;
	  }
#endif
#else
	  for (int l3 = 0; l3 < n; l3 ++) {
	    const real *Ssz3 = &(Ssz[l3 * Ns]);
	    real sum = ZERO;
	    for (int j = 0; j < Ns; j ++) { // LOOP WAS VECTORIZED.
	      sum += qtmp[j] * Ssx1[j] * Ssy2[j] * Ssz3[j];
	    }
	    S[l] = prefac3 * sum;
	    Sw[l] = S[l] * Kweights[l];
	    l ++;
	  }
#endif
	}
      }
#else
      int l = 0;
      int m = 0;
      for (int l1 = 0; l1 < n; l1 ++) {
	const real *Ssx1 = &(Ssx[l1 * Ns]);
	for (int l2 = 0; l2 < n; l2 ++) {
	  const real *Ssy2 = &(Ssy[l2 * Ns]);
	  for (int l3 = 0; l3 < n; l3 ++) {
	    const real *Ssz3 = &(Ssz[l3 * Ns]);
	    const real tmp = Kweights[m];
	    m ++;
	    for (int l4 = 0; l4 < dof; l4 ++) {
	      const real *qtmp4 = &(qtmp[dof * l4]);
	      real sum = ZERO;
	      for (int j = 0; j < Ns; j ++) { // LOOP WAS VECTORIZED.
		sum += qtmp4[j] * Ssx1[j] * Ssy2[j] * Ssz3[j];
	      }
	      S[l] = prefac3 * sum;
	      Sw[l] = S[l] * tmp;
	      l ++;
	    }
	  }
	}
      }
#endif
#else
      int l = 0;
      int m = 0;
      for (int l1 = 0; l1 < n; l1 ++) {
	for (int l2 = 0; l2 < n; l2 ++) {
	  for (int l3 = 0; l3 < n; l3 ++) {
	    const real tmp = Kweights[m];
	    m ++;
	    for (int l4 = 0; l4 < dof; l4 ++) {
	      real sum = ZERO;
	      for (int j = 0; j < Ns; j ++) { // LOOP WAS VECTORIZED.
		int k = dof * sourcelist[sourcesta[A] + j] + l4;
		sum += q[k] * Ss[l1 * Ns + j].x * Ss[l2 * Ns + j].y * Ss[l3 * Ns + j].z;
	      }
	      S[l] = prefac3 * sum;
	      Sw[l] = S[l] * tmp;
	      l ++;
	    }
	  }
	}
      }
#endif
      
#if !defined(SLOW)
      free(Ssx);
      free(Ssy);
      free(Ssz);
      free(qtmp);
#else
      free(Ss);
#endif

      /* Determine the proxy values by pre-multiplying S by U^T */
      real *P = &(proxysval[cutoff * A]);
#if defined(INTEL_MKL_BLAS)
#if defined(SINGLE)
      sgemv(trans, &dofn3, &cutoff, &alpha, V, &dofn3, Sw, &incr, &beta, P, &incr);
#else
      dgemv(trans, &dofn3, &cutoff, &alpha, V, &dofn3, Sw, &incr, &beta, P, &incr);
#endif
#else
#if(0)
      agemv_(trans, &dofn3, &cutoff, &alpha, V, &dofn3, Sw, &incr, &beta, P, &incr);
#else
      atransx(dofn3, cutoff, V, Sw, P);
#endif
#endif
      
      free(Sw);

    } // A
  } // level=maxlev

#ifndef DISABLE_TIMING
  stopTimer(timer_upward_p2m);
  printTimer(stderr, "upward_p2m", timer_upward_p2m);
  freeTimer(&timer_upward_p2m);
#endif

#ifndef DISABLE_TIMING
  timerType *timer_upward_m2m;
  allocTimer(&timer_upward_m2m);
  initTimer(timer_upward_m2m);
  startTimer(timer_upward_m2m);
#endif

  /* Loop over levels, where M2M is performed */
#ifdef _OPENMP
#pragma omp parallel
#endif
  for (int level = maxlev - 1; level >= minlev; level --) { // OpenMP DEFINED REGION WAS PARALLELIZED.
    
    const real L = celeng[level];    // Length of cell
    const real halfL = L / TWO;      // Length of child cell
    const real ihalfL = ONE / halfL; // Inverse of half-length
    
    /* Loop over cells in this level */
#ifdef _OPENMP
#pragma omp for
#endif
    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      
      /* Weighted source values */
      real *Sw = (real *)malloc(dofn3 * sizeof(real));

      /* First gather the sources for all children cells and then
	 gather for the parent cell - otherwise map all sources to
	 Chebyshev nodes */
      
      real *Sy = (real *)calloc(2 * dofn3, sizeof(real)); // initialize
      real *Sz = (real *)calloc(4 * dofn3, sizeof(real)); // initialize
      
      /* Initialization */
      int xcount = 0;
      int ycount = 0;
      int zcount = 0;
      int zindex[8] = {- 1, - 1, - 1, - 1, - 1, - 1, - 1, - 1};
      int yindex[4] = {- 1, - 1, - 1, - 1};
      int xindex[2] = {- 1, - 1};
      
      /* Initialize the source values of this cell */
      real *S = &(sourceval[dofn3 * A]);
      for (int l = 0; l < dofn3; l ++) { // LOOP WAS VECTORIZED.
	S[l] = ZERO;
      }

      /* Determine which children contain sources */
      for (int i = 0; i < 8; i ++) {

	const int num_sources = GET_NUM_PARTICLES(sourcesta, sourceend, A);
	if (num_sources != 0) {
	  zindex[zcount] = i; // list of children that contain sources
	  zcount ++;
	  if (ycount == 0 || yindex[ycount - 1] != i / 2) {
	    yindex[ycount] = i / 2;
	    ycount ++;
	  }
	  if (xcount == 0 || xindex[xcount - 1] != i / 4) {
	    xindex[xcount] = i / 4;
	    xcount ++;
	  }
	}
      }
	
      /* Gather the children source along the z-component */
      for (int i = 0; i < zcount; i ++) {
	int j = zindex[i];
	int l = (int)(j / 2) * dofn3;
	real *Schild = &(sourceval[dofn3 * GET_CHILD_INDEX(A, j)]); // Source values for child cell
	
	int wstart;
	if (j % 2 == 0) {
	  wstart = 0;
	} else {
	  wstart = n2;
	}
	
	for (int l1 = 0; l1 < n2; l1 ++) {
	  int count1 = l1 * n;
	  for (int l3 = 0; l3 < n; l3 ++) {
	    int count2 = wstart + l3 * n;
#if (DOF == 1)
#if(1)
	    Sz[l] += xdot(n, &Schild[count1], dof, &Cweights[count2], incr);
#else
	    Sz[l] += adot_(&n, &Schild[count1], &dof, &Cweights[count2], &incr);
#endif
	    l ++;
#else
	    for (int l4 = 0; l4 < dof; l4 ++) {
	      int count3 = dof * count1 + l4;
#if(1)
	      Sz[l] += xdot(n, &Schild[count3], dof, &Cweights[count2], incr);
#else
	      Sz[l] += adot_(&n, &Schild[count3], &dof, &Cweights[count2], &incr);
#endif
	      l ++;
	    }
#endif
	  }
	}
      }
	
      /* Gather the children sources along the y-component */
      for (int i = 0; i < ycount; i ++) {
	int j = yindex[i];
	int l = (int)(j / 2) * dofn3;
	int wstart;
	if (j % 2 == 0) {
	  wstart = 0;
	} else {
	  wstart = n2;
	}
	
	for (int l1 = 0; l1 < n; l1 ++) {
	  for (int l2 = 0; l2 < n; l2 ++) {
	    int count2 = wstart + l2 * n;
	    for (int l3 = 0; l3 < n; l3 ++) {
	      int count1 = j * n3 + l1 * n2 + l3;
#if (DOF == 1)
#if(1)
	      Sy[l] += xdot(n, &Sz[count1], dofn, &Cweights[count2], incr);
#else
	      Sy[l] += adot_(&n, &Sz[count1], &dofn, &Cweights[count2], &incr);
#endif
	      l ++;
#else
	      for (int l4 = 0; l4 < dof; l4 ++) {
		int count3 = dof * count1 + l4;
#if(1)
		Sy[l] += xdot(n, &Sz[count3], dofn, &Cweights[count2], incr);
#else
		Sy[l] += adot_(&n, &Sz[count3], &dofn, &Cweights[count2], &incr);
#endif
		l ++;
	      }
#endif
	    }
	  }
	}
      }
      free(Sz);
	
      /* Gather the children sources along the z-component and
	 determine the parent sources */
      for (int i = 0; i < xcount; i ++) {
	int j = xindex[i];
	int l = 0;
	int wstart;
	if (j == 0) {
	  wstart = 0;
	} else {
	  wstart = n2;
	}
	
	for (int l1 = 0; l1 < n; l1 ++) {
	  int count2 = wstart + l1 * n;
	  for (int l2 = 0; l2 < n; l2 ++) {
	    for (int l3 = 0; l3 < n; l3 ++) {
	      int count1 = j * n3 + l2 * n + l3;
#if (DOF == 1)
#if(1)
	      S[l] += xdot(n, &Sy[count1], dofn2, &Cweights[count2], incr);
#else
	      S[l] += adot_(&n, &Sy[count1], &dofn2, &Cweights[count2], &incr);
#endif
	      l ++;
#else
	      for (int l4 = 0; l4 < dof; l4 ++) {
		int count3 = dof * count1 + l4;
#if(1)
		S[l] += xdot(n, &Sy[count3], dofn2, &Cweights[count2], incr);
#else
		S[l] += adot_(&n, &Sy[count3], &dofn2, &Cweights[count2], &incr);
#endif
		l ++;
	      }
#endif
	    }
	  }
	}
      }
      free(Sy);

      /* Multiply by prefactor and appropriate weighting for SVD */
#if (DOF == 1)
      for (int l = 0; l < n3; l ++) { // LOOP WAS VECTORIZED.
	S[l] *= prefac3;
	Sw[l] = S[l] * Kweights[l];
      }
#else
      int l = 0;
      for (int l1 = 0; l1 < n3; l1 ++) {
	real tmp = Kweights[l1];
	for (int l4 = 0; l4 < dof; l4 ++) {
	  S[l] *= prefac3;
	  Sw[l] = S[l] * tmp;
	  l ++;
	}
      }
#endif	

      /* Determine the proxy values by pre-multiplying S by U^T */
      real *P = &(proxysval[cutoff * A]);
#if defined(INTEL_MKL_BLAS)
#if defined(SINGLE)
      sgemv(trans, &dofn3, &cutoff, &alpha, V, &dofn3, Sw, &incr, &beta, P, &incr);
#else
      dgemv(trans, &dofn3, &cutoff, &alpha, V, &dofn3, Sw, &incr, &beta, P, &incr);
#endif
#else
#if(0)
      agemv_(trans, &dofn3, &cutoff, &alpha, V, &dofn3, Sw, &incr, &beta, P, &incr);
#else
      atransx(dofn3, cutoff, V, Sw, P);
#endif
#endif
      
      free(Sw);
      
    } // A 
  } // level

#ifndef DISABLE_TIMING
  stopTimer(timer_upward_m2m);
  printTimer(stderr, "upward_m2m", timer_upward_m2m);
  freeTimer(&timer_upward_m2m);
#endif

  /* Free sourceval (keep proxysval) */
  free((*atree)->sourceval);

}
