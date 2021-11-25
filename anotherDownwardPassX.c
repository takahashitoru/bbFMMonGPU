#include "bbfmm.h"

#if(1)
#include "xdot.h"
#endif

void anotherDownwardPassX(anotherTree **atree, real3 *field,
			  real *Cweights, real *Tkz, real *q, real *U,
			  int cutoff, int n, int dof, real homogen, real *phi)
{
  real prefac = TWO / (real)n;          // Prefactor for Sn
  real prefac3  = prefac * prefac * prefac;

  int n2 = n * n;
  int n3 = n * n * n;
  int dofn = dof * n;
  int dofn2 = dof * n2;
  int dofn3 = dof * n3;

  int minlev = (*atree)->minlev;
  int maxlev = (*atree)->maxlev;
  int *levsta = (*atree)->levsta;
  int *levend = (*atree)->levend;
  real *celeng = (*atree)->celeng;
  cell *c = (*atree)->c;
  int *fieldsta = c->fieldsta;
  int *fieldend = c->fieldend;
  real3 *center = c->center;
  
  int *fieldlist = (*atree)->fieldlist;


#ifndef DISABLE_TIMING
  timerType *timer_downward_l2l;
  allocTimer(&timer_downward_l2l);
  initTimer(timer_downward_l2l);
  startTimer(timer_downward_l2l);
#endif

#ifdef _OPENMP
#pragma omp parallel // OpenMP DEFINED REGION WAS PARALLELIZED.
#endif
  for (int level = minlev; level < maxlev; level ++) { // where L2L (from the underlying cell to its children) is performed
    
    const real L = celeng[level]; // Length of cell
    const real halfL = L / TWO; // Length of child cell
    const real ihalfL = ONE / halfL; // Inverse of half-length
    
#ifdef _OPENMP
#pragma omp for
#endif
    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.

      real *F = &((*atree)->fieldval[dofn3 * A]); // Field values for cell 

      /* Add the contributions from the parent cell to each child cell
	 - otherwise compute all direct interactions and then
	 interpolate to the field points */

      /* Initialization */
      real *Fx = (real *)calloc(2 * dofn3, sizeof(real));
      real *Fy = (real *)calloc(4 * dofn3, sizeof(real));
      
      int xcount = 0;
      int ycount = 0;
      int zcount = 0;
      int zindex[8] = {- 1, - 1, - 1, - 1, - 1, - 1, - 1, - 1};
      int yindex[4] = {- 1, - 1, - 1, - 1};
      int xindex[2] = {- 1, - 1};
      
      /* Determine which children cells contain field points */
      for (int i = 0; i < 8; i ++) {
	const int B = GET_CHILD_INDEX(A, i); // i-th child of A
	if (GET_NUM_PARTICLES(c->fieldsta, c->fieldend, B) != 0) {
	  zindex[zcount] = i;
	  zcount++;
	  if (ycount == 0 || yindex[ycount - 1] != i / 2) {
	    yindex[ycount] = i / 2;
	    ycount++;
	  }
	  if (xcount == 0 || xindex[xcount - 1] != i / 4) {
	    xindex[xcount] = i / 4;
	    xcount++;
	  }
	}
      }
      
      /* Interpolate the parent field along the x-component */
      for (int i = 0; i < xcount; i ++) {
	int j = xindex[i]; // 0 or 1
	int l = j * dofn3;
	int wstart;
	if (j == 0) {
	  wstart = 0;
	} else {
	  wstart = n2;
	}
	
	for (int l1 = 0; l1 < n; l1 ++) {
	  int count2 = wstart + l1;
	  for (int l2 = 0; l2 < n; l2 ++) {
	    for (int l3 = 0; l3 < n; l3 ++) {
	      int count1 = l2 * n + l3;
#if (DOF == 1)
#if(1)
	      Fx[l] = xdot(n, &F[count1], dofn2, &Cweights[count2], n);
#else
	      Fx[l] = adot_(&n, &F[count1], &dofn2, &Cweights[count2], &n);
#endif
	      l ++;
#else
	      for (int l4 = 0; l4 < dof; l4 ++) {
		int count3 = dof * count1 + l4;
#if(1)
		Fx[l] = xdot(n, &F[count3], dofn2, &Cweights[count2], n);
#else
		Fx[l] = adot_(&n, &F[count3], &dofn2, &Cweights[count2], &n);
#endif
		l ++;
	      }
#endif
	    }
	  }
	}
      }
      
      /* Interpolate the parent field along the y-component */
      for (int i = 0; i < ycount; i ++) {
	int j = yindex[i]; // 0,1,2,3
	int l = j * dofn3;
	int wstart;
	if (j % 2 == 0) {
	  wstart = 0;
	} else {
	  wstart = n2;
	}
	
	for (int l1 = 0; l1 < n; l1 ++) {
	  for (int l2 = 0; l2 < n; l2 ++) {
	    int count2 = wstart + l2;
	    for (int l3 = 0; l3 < n; l3 ++) {
	      int count1 = (int)(j / 2) * n3 + l1 * n2 + l3;
#if (DOF == 1)
#if(1)
	      Fy[l] = xdot(n, &Fx[count1], dofn, &Cweights[count2], n);
#else
	      Fy[l] = adot_(&n, &Fx[count1], &dofn, &Cweights[count2], &n);
#endif
	      l ++;
#else
	      for (int l4 = 0; l4 < dof; l4 ++) {
		int count3 = dof * count1 + l4;
#if(1)
		Fy[l] = xdot(n, &Fx[count3], dofn, &Cweights[count2], n);
#else
		Fy[l] = adot_(&n, &Fx[count3], &dofn, &Cweights[count2], &n);
#endif
		l ++;
	      }
#endif
	    }
	  }
	}
      }
      
      free(Fx);
      
      /* Interpolate the parent field along the z-component and add
	 to child field */
      for (int i = 0; i < zcount; i ++) {
	int j = zindex[i];
	int l = 0;
	real *Fchild = &((*atree)->fieldval[dofn3 * GET_CHILD_INDEX(A, j)]);
	
	int wstart;
	if (j % 2 == 0) {
	  wstart = 0;
	} else {
	  wstart = n2;
	}
	
	for (int l1 = 0; l1 < n2; l1 ++) {
	  int count1 = (int)(j / 2) * n3 + l1 * n;
	  for (int l3 = 0; l3 < n; l3 ++) {
	    int count2 = wstart + l3;
#if (DOF == 1)
#if(1)
	    Fchild[l] += prefac3 * xdot(n, &Fy[count1], dof, &Cweights[count2], n);
#else
	    Fchild[l] += prefac3 * adot_(&n, &Fy[count1], &dof, &Cweights[count2], &n);
#endif
	    l ++;
#else
	    for (int l4 = 0; l4 < dof; l4 ++) {
	      int count3 = dof * count1 + l4;
#if(1)
	      Fchild[l] += prefac3 * xdot(n, &Fy[count3], dof, &Cweights[count2], n);
#else
	      Fchild[l] += prefac3 * adot_(&n, &Fy[count3], &dof, &Cweights[count2], &n);
#endif
	      l ++;
	    }
#endif
	  }
	}
      }
      
      free(Fy);

    } // A
  } // level	

#ifndef DISABLE_TIMING
  stopTimer(timer_downward_l2l);
  printTimer(stderr, "downward_l2l", timer_downward_l2l);
  freeTimer(&timer_downward_l2l);
#endif

#ifndef DISABLE_TIMING
  timerType *timer_downward_l2p;
  allocTimer(&timer_downward_l2p);
  initTimer(timer_downward_l2p);
  startTimer(timer_downward_l2p);
#endif

  {
    const int level = maxlev; // where evaluation by L is performed
    
    const real L = celeng[level]; // Length of cell
    const real halfL = L / TWO; // Length of child cell
    const real ihalfL = ONE / halfL; // Inverse of half-length

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.

      real3 fcenter = center[A]; // Center of field cell
      int Nf = fieldend[A] - fieldsta[A] + 1; // Number of field points
      real *F = &((*atree)->fieldval[dofn3 * A]); // Field values for cell 

      /* Map all of the field points to the box ([-1 1])^3 */
      real3 *fieldt = (real3 *)malloc(Nf * sizeof(real3));
      for (int i = 0; i < Nf; i ++) {
	int k = fieldlist[fieldsta[A] + i];
	fieldt[i].x = ihalfL * (field[k].x - fcenter.x);
	fieldt[i].y = ihalfL * (field[k].y - fcenter.y);
	fieldt[i].z = ihalfL * (field[k].z - fcenter.z);
      }
      
      /* Compute Sf, the mapping function for the field points */
#if !defined(SLOW)
      real *Sfx = (real *)malloc(n * Nf * sizeof(real));
      real *Sfy = (real *)malloc(n * Nf * sizeof(real));
      real *Sfz = (real *)malloc(n * Nf * sizeof(real));
      ComputeSn2(fieldt, Tkz, n, Nf, Sfx, Sfy, Sfz);
#else
      real3 *Sf = (real3 *)malloc(n * Nf * sizeof(real3));
      ComputeSn(fieldt, Tkz, n, Nf, Sf);
#endif

      free(fieldt);
      
      /* Compute the values at the field points */
#if !defined(SLOW)
#if(DOF == 1)
#if(1) // loop ordering was changed
#if(1) // unrolling
      real *sum = (real *)calloc(Nf, sizeof(real)); // initialize
      int l = 0;
      for (int l1 = 0; l1 < n; l1 ++) {
	real *Sfx1 = &(Sfx[l1 * Nf]);
	for (int l2 = 0; l2 < n; l2 ++) {
	  real *Sfy2 = &(Sfy[l2 * Nf]);
	  for (int l3 = 0; l3 < n - 3; l3 += 4) {
	    real *Sfz30 = &(Sfz[l3 * Nf]);
	    real *Sfz31 = &(Sfz[(l3 + 1) * Nf]);
	    real *Sfz32 = &(Sfz[(l3 + 2) * Nf]);
	    real *Sfz33 = &(Sfz[(l3 + 3) * Nf]);
	    for (int i = 0; i < Nf; i ++) { // LOOP WAS VECTORIZED.
	      sum[i] += Sfx1[i] * Sfy2[i] * (Sfz30[i] * F[l] + Sfz31[i] * F[l + 1] + Sfz32[i] * F[l + 2] + Sfz33[i] * F[l + 3]);
	    }	    
	    l += 4;
	  }
	  if (n % 4) { // n%4!=0
	    if (n % 4 == 1) {
	      real *Sfz30 = &(Sfz[(n - 1) * Nf]);
	      for (int i = 0; i < Nf; i ++) { // LOOP WAS VECTORIZED.
		sum[i] += Sfx1[i] * Sfy2[i] * Sfz30[i] * F[l];
	      }	    
	      l ++;
	    } else if (n % 4 == 2) {
	      real *Sfz30 = &(Sfz[(n - 2) * Nf]);
	      real *Sfz31 = &(Sfz[(n - 1) * Nf]);
	      for (int i = 0; i < Nf; i ++) { // LOOP WAS VECTORIZED.
		sum[i] += Sfx1[i] * Sfy2[i] * (Sfz30[i] * F[l] + Sfz31[i] * F[l + 1]);
	      }	    
	      l += 2;
	    } else if (n % 4 == 3) {
	      real *Sfz30 = &(Sfz[(n - 3) * Nf]);
	      real *Sfz31 = &(Sfz[(n - 2) * Nf]);
	      real *Sfz32 = &(Sfz[(n - 1) * Nf]);
	      for (int i = 0; i < Nf; i ++) { // LOOP WAS VECTORIZED.
		sum[i] += Sfx1[i] * Sfy2[i] * (Sfz30[i] * F[l] + Sfz31[i] * F[l + 1] + Sfz31[i] * F[l + 2]);
	      }	    
	      l += 3;
	    }
	  }
	}
      }
#else
      real *sum = (real *)calloc(Nf, sizeof(real)); // initialize
      int l = 0;
      for (int l1 = 0; l1 < n; l1 ++) {
	real *Sfx1 = &(Sfx[l1 * Nf]); // Sfx[l1*Nf+i] where i=0
	for (int l2 = 0; l2 < n; l2 ++) {
	  real *Sfy2 = &(Sfy[l2 * Nf]); // Sfy[l2*Nf+i] where i=0
	  for (int l3 = 0; l3 < n; l3 ++) {
	    real *Sfz3 = &(Sfz[l3 * Nf]); // Sfz[l3*Nf+i] where i=0
	    for (int i = 0; i < Nf; i ++) { // LOOP WAS VECTORIZED.
	      sum[i] += Sfx1[i] * Sfy2[i] * Sfz3[i] * F[l];
	    }	    
	    l ++;
	  }
	}
      }
#endif
      for (int i = 0; i < Nf; i ++) {
	int k = fieldlist[fieldsta[A] + i];
	phi[k] = q[k] * prefac3 * sum[i];
      }
      free(sum);
#endif
#if(0)
      for (int i = 0; i < Nf; i ++) {
	int k = fieldlist[fieldsta[A] + i];
	/* Due to far field interactions */
	real sum = ZERO;
	int l = 0;
	for (int l1 = 0; l1 < n; l1 ++) {
	  real tmp1 = Sfx[l1 * Nf + i];
	  for (int l2 = 0; l2 < n; l2 ++) {
	    real tmp2 = tmp1 * Sfy[l2 * Nf + i];

	    real sum3 = ZERO;
	    real *pF = &(F[l]); // F[l]
	    real *pSfz = &(Sfz[i]); // Sfz[l3*Nf+i] for l3=0
	    for (int l3 = 0; l3 < n; l3 ++) { // LOOP WAS VECTORIZED.
	      sum3 += (*pF) * (*pSfz);
	      pF ++;
	      pSfz += Nf;
	    }
	    l += n;
	    sum += tmp2 * sum3;
	    
	  }
	}
	phi[k] = q[k] * prefac3 * sum;
      }
#endif
#else
      for (int i = 0; i < Nf; i ++) {
	int k = dof * fieldlist[fieldsta[A] + i];
	for (int l4 = 0; l4 < dof; l4 ++) {
	  /* Due to far field interactions */
	  real sum = ZERO;
	  int l = l4;
	  for (int l1 = 0; l1 < n; l1 ++) {
	    real tmp1 = Sfx[l1 * Nf + i];
	    for (int l2 = 0; l2 < n; l2 ++) {
	      real tmp2 = tmp1 * Sfy[l2 * Nf + i];

	      real sum3 = ZERO;
	      real *pF = &(F[l]); // F[l]
	      real *pSfz = &(Sfz[i]); // Sfz[l3*Nf+i] for l3=0
	      for (int l3 = 0; l3 < n; l3 ++) { // LOOP WAS VECTORIZED.
		sum3 += (*pF) * (*pSfz);
		pF += dof;
		pSfz += Nf;
	      }
	      l += dof * n;
	      sum += tmp2 * sum3;

	    }
	  }
	  phi[k] = q[k] * prefac3 * sum;
	  k ++;
	}
      }
#endif
#else
      for (int i = 0; i < Nf; i ++) {
	int k = dof * fieldlist[fieldsta[A] + i];
	for (int l4 = 0; l4 < dof; l4 ++) {
	  /* Due to far field interactions */
	  real sum = ZERO;
	  int l = l4;
	  for (int l1 = 0; l1 < n; l1 ++) {
	    real tmp1 = Sf[l1 * Nf + i].x;
	    for (int l2 = 0; l2 < n; l2 ++) {
	      real tmp2 = tmp1 * Sf[l2 * Nf + i].y;
	      for (int l3 = 0; l3 < n; l3 ++) {
		sum += F[l] * tmp2 * Sf[l3 * Nf + i].z;
		l += dof;
	      }
	    }
	  }
	  phi[k] = q[k] * prefac3 * sum;
	  k ++;
	}
      }
#endif
      
#if !defined(SLOW)
      free(Sfx);
      free(Sfy);
      free(Sfz);
#else
      free(Sf);
#endif
      
    } // A

  } // level=maxlev

#ifndef DISABLE_TIMING
  stopTimer(timer_downward_l2p);
  printTimer(stderr, "downward_l2p", timer_downward_l2p);
  freeTimer(&timer_downward_l2p);
#endif

  /* Free fieldval */
  free((*atree)->fieldval);

}    
