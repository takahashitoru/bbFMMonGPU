#include "bbfmm.h"

#if defined(CPU9) // alias
#define CPU8
#define CPU8A
#endif

/**************************************************************************/
#if defined(CPU8)
/**************************************************************************/

#if !defined(FIELDPOINTS_EQ_SOURCES)
#error FIELDPOINTS_EQ_SOURCES is assumed.
#endif

#if (DOF != 1)
#error DOF is assumed to be one.
#endif

/*========================================================================*/
#if defined(CPU8B)
/*========================================================================*/
/* Based on CPU8A and CUDA_VER46P */

static void direct(int *ineigh, int *neighbors, int pitch_neighbors,
		   int *liststa, int *listend,
		   real *x, real *y, real *z, real *p, real *q, int A)
{
  /* Compute the neighbor list that does not include any empty
     cells */
  int liststa0[27], listend0[27];
  int ineighA0 = 0; // counter for non-empty neighbors
  {
    const int ineighA = ineigh[A];
    const int *neighborsA = &(neighbors[pitch_neighbors * A]);
    for (int i = 0; i < ineighA; i ++) { 
      const int B = neighborsA[i];
      if (listend[B] - liststa[B] + 1 > 0) { // B is not empty
	liststa0[ineighA0] = liststa[B];
	listend0[ineighA0] = listend[B];
	ineighA0 ++;
      }
    }
  }

  /* Early exit */
  if (ineighA0 == 0) { // there is no non-empty neighbors (including A itself); otherwise ineighA0>=1
    return;
  }

  /* Compute the reduced neighbor list; the maximum value of nneighbor
     is probablly 14 */
  int jsta[27], jend[27];
  int nneighbor = 0;
  int ip = 0;
  jsta[nneighbor] = liststa0[ip]; /// liststa0[0] has a certain value because ineighA0>=1
  for (int i = 1; i < ineighA0; i ++) {
    if (liststa0[i] != listend0[ip] + 1) {
      jend[nneighbor] = listend0[ip];
      nneighbor ++;
      jsta[nneighbor] = liststa0[i];
    }
    ip = i;
  }
  jend[nneighbor] = listend0[ip];
  nneighbor ++;


  /////////////////////////////////////////////////////////////
  //#if(1)
  //  fprintf(stderr, "A=%d: ineigh[A]=%d num=%d ineighA0=%d nneighbor=%d\n",
  //	  A, ineigh[A], listend[A] - liststa[A] + 1, ineighA0, nneighbor);
  //#endif
  /////////////////////////////////////////////////////////////


  /* Loop over particles in cell A */
  for (int i = liststa[A]; i <= listend[A]; i ++) {

    /* Load the position of particle f */
    const real fx = x[i];
    const real fy = y[i];
    const real fz = z[i];

    /* Initialise the field value of f */
    real v = ZERO;
  
    /* Loop over the reduced neighbors of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      /* Obtain the number of particles in the reduced neighbor B */
      int nj = jend[ineighbor] - jsta[ineighbor] + 1;

      const real *xjsta = &(x[jsta[ineighbor]]);
      const real *yjsta = &(y[jsta[ineighbor]]);
      const real *zjsta = &(z[jsta[ineighbor]]);
      const real *qjsta = &(q[jsta[ineighbor]]);

      /* Loop over particles in B */
      for (int j = 0; j < nj; j ++) { // LOOP WAS VECTORIZED.
	
	const real sx = xjsta[j];
	const real sy = yjsta[j];
	const real sz = zjsta[j];
	
	/* Accumulate the interaction between f and s */
	const real dx = sx - fx;
	const real dy = sy - fy;
	const real dz = sz - fz;
	const real rr = dx * dx + dy * dy + dz * dz;
	const real rinv = (rr != ZERO ? ONE / SQRT(rr) : ZERO); // zero is dummy

#ifdef LAPLACIAN
	/* Compute 1/r */
	const real kern = rinv;
#elif LAPLACIANFORCE
	/* Compute r.x/r^3 */
	const real kern = dx * rinv * rinv * rinv;
#elif ONEOVERR4
	/* Compute 1/r^4 */
	const real kern = rinv * rinv * rinv * rinv;
#else
#error kernel is not defined.
#endif
	v += kern * qjsta[j];
	
      } // j

    } // ineighbor
    
    p[i] += q[i] * v;

  } // i
  
}
/*========================================================================*/
#elif defined(CPU8A)
/*========================================================================*/
/* Based on CUDA_VER46F */

static void direct(int *ineigh, int *neighbors, int pitch_neighbors,
		   int *liststa, int *listend,
		   real *x, real *y, real *z, real *p, real *q, int A)
{
  /* Loop over particles in cell A */
  for (int i = liststa[A]; i <= listend[A]; i ++) {

    /* Load the position of particle f */
    real fx = x[i];
    real fy = y[i];
    real fz = z[i];

    /* Initialise the field value of f */
    real v = ZERO;
  
    /* Loop over neighbors of A */
    for (int ineighbor = 0; ineighbor < ineigh[A]; ineighbor ++) {
      
      /* Obtain the index of neighbor */
      int B = neighbors[pitch_neighbors * A + ineighbor];
      
      /* Obtain the indexes of the first and last particles in B */
      int jsta = liststa[B];
      int jend = listend[B];

      /* Obtain the number of particles in B */
      int nj = jend - jsta + 1;
      
      /* Loop over particles in B */
      for (int j = 0; j < nj; j ++) { // LOOP WAS VECTORIZED.
	
	/* Load the position of particle s in B */
	real sx = x[jsta + j];
	real sy = y[jsta + j];
	real sz = z[jsta + j];
	
	/* Accumulate the interaction between f and s */
	real dx = sx - fx;
	real dy = sy - fy;
	real dz = sz - fz;
	real rr = dx * dx + dy * dy + dz * dz;
	real rinv = (rr != ZERO ? ONE / SQRT(rr) : ZERO); // zero is dummy

#ifdef LAPLACIAN
	/* Compute 1/r */
	real kern = rinv;
#elif LAPLACIANFORCE
	/* Compute r.x/r^3 */
	real kern = dx * rinv * rinv * rinv;
#elif ONEOVERR4
	/* Compute 1/r^4 */
	real kern = rinv * rinv * rinv * rinv;
#else
#error kernel is not defined.
#endif
	v += kern * q[jsta + j];
	
      } // j
      
    } // ineighbor
    
    p[i] += q[i] * v;

  } // i

}
/*========================================================================*/
#else
/*========================================================================*/
#error No minor version was specified.
/*========================================================================*/
#endif
/*========================================================================*/

#if !defined(FIELDPOINTS_EQ_SOURCES)
#error FIELDPOINTS_EQ_SOURCES is assumed.
#endif

void anotherNearField(anotherTree **atree, int dof, real homogen,
		      real3 *field, real3 *source, real *phi, real *q)
{
  /* Setup timers */  

  timerType *timer_downward_nearby_kernel;
  allocTimer(&timer_downward_nearby_kernel);
  initTimer(timer_downward_nearby_kernel);

  /* Aliases */

  int ncell = (*atree)->ncell;
  int maxlev = (*atree)->maxlev;
  int *levsta = (*atree)->levsta;
  int *levend = (*atree)->levend;
  cell *c = (*atree)->c;
  int Nf = (*atree)->Nf; // number of all field points
  int Ns = (*atree)->Ns; // number of all sources
  int *ineigh = c->ineigh;
  int *neighbors = c->neighbors;
  int pitch_neighbors = c->pitch_neighbors;
  int *fieldlist = (*atree)->fieldlist;
  int *fieldsta = c->fieldsta;
  int *fieldend = c->fieldend;

  int N = Nf;
  int *list = fieldlist; // =sourcelist
  int *liststa = fieldsta; // =sourcesta
  int *listend = fieldend; // =sourceend
  real3 *position = field; // =source

  /* Set up field points and sources */

  real *x_sorted = (real *)malloc(N * sizeof(real));
  real *y_sorted = (real *)malloc(N * sizeof(real));
  real *z_sorted = (real *)malloc(N * sizeof(real));
  real *p_sorted = (real *)malloc(N * sizeof(real));
  real *q_sorted = (real *)malloc(N * sizeof(real));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
    for (int i = liststa[A]; i <= listend[A]; i ++) {
      int index = list[i];
      x_sorted[i] = position[index].x;
      y_sorted[i] = position[index].y;
      z_sorted[i] = position[index].z;
      p_sorted[i] = phi[index];
      q_sorted[i] = q[index];
    }
  }

  /* Statistics */
#if defined(MYDEBUG)
  int nc = levend[maxlev] - levsta[maxlev] + 1; // number of cells at maxlev
  double ave_num_particles_per_cell = 0;
  int max_num_particles_per_cell = 0;
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
    int num = listend[A] - liststa[A] + 1;
    ave_num_particles_per_cell += num;
    max_num_particles_per_cell = MAX(max_num_particles_per_cell, num);
  }
  ave_num_particles_per_cell /= nc;
  INFO("ave_num_particles_per_cell = %6.1f\n", ave_num_particles_per_cell);
  INFO("max_num_particles_per_cell = %d\n", max_num_particles_per_cell);
#endif
  
  /* Loop over cells at maxlev */

  startTimer(timer_downward_nearby_kernel);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.

    /* Perform direct computaions */

    direct(ineigh, neighbors, pitch_neighbors,
	   liststa, listend,
	   x_sorted, y_sorted, z_sorted, p_sorted, q_sorted, A);

  }    

  stopTimer(timer_downward_nearby_kernel);

  /* Store the field values on CPU */

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    for (int i = liststa[A]; i <= listend[A]; i ++) {
      int index = list[i];
      phi[index] = p_sorted[i];
    }
  }

  /* Free */

  free(x_sorted);
  free(y_sorted);
  free(z_sorted);
  free(p_sorted);
  free(q_sorted);

#if defined(CHECK_PERFORMANCE)
  /* Calculate kernel's performance */
  double num_pairwise_interactions = 0;
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
    int nf = listend[A] - liststa[A] + 1;
    for (int i = 0; i < ineigh[A]; i ++) {
      int B = neighbors[pitch_neighbors * A + i];
      int ns = listend[B] - liststa[B] + 1;
      num_pairwise_interactions += nf * ns;
    }
  }
  double num_pairwise_interactions_per_sec = num_pairwise_interactions / getTimer(*timer_downward_nearby_kernel);
  INFO("num_pairwise_interactions_per_sec = %f [G interaction/s]\n", num_pairwise_interactions_per_sec / giga);
#endif

  /* Free timers */

  printTimer(stderr, "time_downward_nearby_kernel", timer_downward_nearby_kernel);
  freeTimer(&timer_downward_nearby_kernel);

}

/**************************************************************************/
#else
/**************************************************************************/

void anotherNearField(anotherTree **atree, int dof, real homogen,
		      real3 *field, real3 *source, real *phi, real *q)
{
  /* This is necessary only for compiling direct method */
  exit(EXIT_FAILURE);
}

/**************************************************************************/
#endif
/**************************************************************************/
