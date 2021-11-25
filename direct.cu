#ifndef DIRECT_CU
#define DIRECT_CU

#include "bbfmm.h"
#ifdef __DEVICE_EMULATION__
#include <assert.h>
#endif

#ifndef DEGREE_OF_FREEDOM
#define DEGREE_OF_FREEDOM (1)
#else
#if (DEGREE_OF_FREEDOM != 1)
#error DEGREE_OF_FREEDOM is supposed to be one here.
#endif
#endif

/**************************************************************************/
#if defined(CUDA_VER46)
/**************************************************************************/
/**************************************************************************/
#if defined(CUDA_VER46Q)
/**************************************************************************/
/* based on CUDA_VER46L */

/* Definition of pairwise interaction kernel */
/* Compute 1/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)
/* Compute r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv
/* Compute 1/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)
#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real dy = (s).y - (f).y;				\
    real dz = (s).z - (f).z;				\
    real rr = dx * dx + dy * dy + dz * dz;		\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }

#define ACCUMULATE_INTERACTION2(p, f, s)		\
  {							\
    real dx = (s).x - (f).x;				\
    real dy = (s).y - (f).y;				\
    real dz = (s).z - (f).z;				\
    real rr = dx * dx + dy * dy + dz * dz;		\
    COMP_KERNEL();					\
    (p) += kern * (s).w;				\
  }

#define bx blockIdx.x
#define tx threadIdx.x
#define bDx blockDim.x // number of threads per thread-block

#if !defined(DIRECT_SHARE_SIZE)
#define DIRECT_SHARE_SIZE (64)
#endif

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
  assert(bDx >= DIRECT_SHARE_SIZE);
#endif

  /* Obtain the index of the current cell (leaf) A, to which the
     current thread-block is assigned */
  int A = A0 + bx;

  /* Obtain the offset address for neighbors[] */
  int offset = pitch_neighbors * A;

  /* Obtain the number of neighbors of A */
  int nneighbor = ineigh[A];
  
  /* Obtain the indexes of the first and last particles in A */
  int ista = liststa[A];
  int iend = listend[A];

  /* Obtain the number of particles in A */
  int ni = iend - ista + 1;
 
  /* Every bDx particles in A are processed by bDx threads. Compute
     the number of such groups of particles in A */
  int nigroup = (ni / bDx) + (ni % bDx == 0 ? 0 : 1);
  
  /* Share the information of neighbors of A */
  __shared__ int jsta[27], nj[27], Bs[27];
  for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
    /* Obtain the index of neighbor */
    int B = neighbors[offset + ineighbor];
    Bs[ineighbor] = B;

    /* Share the index of the first particle in B */
    jsta[ineighbor] = liststa[B];
      
    /* Share the number of particles in B */
    nj[ineighbor] = listend[B] - liststa[B] + 1;
  }
  __syncthreads();

  /* Loop over groups */
  for (int igroup = 0; igroup < nigroup; igroup ++) {

    /* Assign the tx-th thread to the i-th particle, say f, in A */
    int i = igroup * bDx + tx;

    /* Load the position and charge of f */
    real4 xf;
    if (i < ni) { // screening existing particle
      xf = x[ista + i];
    } else {
      xf.x = xf.y = xf.z = xf.w = ZERO; // set dummy
    }

    /* Initialize the field value of f (loop over j is unrolled
       bellow) */
    real phi0 = ZERO;
    real phi1 = ZERO;
    real phi2 = ZERO;
    real phi3 = ZERO;
  
    /* Loop over neighbors of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      /* Every DIRECT_SHARE_SIZE particles in B are shared by the
	 thread-block.  Compute the number of such groups of
	 particles in B */
      int njgroup = (nj[ineighbor] / DIRECT_SHARE_SIZE) + (nj[ineighbor] % DIRECT_SHARE_SIZE == 0 ? 0 : 1);
      
      /* Loop over groups of particles in B */
      for (int jgroup = 0; jgroup < njgroup; jgroup ++) {
	
	/* Obtain the number of particles in this group */
	int nk = MIN(DIRECT_SHARE_SIZE, nj[ineighbor] - jgroup * DIRECT_SHARE_SIZE);
	
	/* Share the particles in this group. Ensure
	   bDx>=DIRECT_SHARE_SIZE */
	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	if (tx < nk) {
	  xs[tx] = x[jsta[ineighbor] + jgroup * DIRECT_SHARE_SIZE + tx];
	}
	__syncthreads();

	if (Bs[ineighbor] != A) {
	  
	  /* Loop over the particles in this group */
	  for (int k = 0; k < nk - 3; k += 4) { // unrolling x4
	    
	    /* Accumulate the interaction between f and s */
	    ACCUMULATE_INTERACTION2(phi0, xf, xs[k]);
	    ACCUMULATE_INTERACTION2(phi1, xf, xs[k + 1]);
	    ACCUMULATE_INTERACTION2(phi2, xf, xs[k + 2]);
	    ACCUMULATE_INTERACTION2(phi3, xf, xs[k + 3]);
	    
	  } // k
	  
	  for (int k = (nk / 4) * 4; k < nk; k ++) { // if nk mod 4 is not zero
	    ACCUMULATE_INTERACTION2(phi0, xf, xs[k]);
	  }
	  
	} else { // B==A

	  /* Loop over the particles in this group */
	  for (int k = 0; k < nk - 3; k += 4) { // unrolling x4
	    
	    /* Accumulate the interaction between f and s */
	    ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	    ACCUMULATE_INTERACTION(phi1, xf, xs[k + 1]);
	    ACCUMULATE_INTERACTION(phi2, xf, xs[k + 2]);
	    ACCUMULATE_INTERACTION(phi3, xf, xs[k + 3]);
	    
	  } // k
	  
	  for (int k = (nk / 4) * 4; k < nk; k ++) { // if nk mod 4 is not zero
	    ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	  }

	}

	/* Confirm all the threads no longer use xs[] */
	__syncthreads();
	
      } // jgroup
      
    } // ineighbor
    
    /* Store the field value of f on device memory */
    if (i < ni) { // screening existing particle
      p[ista + i] += xf.w * (phi0 + phi1 + phi2 + phi3);
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46P)
/**************************************************************************/
/* based on CUDA_VER46N */

/* Defintion of pairwise interaction kernel */
/* Compute 1/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)
/* Compute r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv
/* Compute 1/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)
#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }

#define bx blockIdx.x
#define tx threadIdx.x
#define bDx blockDim.x // number of threads per thread-block

#if !defined(DIRECT_SHARE_SIZE)
#define DIRECT_SHARE_SIZE (64)
#endif

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
#endif

  /* Obtain the index of the current cell (leaf) A, to which the
     current thread-block is assigned */
  const int A = A0 + bx;

  //111107  /* Obtain the offset address for neighbors[] */
  //111107  const int offset = pitch_neighbors * A;

  /* Obtain the offset neighbors[] for A */
  const int *neighborsA = &(neighbors[pitch_neighbors * A]);

  /* Obtain the number of neighbors of A */
  //111107  const int nneighbor = ineigh[A];
  const int ineighA = ineigh[A];
  
  /* Obtain the indexes of the first and last particles in A */
  const int ista = liststa[A];
  const int iend = listend[A];

  /* Obtain the number of particles in A */
  const int ni = iend - ista + 1;

  /* Every bDx particles in A are processed by bDx threads. Compute
     the number of such groups of particles in A */
  const int nigroup = (ni / bDx) + (ni % bDx == 0 ? 0 : 1);
  
  //111107  /* Share the information of neighbors of A */
  //111107  __shared__ int jsta[27], nj[27];
  //111107  for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
  //111107      
  //111107    /* Obtain the index of neighbor */
  //111107    const int B = neighbors[offset + ineighbor];
  //111107
  //111107    /* Share the index of the first particle in B */
  //111107    jsta[ineighbor] = liststa[B];
  //111107      
  //111107    /* Share the number of particles in B */
  //111107    nj[ineighbor] = listend[B] - liststa[B] + 1;
  //111107  }
  //111107  __syncthreads();


#error This implmentation does not work if there are any empty cells in neighbors[A].

  /* Compute the reduced neighbor list of A; the maximum value of
     nneighbor is probablly 14 */
  __shared__ int jsta[27], nj[27]; //jend[27];
  int nneighbor = 0;
  int P = neighborsA[0]; // i=0
  jsta[nneighbor] = liststa[P];
  for (int i = 1; i < ineighA; i ++) {
    int B = neighborsA[i];
    if (B != P + 1) { // Unless B is the next of P
      nj[nneighbor] = listend[P] - jsta[nneighbor] + 1; //jend[nneighbor] = listend[P];
      nneighbor ++;
      jsta[nneighbor] = liststa[B];
    }
    P = B;
  }
  nj[nneighbor] = listend[P] - jsta[nneighbor] + 1; //jend[nneighbor] = listend[P];
  nneighbor ++;
  __syncthreads();


  /* Loop over groups */
  for (int igroup = 0; igroup < nigroup; igroup ++) {

    /* Assign the tx-th thread to the i-th particle, say f, in A */
    const int i = igroup * bDx + tx;

    /* Load the position and charge of f */
    real4 xf;
    if (i < ni) {
      xf = x[ista + i];
    } else {
      xf.x = xf.y = xf.z = xf.w = ZERO; // set dummy
    }

    /* Initialize the field value of f (loop over j is unrolled
       bellow) */
    real phi0 = ZERO;
    real phi1 = ZERO;
    real phi2 = ZERO;
    real phi3 = ZERO;
  
    /* Loop over neighbors of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      /* Every DIRECT_SHARE_SIZE particles in B are shared by the
	 thread-block.  Compute the number of such groups of particles
	 in B */
      const int njgroup = (nj[ineighbor] / DIRECT_SHARE_SIZE) + (nj[ineighbor] % DIRECT_SHARE_SIZE == 0 ? 0 : 1);

      /* Loop over groups of particles in B */
      for (int jgroup = 0; jgroup < njgroup; jgroup ++) {

	/* Obtain the number of particles in this group */
	const int nk = MIN(DIRECT_SHARE_SIZE, nj[ineighbor] - jgroup * DIRECT_SHARE_SIZE);

	/* Share the particles in this group (bDx>=DIRECT_SHARE_SIZE is unnecessary) */
	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	const int nread = (DIRECT_SHARE_SIZE / bDx) + (DIRECT_SHARE_SIZE % bDx == 0 ? 0 : 1);
	for (int l = 0; l < nread - 1; l ++) {
	  xs[bDx * l + tx] = x[jsta[ineighbor] + jgroup * DIRECT_SHARE_SIZE + bDx * l + tx];
	}
	const int l = nread - 1;
	if (bDx * l + tx < nk) {
	  xs[bDx * l + tx] = x[jsta[ineighbor] + jgroup * DIRECT_SHARE_SIZE + bDx * l + tx];
	}
	__syncthreads();

	/* Loop over the particles in this group */
	for (int k = 0; k < nk - 3; k += 4) { // unrolling x4
	
	  /* Accumulate the interaction between f and s */
	  ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	  ACCUMULATE_INTERACTION(phi1, xf, xs[k + 1]);
	  ACCUMULATE_INTERACTION(phi2, xf, xs[k + 2]);
	  ACCUMULATE_INTERACTION(phi3, xf, xs[k + 3]);
	
	} // k

	for (int k = (nk / 4) * 4; k < nk; k ++) { // for when nk mod 4 is not zero
	  ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	}

	/* Confirm all the threads no longer use xs[] */
	__syncthreads();

      } // jgroup

    } // ineighbor
    
    /* Store the field value of f on device memory */
    if (i < ni) {
      p[ista + i] += xf.w * (phi0 + phi1 + phi2 + phi3);
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46O)
/**************************************************************************/
/* Skip this minor-version */
#error Not implemented yet.
/**************************************************************************/
#elif defined(CUDA_VER46N)
/**************************************************************************/
/* based on CUDA_VER46L */

/* Defintion of pairwise interaction kernel */
/* Compute 1/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)
/* Compute r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv
/* Compute 1/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)
#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }

#define bx blockIdx.x
#define tx threadIdx.x
#define bDx blockDim.x // number of threads per thread-block

#if !defined(DIRECT_SHARE_SIZE)
#define DIRECT_SHARE_SIZE (64)
#endif

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
  //111107  assert(bDx >= DIRECT_SHARE_SIZE);
#endif

  /* Obtain the index of the current cell (leaf) A, to which the
     current thread-block is assigned */
  const int A = A0 + bx;

  /* Obtain the offset address for neighbors[] */
  const int offset = pitch_neighbors * A;

  /* Obtain the number of neighbors of A */
  const int nneighbor = ineigh[A];
  
  /* Obtain the indexes of the first and last particles in A */
  const int ista = liststa[A];
  const int iend = listend[A];

  /* Obtain the number of particles in A */
  const int ni = iend - ista + 1;
 
  /* Every bDx particles in A are processed by bDx threads. Compute
     the number of such groups of particles in A */
  const int nigroup = (ni / bDx) + (ni % bDx == 0 ? 0 : 1);
  
  /* Share the information of neighbors of A */
  __shared__ int jsta[27], nj[27];
  for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
    /* Obtain the index of neighbor */
    const int B = neighbors[offset + ineighbor];

    /* Share the index of the first particle in B */
    jsta[ineighbor] = liststa[B];
      
    /* Share the number of particles in B */
    nj[ineighbor] = listend[B] - liststa[B] + 1;
  }
  __syncthreads();

  /* Loop over groups */
  for (int igroup = 0; igroup < nigroup; igroup ++) {

    /* Assign the tx-th thread to the i-th particle, say f, in A */
    const int i = igroup * bDx + tx;

    /* Load the position and charge of f */
    real4 xf;
    if (i < ni) {
      xf = x[ista + i];
    } else {
      xf.x = xf.y = xf.z = xf.w = ZERO; // set dummy
    }

    /* Initialize the field value of f (loop over j is unrolled
       bellow) */
    real phi0 = ZERO;
    real phi1 = ZERO;
    real phi2 = ZERO;
    real phi3 = ZERO;
  
    /* Loop over neighbors of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      /* Every DIRECT_SHARE_SIZE particles in B are shared by the
	 thread-block.  Compute the number of such groups of particles
	 in B */
      const int njgroup = (nj[ineighbor] / DIRECT_SHARE_SIZE) + (nj[ineighbor] % DIRECT_SHARE_SIZE == 0 ? 0 : 1);

      /* Loop over groups of particles in B */
      for (int jgroup = 0; jgroup < njgroup; jgroup ++) {

	/* Obtain the number of particles in this group */
	const int nk = MIN(DIRECT_SHARE_SIZE, nj[ineighbor] - jgroup * DIRECT_SHARE_SIZE);

	//111107	/* Share the particles in this group. Ensure
	//111107	   bDx>=DIRECT_SHARE_SIZE */
	//111107	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	//111107	if (tx < nk) {
	//111107	  xs[tx] = x[jsta[ineighbor] + jgroup * DIRECT_SHARE_SIZE + tx];
	//111107	}
	//111107	__syncthreads();

	/* Share the particles in this group (bDx>=DIRECT_SHARE_SIZE is unnecessary) */
	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	const int nread = (DIRECT_SHARE_SIZE / bDx) + (DIRECT_SHARE_SIZE % bDx == 0 ? 0 : 1);
	for (int l = 0; l < nread - 1; l ++) {
	  xs[bDx * l + tx] = x[jsta[ineighbor] + jgroup * DIRECT_SHARE_SIZE + bDx * l + tx];
	}
	const int l = nread - 1;
	if (bDx * l + tx < nk) {
	  xs[bDx * l + tx] = x[jsta[ineighbor] + jgroup * DIRECT_SHARE_SIZE + bDx * l + tx];
	}
	__syncthreads();

	/* Loop over the particles in this group */
	for (int k = 0; k < nk - 3; k += 4) { // unrolling x4
	
	  /* Accumulate the interaction between f and s */
	  ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	  ACCUMULATE_INTERACTION(phi1, xf, xs[k + 1]);
	  ACCUMULATE_INTERACTION(phi2, xf, xs[k + 2]);
	  ACCUMULATE_INTERACTION(phi3, xf, xs[k + 3]);
	
	} // k

	for (int k = (nk / 4) * 4; k < nk; k ++) { // for when nk mod 4 is not zero
	  ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	}

	/* Confirm all the threads no longer use xs[] */
	__syncthreads();

      } // jgroup

    } // ineighbor
    
    /* Store the field value of f on device memory */
    if (i < ni) {
      p[ista + i] += xf.w * (phi0 + phi1 + phi2 + phi3);
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46M)
/**************************************************************************/
/* based on CUDA_VER46L */

/* Defintion of pairwise interaction kernel */
/* Compute 1/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)
/* Compute r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv
/* Compute 1/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)
#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }

#define bx blockIdx.x
#define tx threadIdx.x
#define bDx blockDim.x // number of threads per thread-block

#if !defined(DIRECT_SHARE_SIZE)
#define DIRECT_SHARE_SIZE (64)
#endif

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
  assert(bDx >= DIRECT_SHARE_SIZE);
#endif

  /* Obtain the index of the current cell (leaf) A, to which the
     current thread-block is assigned */
  int A = A0 + bx;

  /* Obtain the offset address for neighbors[] */
  const int offset = pitch_neighbors * A;

  /* Obtain the number of neighbors of A */
  const int nneighbor = ineigh[A];
  
  /* Obtain the indexes of the first and last particles in A */
  const int ista = liststa[A];
  const int iend = listend[A];

  /* Obtain the number of particles in A */
  const int ni = iend - ista + 1;
 
  /* Every "2*bDx" particles in A are processed by bDx
     threads. Compute the number of such groups of particles in A */
  const int nigroup = (ni / (2 * bDx)) + (ni % (2 * bDx) == 0 ? 0 : 1);
  
  /* Share the information of neighbors of A */
  __shared__ int jsta[27], nj[27];
  for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
    /* Obtain the index of neighbor */
    const int B = neighbors[offset + ineighbor];

    /* Share the index of the first particle in B */
    jsta[ineighbor] = liststa[B];
      
    /* Share the number of particles in B */
    nj[ineighbor] = listend[B] - liststa[B] + 1;
  }
  __syncthreads();

  /* Loop over groups */
  for (int igroup = 0; igroup < nigroup; igroup ++) {

    /* Assign the tx-th thread to the i-th and (i+1)-th particles, say
       f0 and f1, in A */
    const int i = igroup * (2 * bDx) + 2 * tx;

    /* Load the positions and charges of f0 and f1 */
    real4 xf0;
    if (i < ni) {
      xf0 = x[ista + i];
    } else {
      xf0.x = xf0.y = xf0.z = xf0.w = ZERO; // set dummy
    }
    real4 xf1;
    if (i + 1 < ni) {
      xf1 = x[ista + i + 1];
    } else {
      xf1.x = xf1.y = xf1.z = xf1.w = ZERO; // set dummy
    }

    /* Initialize the field values of f0 and f1 (loop over j is
       unrolled bellow) */
    real p00 = ZERO;
    real p01 = ZERO;
    real p02 = ZERO;
    real p03 = ZERO;
    real p10 = ZERO;
    real p11 = ZERO;
    real p12 = ZERO;
    real p13 = ZERO;
  
    /* Loop over neighbors of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      /* Every DIRECT_SHARE_SIZE particles in B are shared by the
	 thread-block.  Compute the number of such groups of particles
	 in B */
      const int njgroup = (nj[ineighbor] / DIRECT_SHARE_SIZE) + (nj[ineighbor] % DIRECT_SHARE_SIZE == 0 ? 0 : 1);

      /* Loop over groups of particles in B */
      for (int jgroup = 0; jgroup < njgroup; jgroup ++) {

	/* Obtain the number of particles in this group */
	const int nk = MIN(DIRECT_SHARE_SIZE, nj[ineighbor] - jgroup * DIRECT_SHARE_SIZE);

	/* Share the particles in this group. Ensure
	   bDx>=DIRECT_SHARE_SIZE */
	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	if (tx < nk) {
	  xs[tx] = x[jsta[ineighbor] + jgroup * DIRECT_SHARE_SIZE + tx];
	}
	__syncthreads();

	/* Loop over the particles in this group */
	for (int k = 0; k < nk - 3; k += 4) { // unrolling x4
	
	  /* Accumulate the interaction between f0 and s */
	  ACCUMULATE_INTERACTION(p00, xf0, xs[k]);
	  ACCUMULATE_INTERACTION(p01, xf0, xs[k + 1]);
	  ACCUMULATE_INTERACTION(p02, xf0, xs[k + 2]);
	  ACCUMULATE_INTERACTION(p03, xf0, xs[k + 3]);

	  /* Accumulate the interaction between f1 and s */
	  ACCUMULATE_INTERACTION(p10, xf1, xs[k]);
	  ACCUMULATE_INTERACTION(p11, xf1, xs[k + 1]);
	  ACCUMULATE_INTERACTION(p12, xf1, xs[k + 2]);
	  ACCUMULATE_INTERACTION(p13, xf1, xs[k + 3]);
	
	} // k

	for (int k = (nk / 4) * 4; k < nk; k ++) { // only when nk%4!=0
	  ACCUMULATE_INTERACTION(p00, xf0, xs[k]);
	  ACCUMULATE_INTERACTION(p10, xf1, xs[k]);
	}

	/* Confirm all the threads no longer use xs[] */
	__syncthreads();

      } // jgroup

    } // ineighbor
    
    /* Store the field values of f0 and f1 on device memory */
    if (i < ni) {
      p[ista + i    ] += xf0.w * (p00 + p01 + p02 + p03);
    }
    if (i + 1 < ni) {
      p[ista + i + 1] += xf1.w * (p10 + p11 + p12 + p13);
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46L)
/**************************************************************************/
/* based on CUDA_VER46J */

/* Defintion of pairwise interaction kernel */
/* Compute 1/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)
/* Compute r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv
/* Compute 1/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)
#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }

#define bx blockIdx.x
#define tx threadIdx.x
#define bDx blockDim.x // number of threads per thread-block

#if !defined(DIRECT_SHARE_SIZE)
#define DIRECT_SHARE_SIZE (64)
#endif

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
  assert(bDx >= DIRECT_SHARE_SIZE);
#endif

  /* Obtain the index of the current cell (leaf) A, to which the
     current thread-block is assigned */
  int A = A0 + bx;

  /* Obtain the offset address for neighbors[] */
  int offset = pitch_neighbors * A;

  /* Obtain the number of neighbors of A */
  int nneighbor = ineigh[A];
  
  /* Obtain the indexes of the first and last particles in A */
  int ista = liststa[A];
  int iend = listend[A];

  /* Obtain the number of particles in A */
  int ni = iend - ista + 1;
 
  /* Every bDx particles in A are processed by bDx threads. Compute
     the number of such groups of particles in A */
  int nigroup = (ni / bDx) + (ni % bDx == 0 ? 0 : 1);
  
  /* Share the information of neighbors of A */
  __shared__ int jsta[27], nj[27];
  for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
    /* Obtain the index of neighbor */
    int B = neighbors[offset + ineighbor];

    /* Share the index of the first particle in B */
    jsta[ineighbor] = liststa[B];
      
    /* Share the number of particles in B */
    nj[ineighbor] = listend[B] - liststa[B] + 1;
  }
  __syncthreads();

  /* Loop over groups */
  for (int igroup = 0; igroup < nigroup; igroup ++) {

    /* Assign the tx-th thread to the i-th particle, say f, in A */
    int i = igroup * bDx + tx;

    /* Load the position and charge of f */
    real4 xf;
    if (i < ni) { // screening existing particle
      xf = x[ista + i];
    } else {
      xf.x = xf.y = xf.z = xf.w = ZERO; // set dummy
    }

    /* Initialize the field value of f (loop over j is unrolled
       bellow) */
    real phi0 = ZERO;
    real phi1 = ZERO;
    real phi2 = ZERO;
    real phi3 = ZERO;
  
    /* Loop over neighbors of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      //110919      /* Obtain the index of neighbor */
      //110919      int B = neighbors[offset + ineighbor];
      //110919      
      //110919      /* Obtain the indexes of the first and last particle in B */
      //110919      int jsta = liststa[B];
      //110919      int jend = listend[B];
      //110919
      //110919      /* Obtain the number of particles in B */
      //110919      int nj = jend - jsta + 1;

      /* Every DIRECT_SHARE_SIZE particles in B are shared by the
	  thread-block.  Compute the number of such groups of
	  particles in B */
      //110919      int njgroup = (nj / DIRECT_SHARE_SIZE) + (nj % DIRECT_SHARE_SIZE == 0 ? 0 : 1);
      int njgroup = (nj[ineighbor] / DIRECT_SHARE_SIZE) + (nj[ineighbor] % DIRECT_SHARE_SIZE == 0 ? 0 : 1);

      /* Loop over groups of particles in B */
      for (int jgroup = 0; jgroup < njgroup; jgroup ++) {

	/* Obtain the number of particles in this group */
	//110919	int nk = MIN(DIRECT_SHARE_SIZE, nj - jgroup * DIRECT_SHARE_SIZE);
	int nk = MIN(DIRECT_SHARE_SIZE, nj[ineighbor] - jgroup * DIRECT_SHARE_SIZE);

	/* Share the particles in this group. Ensure
	   bDx>=DIRECT_SHARE_SIZE */
	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	if (tx < nk) {
	  //110919	  xs[tx] = x[jsta + jgroup * DIRECT_SHARE_SIZE + tx];
	  xs[tx] = x[jsta[ineighbor] + jgroup * DIRECT_SHARE_SIZE + tx];
	}
	__syncthreads();

	/* Loop over the particles in this group */
	for (int k = 0; k < nk - 3; k += 4) { // unrolling x4
	
	  /* Accumulate the interaction between f and s */
	  ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	  ACCUMULATE_INTERACTION(phi1, xf, xs[k + 1]);
	  ACCUMULATE_INTERACTION(phi2, xf, xs[k + 2]);
	  ACCUMULATE_INTERACTION(phi3, xf, xs[k + 3]);
	
	} // k

	for (int k = (nk / 4) * 4; k < nk; k ++) { // for when nk mod 4 is not zero
	  ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	}

	/* Confirm all the threads no longer use xs[] */
	__syncthreads();

      } // jgroup

    } // ineighbor
    
    /* Store the field value of f on device memory */
    if (i < ni) { // screening existing particle
      p[ista + i] += xf.w * (phi0 + phi1 + phi2 + phi3);
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46K)
/**************************************************************************/
/* based on CUDA_VER46H */

/* Compute q/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)

/* Compute q*r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv

/* Compute q/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)

#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }


#define bx blockIdx.x
#define tx threadIdx.x

#define bDx blockDim.x

#if !defined(DIRECT_SHARE_SIZE)
#define DIRECT_SHARE_SIZE (32)
#endif

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
  assert(bDx >= DIRECT_SHARE_SIZE);
#endif

  /* Obtain the index of the current field-cell A */
  int A = A0 + bx;

  /* Preload the offset address for neighbors[] */
  int offset = pitch_neighbors * A;

  /* Preload the number of neighbour-cells (or source-cells) of A */
  int nneighbor = ineigh[A];
  
  /* Preload the start and last addresses for A */
  int ista = liststa[A];
  int iend = listend[A];

  /* Obtain the number of particles in A */
  int ni = iend - ista + 1;
 
  //  /* Every bDx field-particles is handled as one group. Compute the
  //     number of such groups. */
  //  int nigroup = (ni / bDx) + (ni % bDx == 0 ? 0 : 1);

  /* Every 2*bDx field-particles is handled as one group. Compute the
     number of such groups. */
  int bDx2 = 2 * bDx;
  int nigroup = (ni / bDx2) + (ni % bDx2 == 0 ? 0 : 1);

  /* Loop over groups of field-particles */
  for (int igroup = 0; igroup < nigroup; igroup ++) {

    //    /* Consider the i-th particle if it exists */
    //    int i = igroup * bDx + tx;

    /* Assign the current thread to i-th and (i+1)-th particle */
    int i = igroup * bDx2 + tx * 2;

    //    /* Load the position and charge of f */
    //    real4 xf;
    //    if (i < ni) { // screening
    //      xf = x[ista + i];
    //    } else {
    //      xf.x = xf.y = xf.z = xf.w = ZERO; // dummy
    //    }

    /* Load the position and charge of i-th particle (f0) and (i+1)-th
       particle (f1) */
    real4 xf0;
    if (i < ni) { // screening
      xf0 = x[ista + i];
    } else {
      xf0.x = xf0.y = xf0.z = xf0.w = ZERO; // dummy
    }

    real4 xf1;
    if (i + 1 < ni) { // screening
      xf1 = x[ista + i + 1];
    } else {
      xf1.x = xf1.y = xf1.z = xf1.w = ZERO; // dummy
    }

    //    /* Initialise the field value of f */
    //    real phi = ZERO;

    /* Initialise the field value of f0 and f1 */
    real phi0 = ZERO;
    real phi1 = ZERO;
  
    /* Loop over neighbour-cells of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      /* Obtain the index of neighbour-cell */
      int B = neighbors[offset + ineighbor];
      
      /* Obtain the start and last addresses for B */
      int jsta = liststa[B];
      int jend = listend[B];

      /* Obtain the number of particles in B */
      int nj = jend - jsta + 1;

      /* Every DIRECT_SHARE_SIZE source-particles is handled as one
	 group. Compute the number of such groups. */
      int njgroup = (nj / DIRECT_SHARE_SIZE) + (nj % DIRECT_SHARE_SIZE == 0 ? 0 : 1);

      /* Loop over groups of source-particles */
      for (int jgroup = 0; jgroup < njgroup; jgroup ++) {

	/* Obtain the number of source-particles in this group */
	int nk = MIN(DIRECT_SHARE_SIZE, nj - jgroup * DIRECT_SHARE_SIZE);

	/* Share source-particles in this group. Ensure
	   bDx>=DIRECT_SHARE_SIZE */
	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	if (tx < nk) {
	  xs[tx] = x[jsta + jgroup * DIRECT_SHARE_SIZE + tx];
	}
	__syncthreads();

	/* Loop over source-particles in this group */
	for (int k = 0; k < nk; k ++) {
	
	  //	  /* Accumulate the interaction between f and s */
	  //	  ACCUMULATE_INTERACTION(phi, xf, xs[k]);

	  /* Accumulate the interaction between f0 and s */
	  ACCUMULATE_INTERACTION(phi0, xf0, xs[k]);

	  /* Accumulate the interaction between f1 and s */
	  ACCUMULATE_INTERACTION(phi1, xf1, xs[k]);
	
	} // j

	/* Confirm all the threads no longer use xs[] */
	__syncthreads();

      } // jgroup

    } // ineighbour
    
    //    /* Store the field value of f on device memory */
    //    if (i < ni) { // screening
    //      p[ista + i] += xf.w * phi;
    //    }

    /* Store the field values of f0 and f1 on device memory */
    if (i < ni) { // screening
      p[ista + i] += xf0.w * phi0;
    }
    if (i + 1 < ni) { // screening
      p[ista + i + 1] += xf1.w * phi1;
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46J)
/**************************************************************************/
/* based on CUDA_VER46I */

/* Defintion of pairwise interaction kernel */
/* Compute 1/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)
/* Compute r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv
/* Compute 1/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)
#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }

#define bx blockIdx.x
#define tx threadIdx.x
#define bDx blockDim.x // number of threads per thread-block

#if !defined(DIRECT_SHARE_SIZE)
#define DIRECT_SHARE_SIZE (64)
#endif

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
  assert(bDx >= DIRECT_SHARE_SIZE);
#endif

  /* Obtain the index of the current cell (leaf) A, to which the
     current thread-block is assigned */
  int A = A0 + bx;

  /* Obtain the offset address for neighbors[] */
  int offset = pitch_neighbors * A;

  /* Obtain the number of neighbors of A */
  int nneighbor = ineigh[A];
  
  /* Obtain the indexes of the first and last particles in A */
  int ista = liststa[A];
  int iend = listend[A];

  /* Obtain the number of particles in A */
  int ni = iend - ista + 1;
 
  /* Every bDx particles in A are processed by bDx threads. Compute
     the number of such groups of particles in A */
  int nigroup = (ni / bDx) + (ni % bDx == 0 ? 0 : 1);

  /* Loop over groups */
  for (int igroup = 0; igroup < nigroup; igroup ++) {

    /* Assign the tx-th thread to the i-th particle, say f, in A */
    int i = igroup * bDx + tx;

    /* Load the position and charge of f */
    real4 xf;
    if (i < ni) { // screening existing particle
      xf = x[ista + i];
    } else {
      xf.x = xf.y = xf.z = xf.w = ZERO; // set dummy
    }

    /* Initialize the field value of f (loop over j is unrolled
       bellow) */
    real phi0 = ZERO;
    real phi1 = ZERO;
    real phi2 = ZERO;
    real phi3 = ZERO;
  
    /* Loop over neighbors of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      /* Obtain the index of neighbor */
      int B = neighbors[offset + ineighbor];
      
      /* Obtain the indexes of the first and last particles in B */
      int jsta = liststa[B];
      int jend = listend[B];

      /* Obtain the number of particles in B */
      int nj = jend - jsta + 1;

      /* Every DIRECT_SHARE_SIZE particles in B are shared by the
	  thread-block.  Compute the number of such groups of
	  particles in B */
      int njgroup = (nj / DIRECT_SHARE_SIZE) + (nj % DIRECT_SHARE_SIZE == 0 ? 0 : 1);

      /* Loop over groups of particles in B */
      for (int jgroup = 0; jgroup < njgroup; jgroup ++) {

	/* Obtain the number of particles in this group */
	int nk = MIN(DIRECT_SHARE_SIZE, nj - jgroup * DIRECT_SHARE_SIZE);

	/* Share the particles in this group. Ensure
	   bDx>=DIRECT_SHARE_SIZE */
	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	if (tx < nk) {
	  xs[tx] = x[jsta + jgroup * DIRECT_SHARE_SIZE + tx];
	}
	__syncthreads();

	/* Loop over the particles in this group */
	for (int k = 0; k < nk - 3; k += 4) { // unrolling x4
	
	  /* Accumulate the interaction between f and s */
	  ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	  ACCUMULATE_INTERACTION(phi1, xf, xs[k + 1]);
	  ACCUMULATE_INTERACTION(phi2, xf, xs[k + 2]);
	  ACCUMULATE_INTERACTION(phi3, xf, xs[k + 3]);
	
	} // k

	for (int k = (nk / 4) * 4; k < nk; k ++) { // for when nk mod 4 is not zero
	  ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	}

	/* Confirm all the threads no longer use xs[] */
	__syncthreads();

      } // jgroup

    } // ineighbor
    
    /* Store the field value of f on device memory */
    if (i < ni) { // screening existing particle
      p[ista + i] += xf.w * (phi0 + phi1 + phi2 + phi3);
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46I)
/**************************************************************************/
/* based on CUDA_VER46H */

/* Compute q/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)

/* Compute q*r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv

/* Compute q/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)

#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }


#define bx blockIdx.x
#define tx threadIdx.x

#define bDx blockDim.x

#if !defined(DIRECT_SHARE_SIZE)
//#define DIRECT_SHARE_SIZE (32)
#define DIRECT_SHARE_SIZE (64)
//#define DIRECT_SHARE_SIZE (128)
#endif

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
  assert(bDx >= DIRECT_SHARE_SIZE);
#endif

  /* Obtain the index of the current field-cell A */
  int A = A0 + bx;

  /* Preload the offset address for neighbors[] */
  int offset = pitch_neighbors * A;

  /* Preload the number of neighbour-cells (or source-cells) of A */
  int nneighbor = ineigh[A];
  
  /* Preload the start and last addresses for A */
  int ista = liststa[A];
  int iend = listend[A];

  /* Obtain the number of particles in A */
  int ni = iend - ista + 1;
 
  /* Every bDx field-particles is handled as one group. Compute the
     number of such groups. */
  int nigroup = (ni / bDx) + (ni % bDx == 0 ? 0 : 1);

  /* Loop over groups of field-particles */
  for (int igroup = 0; igroup < nigroup; igroup ++) {

    /* Consider the i-th particle if it exists */
    int i = igroup * bDx + tx;

    /* Load the position and charge of f */
    real4 xf;
    if (i < ni) { // screening
      xf = x[ista + i];
    } else {
      xf.x = xf.y = xf.z = xf.w = ZERO; // dummy
    }

    /* Initialise the field value of f */
    //    real phi = ZERO;
    real phi0 = ZERO;
    real phi1 = ZERO;
  
    /* Loop over neighbour-cells of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      /* Obtain the index of neighbour-cell */
      int B = neighbors[offset + ineighbor];
      
      /* Obtain the start and last addresses for B */
      int jsta = liststa[B];
      int jend = listend[B];

      /* Obtain the number of particles in B */
      int nj = jend - jsta + 1;

      /* Every DIRECT_SHARE_SIZE source-particles is handled as one
	 group. Compute the number of such groups. */
      int njgroup = (nj / DIRECT_SHARE_SIZE) + (nj % DIRECT_SHARE_SIZE == 0 ? 0 : 1);

      /* Loop over groups of source-particles */
      for (int jgroup = 0; jgroup < njgroup; jgroup ++) {

	/* Obtain the number of source-particles in this group */
	int nk = MIN(DIRECT_SHARE_SIZE, nj - jgroup * DIRECT_SHARE_SIZE);

	/* Share source-particles in this group. Ensure
	   bDx>=DIRECT_SHARE_SIZE */
	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	if (tx < nk) {
	  xs[tx] = x[jsta + jgroup * DIRECT_SHARE_SIZE + tx];
	}
	__syncthreads();

	/* Loop over source-particles in this group */
	//	for (int k = 0; k < nk; k ++) {
	for (int k = 0; k < nk - 1; k += 2) { // unrolling x2
	
	  /* Accumulate the interaction between f and s */
	  //	  ACCUMULATE_INTERACTION(phi, xf, xs[k]);
	  ACCUMULATE_INTERACTION(phi0, xf, xs[k]);
	  ACCUMULATE_INTERACTION(phi1, xf, xs[k + 1]);
	
	} // j

	if (nk % 2 == 1) { // computation for k=nk-1
	  ACCUMULATE_INTERACTION(phi0, xf, xs[nk - 1]);	  
	}

	/* Confirm all the threads no longer use xs[] */
	__syncthreads();

      } // jgroup

    } // ineighbour
    
    /* Store the field value of f on device memory */
    if (i < ni) { // screening
      //      p[ista + i] += xf.w * phi;
      p[ista + i] += xf.w * (phi0 + phi1);
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46H)
/**************************************************************************/
/* based on CUDA_VER46F */

/* Compute q/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)

/* Compute q*r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv

/* Compute q/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)

#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }


#define bx blockIdx.x
#define tx threadIdx.x

#define bDx blockDim.x

#if !defined(DIRECT_SHARE_SIZE)
#define DIRECT_SHARE_SIZE (32)
#endif

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
  assert(bDx >= DIRECT_SHARE_SIZE);
#endif

  /* Obtain the index of the current field-cell A */
  int A = A0 + bx;

  /* Preload the offset address for neighbors[] */
  int offset = pitch_neighbors * A;

  /* Preload the number of neighbour-cells (or source-cells) of A */
  int nneighbor = ineigh[A];
  
  /* Preload the start and last addresses for A */
  int ista = liststa[A];
  int iend = listend[A];

  /* Obtain the number of particles in A */
  int ni = iend - ista + 1;
 
  /* Every bDx field-particles is handled as one group. Compute the
     number of such groups. */
  int nigroup = (ni / bDx) + (ni % bDx == 0 ? 0 : 1);

  /* Loop over groups of field-particles */
  for (int igroup = 0; igroup < nigroup; igroup ++) {

    /* Consider the i-th particle if it exists */
    int i = igroup * bDx + tx;

    /* Load the position and charge of f */
    real4 xf;
    if (i < ni) { // screening
      xf = x[ista + i];
    } else {
      xf.x = xf.y = xf.z = xf.w = ZERO; // dummy
    }

    /* Initialise the field value of f */
    real phi = ZERO;
  
    /* Loop over neighbour-cells of A */
    for (int ineighbor = 0; ineighbor < nneighbor; ineighbor ++) {
      
      /* Obtain the index of neighbour-cell */
      int B = neighbors[offset + ineighbor];
      
      /* Obtain the start and last addresses for B */
      int jsta = liststa[B];
      int jend = listend[B];

      /* Obtain the number of particles in B */
      int nj = jend - jsta + 1;

      /* Every DIRECT_SHARE_SIZE source-particles is handled as one
	 group. Compute the number of such groups. */
      int njgroup = (nj / DIRECT_SHARE_SIZE) + (nj % DIRECT_SHARE_SIZE == 0 ? 0 : 1);

      /* Loop over groups of source-particles */
      for (int jgroup = 0; jgroup < njgroup; jgroup ++) {

	/* Obtain the number of source-particles in this group */
	int nk = MIN(DIRECT_SHARE_SIZE, nj - jgroup * DIRECT_SHARE_SIZE);

	/* Share source-particles in this group. Ensure
	   bDx>=DIRECT_SHARE_SIZE */
	__shared__ real4 xs[DIRECT_SHARE_SIZE];
	if (tx < nk) {
	  xs[tx] = x[jsta + jgroup * DIRECT_SHARE_SIZE + tx];
	}
	__syncthreads();

	/* Loop over source-particles in this group */
	for (int k = 0; k < nk; k ++) {
	
	  /* Accumulate the interaction between f and s */
	  ACCUMULATE_INTERACTION(phi, xf, xs[k]);
	
	} // j

	/* Confirm all the threads no longer use xs[] */
	__syncthreads();

      } // jgroup

    } // ineighbour
    
    /* Store the field value of f on device memory */
    if (i < ni) { // screening
      p[ista + i] += xf.w * phi;
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46G)
/**************************************************************************/
/* based on CUDA_VER46F */

/* Compute q/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)

/* Compute q*r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv

/* Compute q/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)

#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }


#define bx blockIdx.x
#define tx threadIdx.x

#define bDx blockDim.x

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
#endif

  /* Obtain the index of the current field-cell A */
  int A = A0 + bx;

  /* Preload the offset address for neighbors[] */
  int offset = pitch_neighbors * A;

  /* Preload the number of neighbour-cells (or source-cells) of A */
  int nneigh = ineigh[A];
  
  /* Preload the start and last addresses for A */
  int ista = liststa[A];
  int iend = listend[A];

  /* Obtain the number of particles in A */
  int nf = iend - ista + 1;
 
  /* Every bDx field-particles is handled in one group. Compute the
     number of such groups. */
  int ngroup = (nf / bDx) + (nf % bDx == 0 ? 0 : 1);

  /* Loop over groups of field-particles */
  for (int igroup = 0; igroup < ngroup; igroup ++) {

    /* Consider the i-th particle if it exists */
    int i = igroup * bDx + tx;

    /* Load the position and charge of f */
    real4 xf;
    if (i < nf) { // Screening
      xf = x[ista + i];
    } else {
      xf.x = xf.y = xf.z = xf.w = ZERO; // dummy
    }

    /* Initialise the field value of f */
    //    real phi = ZERO;
    real phi0 = ZERO, phi1 = ZERO;
  
    /* Loop over neighbour-cells of A */
    for (int ineighbor = 0; ineighbor < nneigh; ineighbor ++) {
      
      /* Obtain the index of neighbour-cell */
      int B = neighbors[offset + ineighbor];

      /* Preload the start and last addresses for B */
      int jsta = liststa[B];
      int jend = listend[B];

      /* Obtain the number of particles in B */
      int ns = jend - jsta + 1;

      //      /* Loop over particles in B */
      //      for (int j = liststa[B]; j <= listend[B]; j ++) {

      /* Loop over particles in B */
      for (int j = 0; j < ns - 1; j += 2) { // unrolling x2
	
	/* Load the position and charge of s */
	//	real4 xs = x[j];
	real4 xs0 = x[jsta + j];
	real4 xs1 = x[jsta + j + 1];
	
	/* Accumulate the interaction between f and s */
	//	ACCUMULATE_INTERACTION(phi, xf, xs);
	ACCUMULATE_INTERACTION(phi0, xf, xs0);
	ACCUMULATE_INTERACTION(phi1, xf, xs1);
	
      } // j

      if (ns % 2 == 1) { // computation for j=ns-1

	/* Load the position and charge of the last particle in B */
	real4 xs0 = x[jsta + ns - 1];
	
	/* Accumulate the interaction between f and s */
	ACCUMULATE_INTERACTION(phi0, xf, xs0);
	
      }
      
    } // ineighbour
    
    /* Store the field value of f on device memory */
    if (i < nf) { // Screening
      //      p[ista + i] += xf.w * phi;
      p[ista + i] += xf.w * (phi0 + phi1);
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46F)
/**************************************************************************/
/* based on CUDA_VER46E */

/* Compute 1/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)
/* Compute r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv
/* Compute 1/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)
#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }

#define bx blockIdx.x
#define tx threadIdx.x
#define bDx blockDim.x // number of threads per thread-block

__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       real *p, const real4 *x, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
#endif

  /* Obtain the index of the current cell (leaf) A, to which the
     current thread-block is assigned */
  int A = A0 + bx;

  /* Obtain the offset address for neighbors[] */
  int offset = pitch_neighbors * A;

  /* Obtain the number of neighbors of A */
  int nneigh = ineigh[A];
  
  /* Obtain the indexes of the first and last particles in A */
  int ista = liststa[A];
  int iend = listend[A];

  /* Obtain the number of particles in A */
  int nf = iend - ista + 1;
 
  /* Every bDx particles in A are processed by bDx threads. Compute
     the number of such groups of particles in A */
  int ngroup = (nf / bDx) + (nf % bDx == 0 ? 0 : 1);

  /* Loop over groups */
  for (int igroup = 0; igroup < ngroup; igroup ++) {

    /* Assign the tx-th thread to the i-th particle, say f, in A */
    int i = igroup * bDx + tx;

    /* Load the position and charge of f */
    real4 xf;
    if (i < nf) { // Screening existing particle
      xf = x[ista + i];
    } else {
      xf.x = xf.y = xf.z = xf.w = ZERO; // set dummy
    }

    /* Initialise the field value of f */
    real phi = ZERO;
  
    /* Loop over neighbour-cells of A */
    for (int ineighbor = 0; ineighbor < nneigh; ineighbor ++) {
      
      /* Obtain the index of neighbour */
      int B = neighbors[offset + ineighbor];
      
      /* Loop over particles in B */
      for (int j = liststa[B]; j <= listend[B]; j ++) {
	
	/* Load the position and charge of the particle, say s */
	real4 xs = x[j];
	
	/* Accumulate the interaction between f and s */
	ACCUMULATE_INTERACTION(phi, xf, xs);
	
      } // j
      
    } // ineighbour
    
    /* Store the field value of f on device memory */
    if (i < nf) { // Screening existing particle
      p[ista + i] += xf.w * phi;
    }

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46E)
/**************************************************************************/
/* based on CUDA_VER46C */

/* Compute q/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)

/* Compute q*r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv

/* Compute q/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)

#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s, q)		\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (q);				\
    }							\
  }


#define bx blockIdx.x
#define tx threadIdx.x

#define bDx blockDim.x

__global__ void
//direct(int *ineigh, int *neighbors, int pitch_neighbors,
//       int *particlelist, int *liststa, int *listend,
//       real3 *x, real *p, real *q, int A0)
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors,
       const int *liststa, const int *listend,
       const real3 *x, real *p, const real *q, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
#endif

  /* Obtain the index of the current field-cell A */
  int A = A0 + bx;

  /* Preload the offset address for neighbors[] */
  int offset = pitch_neighbors * A;

  /* Preload the number of neighbour-cells (or source-cells) of A */
  int nneigh = ineigh[A];
  
  /* Preload the start and last addresses for A */
  int lsta = liststa[A];
  int lend = listend[A];

  /* Obtain the number of particles in A */
  //  int nf = listend[A] - liststa[A] + 1;
  int nf = lend - lsta + 1;
 
  /* Every bDx field-particles is handled in one group. Compute the
     number of such groups. */
  int ngroup = (nf / bDx) + (nf % bDx == 0 ? 0 : 1);

  /* Loop over groups of field-particles */
  for (int igroup = 0; igroup < ngroup; igroup ++) {

    /* Consider the i-th particle if it exists */
    int i = igroup * bDx + tx;

    if (i < nf) { // Screening
      
      //      /* Obtain the index of a field particle f */
      //      int findex = particlelist[liststa[A] + i];
      //      
      //      /* Load the position  of f */
      //      real3 xf = x[findex];

      /* Load the position  of f */
      real3 xf = x[lsta + i];
      
      /* Initialise the field value of f */
      real phi = ZERO;
  
      /* Loop over neighbour-cells of A */
      for (int ineighbor = 0; ineighbor < nneigh; ineighbor ++) {
	
	/* Obtain the index of neighbour-cell */
	//	int B = neighbors[pitch_neighbors * A + ineighbor];
	int B = neighbors[offset + ineighbor];
	
	/* Loop over particles in B */
	for (int j = liststa[B]; j <= listend[B]; j ++) {
	  
	  //	  /* Obtain the index of the particle s */
	  //	  int sindex = particlelist[j];
	  //	  
	  //	  /* Load the position of s */
	  //	  real3 xs = x[sindex];
	  //
	  //	  /* Load the charge of s */
	  //	  real qs = q[sindex];

	  /* Load the position of s */
	  real3 xs = x[j];

	  /* Load the charge of s */
	  real qs = q[j];
	  
	  /* Accumulate the interaction between f and s */
	  ACCUMULATE_INTERACTION(phi, xf, xs, qs);

	} // j
	
      } // ineighbour
      
      /* Store the field value of f on device memory */
      //      p[findex] += q[findex] * phi;
      p[lsta + i] += q[lsta + i] * phi;

    } // i < nf

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46D)
/**************************************************************************/
/* based on CUDA_VER46C */

/* Compute q/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)

/* Compute q*r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv

/* Compute q/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)

#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s, q)		\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (q);				\
    }							\
  }


#define bx blockIdx.x
#define tx threadIdx.x

#define bDx blockDim.x

__device__ void
direct_sub(const int *neighbors, const int pitch_neighbors, 
	   const int *particlelist, const int *liststa, const int *listend,
	   const real3 *x, real *p, const real *q,
	   const int A, const int nneigh, const int i)
{
  /* Obtain the index of a field particle f */
  int findex = particlelist[liststa[A] + i];
      
  /* Load the position  of f */
  real3 xf = x[findex];
      
  /* Initialise the field value of f */
  real phi = ZERO;
  
  /* Loop over neighbour-cells of A */
  for (int ineighbor = 0; ineighbor < nneigh; ineighbor ++) {
    
    /* Obtain the index of neighbour-cell */
    int B = neighbors[pitch_neighbors * A + ineighbor];
    
    /* Loop over particles in B */
    for (int j = liststa[B]; j <= listend[B]; j ++) {
      
      /* Obtain the index of the particle s */
      int sindex = particlelist[j];
      
      /* Load the position of s */
      real3 xs = x[sindex];
      
      /* Load the charge of s */
      real qs = q[sindex];
      
      /* Accumulate the interaction between f and s */
      ACCUMULATE_INTERACTION(phi, xf, xs, qs);
      
    } // j
    
  } // ineighbour
  
  p[findex] += q[findex] * phi;
}


__global__ void
direct(const int *ineigh, const int *neighbors, const int pitch_neighbors, 
       const int *particlelist, const int *liststa, const int *listend,
       const real3 *x, real *p, const real *q, const int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(gridDim.x > 0);
#endif

  /* Obtain the index of the current field-cell A */
  int A = A0 + bx;

  /* Obtain the number of neighbour-cells (or source-cells) of A */
  int nneigh = ineigh[A];
  
  /* Obtain the number of particles in A */
  int nf = listend[A] - liststa[A] + 1;
  
  /* Every bDx field-particles is handled in one group. Compute the
     number of such groups. */
  int ngroup = (nf / bDx) + (nf % bDx == 0 ? 0 : 1);

#if(1)

  /* Loop over groups of field-particles except for the last group */
  for (int igroup = 0; igroup < ngroup - 1; igroup ++) {

    /* Consider the i-th particle, which always exists */
    int i = igroup * bDx + tx;
    
    direct_sub(neighbors, pitch_neighbors,
	       particlelist, liststa, listend,
	       x, p, q,
	       A, nneigh, i);
  }
  {
    /* Consider the i-th particle in the last group if it exists */
    int i = (ngroup - 1) * bDx + tx;
    
    if (i < nf) { // Screening
      direct_sub(neighbors, pitch_neighbors,
		 particlelist, liststa, listend,
		 x, p, q,
		 A, nneigh, i);
    }
  }

#else

  /* Loop over groups of field-particles */
  for (int igroup = 0; igroup < ngroup; igroup ++) {

    /* Consider the i-th particle if it exists */
    int i = igroup * bDx + tx;

    if (i < nf) { // Screening
      direct_sub(neighbors, pitch_neighbors,
		 particlelist, liststa, listend,
		 x, p, q,
		 A, nneigh, i);
    }

  } // igroup

#endif

}
/**************************************************************************/
#elif defined(CUDA_VER46C)
/**************************************************************************/
/* based on CUDA_VER46B */

/* Compute q/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = RSQRT(rr)

/* Compute q*r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv

/* Compute q/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)

#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s, q)		\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (q);				\
    }							\
  }


#define bx blockIdx.x
#define tx threadIdx.x

#define bDx blockDim.x

__global__ void
direct(int *ineigh, int *neighbors, int pitch_neighbors, 
       int *particlelist, int *liststa, int *listend,
       real3 *x, real *p, real *q, int A0)
{
#ifdef __DEVICE_EMULATION___
  //  assert(bDx >= 27 && gridDim.x > 0);
  assert(gridDim.x > 0);
#endif

  /* Obtain the index of the current field-cell A */
  int A = A0 + bx;

  /* Obtain the number of neighbour-cells (or source-cells) of A */
  int nneigh = ineigh[A];
  
  /* Obtain the number of particles in A */
  int nf = listend[A] - liststa[A] + 1;
  
  //  /* Load the list of neighbour-cells into shared-memory.  In order to
  //     read the list fully, the number of threads per thread-block must
  //     be equal to 27 or more */
  //  __shared__ int s_neighbors[27];
  //
  //  if (tx < nneigh) {
  //    int B = neighbors[pitch_neighbors * A + tx];
  //    s_neighbors[tx] = B;
  //  }
  //  __syncthreads();

  /* Every bDx field-particles is handled in one group. Compute the
     number of such groups. */
  int ngroup = (nf / bDx) + (nf % bDx == 0 ? 0 : 1);

  /* Loop over groups of field-particles */
  for (int igroup = 0; igroup < ngroup; igroup ++) {

    /* Consider the i-th particle if it exists */
    int i = igroup * bDx + tx;

    if (i < nf) { // Screening
      
      /* Obtain the index of a field particle f */
      int findex = particlelist[liststa[A] + i];
      
      /* Load the position  of f */
      real3 xf = x[findex];
      
      /* Initialise the field value of f */
      real phi = ZERO;
  
      /* Loop over neighbour-cells of A */
      for (int ineighbor = 0; ineighbor < nneigh; ineighbor ++) {
	
	/* Obtain the index of neighbour-cell */
	//	int B = s_neighbors[ineighbor];
	int B = neighbors[pitch_neighbors * A + ineighbor];
	
	/* Loop over particles in B */
	for (int j = liststa[B]; j <= listend[B]; j ++) {
	  
	  /* Obtain the index of the particle s */
	  int sindex = particlelist[j];
	  
	  /* Load the position of s */
	  real3 xs = x[sindex];

	  /* Load the charge of s */
	  real qs = q[sindex];
	  
	  /* Accumulate the interaction between f and s */
	  ACCUMULATE_INTERACTION(phi, xf, xs, qs);

	} // j
	
      } // ineighbour
      
      /* Store the field value of f on device memory */
      p[findex] += q[findex] * phi;

    } // i < nf

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46B)
/**************************************************************************/
/* based on CUDA_VER46A */
//ptxas info    : Compiling entry function '_Z6directPiS_iS_S_S_S_S_S_P7double4S1_i' for 'sm_20'
//ptxas info    : Used 53 registers, 108+0 bytes smem, 124 bytes cmem[0], 20 bytes cmem[16]
//ptxas info    : Compiling entry function '_Z6directPiS_iS_S_S_S_S_S_P6float4S1_i' for 'sm_20'
//ptxas info    : Used 49 registers, 108+0 bytes smem, 124 bytes cmem[0], 16 bytes cmem[16]

/* Compute 1/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = ONE / SQRT(rr)
/* Compute r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = RSQRT(rr); real kern = dx * rinv * rinv * rinv
/* Compute 1/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)
#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }


#define bx blockIdx.x
#define tx threadIdx.x
#define bDx blockDim.x // number of threads per thread-block

__global__ void
direct(int *ineigh, int *neighbors, int pitch_neighbors, 
       int *fieldlist, int *fieldsta, int *fieldend,
       int *sourcelist, int *sourcesta, int *sourceend,
       real4 *field, real4 *source, int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(bDx >= 27 && gridDim.x > 0);
#endif

  /* Obtain the index of the current field-cell A */
  int A = A0 + bx;

  /* Obtain the number of neighbour-cells (or source-cells) of A */
  int nneigh = ineigh[A];
  
  /* Obtain the number of particles in A */
  int nf = fieldend[A] - fieldsta[A] + 1;
  
  /* Load the list of neighbour-cells into shared-memory.  In order to
     read the list fully, the number of threads per thread-block must
     be equal to 27 or more */
  __shared__ int s_neighbors[27];

  if (tx < nneigh) {
    int B = neighbors[pitch_neighbors * A + tx];
    s_neighbors[tx] = B;
  }
  __syncthreads();

  /* Every bDx field-particles is handled in one group. Compute the
     number of such groups. */
  int ngroup = (nf / bDx) + (nf % bDx == 0 ? 0 : 1);

  /* Loop over groups of field-particles */
  for (int igroup = 0; igroup < ngroup; igroup ++) {

    /* Consider the i-th particle if it exists */
    int i = igroup * bDx + tx;

    if (i < nf) { // Screening
      
      /* Obtain the index of a field particle f */
      int findex = fieldlist[fieldsta[A] + i];
      
      /* Load the position (and field value) of f */
      real4 f = field[findex];
      
      /* Initialise the field value of f */
      real phi = ZERO;
  
      /* Loop over neighbour-cells of A */
      for (int ineighbor = 0; ineighbor < nneigh; ineighbor ++) {
	
	/* Obtain the index of neighbour-cell */
	int B = s_neighbors[ineighbor];
	
	/* Loop over particles in B */
	for (int j = sourcesta[B]; j <= sourceend[B]; j ++) {
	  
	  /* Obtain the index of the particle s */
	  int sindex = sourcelist[j];
	  
	  /* Load the position and charge of s */
	  real4 s = source[sindex];
	  
	  /* Accumulate the interaction between f and s */
	  ACCUMULATE_INTERACTION(phi, f, s);

	} // j
	
      } // ineighbour
      
      field[findex].w += source[findex].w * phi;

    } // i < nf

  } // igroup

}
/**************************************************************************/
#elif defined(CUDA_VER46A)
/**************************************************************************/
/* based on CUDA_VER22C and VER45 */

#error This does not work.

/* Compute q/r */
#ifdef LAPLACIAN
#define COMP_KERNEL() real kern = ONE / SQRT(rr)

/* Compute q*r.x/r^3 */
#elif LAPLACIANFORCE
#define COMP_KERNEL() real rinv = DIVIDE(ONE, SQRT(rr)); real kern = dx * rinv * rinv * rinv

/* Compute q/r^4 */
#elif ONEOVERR4
#define COMP_KERNEL() real kern = ONE / (rr * rr)

#else
#error kernel is not defined.
#endif

#define ACCUMULATE_INTERACTION(p, f, s)			\
  {							\
    real dx = (s).x - (f).x;				\
    real rr = dx * dx;					\
    real dy = (s).y - (f).y;				\
    rr += dy * dy;					\
    real dz = (s).z - (f).z;				\
    rr += dz * dz;					\
    if (rr != ZERO) {					\
      COMP_KERNEL();					\
      (p) += kern * (s).w;				\
    }							\
  }


#define bx blockIdx.x
#define tx threadIdx.x

#define bDx blockDim.x

#define NULL_PARTICLE (- 1)

__global__ void
direct(int *ineigh, int *neighbors, int pitch_neighbors, 
       int *Nf, int *fieldlist, int pitch_fieldlist,
       int *Ns, int *sourcelist, int pitch_sourcelist, 
       real4 *field, real4 *source, int A0)
{
#ifdef __DEVICE_EMULATION___
  assert(bDx > 27 && gridDim.x > 0);
#endif

  /* index of the current field-cell A */
  int A = A0 + bx;

  /* number of neighbour-cells (or source-cells) of A */
  int nneigh = ineigh[A];

  /* number of field-particles in A */
  int nf = Nf[A];
  
  /* Load the list of neighbour-cells and the number of
     source-particles in each neighbour-cell into shared-memory.  In
     order to read the list fully, the number of threads per
     thread-block must be equal to 27 or more */
  __shared__ int s_neighbors[27], s_ns[27];
  
  if (tx < nneigh) {
    int B = neighbors[pitch_neighbors * A + tx];
    s_neighbors[tx] = B;
    s_ns[tx] = Ns[B];
  }
  __syncthreads();

  /* Compute the number of groups of field particles. One group
     consists of bDx particles */
  int ngroup = (nf / bDx) + (nf % bDx == 0 ? 0 : 1);

  /* Loop over groups of field particles in A */
  for (int igroup = 0; igroup < ngroup; igroup ++) {

    /* Load index of field particle to which the current thread is
       assigned and load the coordinates and potential of the field
       particles */

    int findex;
    real4 f;

    int i = igroup * bDx + tx;

    if (i < nf) { // Assing a field particle in A
      
      findex = fieldlist[pitch_fieldlist * A + i];
      f = field[findex];
      
    } else { // Assign none of field particles in A
      
      findex = NULL_PARTICLE;

    }
  
    /* Initialise the field value of f */
    real phi = ZERO;
  
    /* Loop over neighbour-cells of A */
    for (int ineighbor = 0; ineighbor < nneigh; ineighbor ++) {

      /* Obtain the index of neighbour-cell */
      int B = s_neighbors[ineighbor];

      /* Obtain the number of source-particles in B */
      int ns = s_ns[ineighbor];

      /* Loop over source-particles in B */
      for (int j = 0; j < ns; j ++) {

	/* Obtain the index of source particle */
	int sindex = sourcelist[pitch_sourcelist * B + j];

	/* Load the source particle (LDU instruction is used?) */
	real4 s = source[sindex];
      
	if (findex != NULL_PARTICLE) {

	  /* Accumulate the interaction between f and s */
	  ACCUMULATE_INTERACTION(phi, f, s);

	}

      } // j

    } // ineighbour

    if (findex != NULL_PARTICLE) {
      field[findex].w += source[findex].w * phi;
    }

  } // igroup

}
/**************************************************************************/
#else
/**************************************************************************/
#error No minor version was specified.
/**************************************************************************/
#endif
/**************************************************************************/
/**************************************************************************/
#else
/**************************************************************************/
#error Use VER46 or later. Old versions were removed.
/**************************************************************************/
#endif
/**************************************************************************/
#endif /* DIRECT_CU */
