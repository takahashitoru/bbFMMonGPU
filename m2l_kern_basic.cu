#ifndef M2L_KERN_BASIC_CU
#define M2L_KERN_BASIC_CU

/**************************************************************************/
#if defined(CUDA_VER45)
/**************************************************************************/
/**************************************************************************/
#if defined(CUDA_VER45B) || defined(CUDA_VER45C) || defined(CUDA_VER45D) || defined(CUDA_VER45E) || defined(CUDA_VER45F) || defined(CUDA_VER45G) || defined(CUDA_VER45H) || defined(CUDA_VER45I)
/**************************************************************************/
/* Based on CUDA_VER45A */
#include "real.h"

#if !defined(BA_UNROLLING_r256)
#define BA_UNROLLING_r256 (4) // 1, 2, or 4; No significant difference for C2050+SDK3.2
#endif
#if !defined(BA_UNROLLING_r32)
#define BA_UNROLLING_r32  (32) // 1 or 32; No significant difference for C2050+SDK3.2
#endif

#define bx blockIdx.x
#define tx threadIdx.x

#define cal256(jp0, jp1)			\
  K1 = *Kptr; Kptr += 256;			\
  *Lf += K0 * sM[jp0];				\
  K0 = *Kptr; Kptr += 256;			\
  *Lf += K1 * sM[jp1]

__device__ void BA_comp256(real *K, int kindx[189], real *Lf, real sM[256], int iinter)
{
  /* Set a ponter to K_{tx,0}(F,S), i.e. the element of tx-th row
     and 0th column. Note that K is stored in column major */
  real *Kptr = K + (kindx[iinter] << 16) + tx; // log2(r^2)=16
#if (BA_UNROLLING_r256 == 4) // Prefetch K-data and unroll the column loop four times
  //    real K0 = *Kptr; Kptr += 256;      /* fetch j=0 */
  //    for (int j = 0; j < 252; j += 4) { /* loop  j=0,4,...,248 */
  //      real K1 = *Kptr; Kptr += 256;    /* fetch j=1,5,...,249 */
  //      *Lf += K0 * sM[j];               /* comp  j=0,4,...,248 */
  //      K0 = *Kptr; Kptr += 256;         /* fetch j=2,6,...,250 */
  //      *Lf += K1 * sM[j + 1];           /* comp  j=1,5,...,249 */
  //      K1 = *Kptr; Kptr += 256;         /* fetch j=3,7,...,251 */
  //      *Lf += K0 * sM[j + 2];           /* comp  j=2,6,...,250 */
  //      K0 = *Kptr; Kptr += 256;         /* fetch j=4,8,...,252 */
  //      *Lf += K1 * sM[j + 3];           /* comp  j=3,7,...,251 */
  //    }
  //    real K1 = *Kptr; Kptr += 256;      /* fetch j=253 */
  //    *Lf += K0 * sM[252];               /* comp  j=252 */
  //    K0 = *Kptr; Kptr += 256;           /* fetch j=254 */
  //    *Lf += K1 * sM[253];               /* comp  j=253 */
  //    K1 = *Kptr;                        /* fetch j=255 */
  //    *Lf += K0 * sM[254];               /* comp  j=254 */
  //    *Lf += K1 * sM[255];               /* comp  j=255 */
  real K0, K1;
  K0 = *Kptr; Kptr += 256;
  for (int j = 0; j < 252; j += 4) {
    cal256(j    , j + 1);
    cal256(j + 2, j + 3);
  }
  cal256(252, 253);
  K1 = *Kptr;
  *Lf += K0 * sM[254];
  *Lf += K1 * sM[255];
#elif (BA_UNROLLING_r256 == 2) // Prefetch K-data and unroll the column loop twice
  real K0, K1;
  K0 = *Kptr; Kptr += 256;
  for (int j = 0; j < 254; j += 2) {
    cal256(j    , j + 1);
  }
  K1 = *Kptr;
  *Lf += K0 * sM[254];
  *Lf += K1 * sM[255];
#elif (BA_UNROLLING_r256 == 1) // Unoptimized version
  for (int j = 0; j < 256; j ++) {
    *Lf += *Kptr * sM[j]; Kptr += 256;
  }
#else
#error Invalid parameter.
#endif
}


__global__ void m2l_kern_basic_r256(real *L, real *K, real *M,
				    int Fsta, int pitch_interaction,
				    int *interaction, int *iinter, int *Kindex)
{
  /* Compute the index of the current observation cell F, to which the
     current thread-block bx is assigned */
  int F = Fsta + bx;

  /* Load the number of source cells for F */
  int ninter = iinter[F];

  /* Load the interaction-list of F and the list of base addresses to
     K-matrices with all the threads */
  __shared__ int ilist[189], kindx[189] ;
  if (tx < ninter) {
    int itmp = pitch_interaction * F + tx;
    ilist[tx] = interaction[itmp];
    kindx[tx] = Kindex[itmp];
  }

  /* Ensure that ilist[] and kindx[] were loaded */
  __syncthreads();

  /* Initialize the tx-th element of vector L(F), to which the current
     thread tx is assigned */
  real Lf = ZERO;

  /* Loop over source cells S */
  for (int iinter = 0; iinter < ninter; iinter ++) {
    
    /* Load the vector M(S) with all the threads */
    __shared__ real sM[256];
    sM[tx] = *(M + (ilist[iinter] << 8) + tx); // log2(r)=8
    
    /* Ensure that sM[] was loaded */
    __syncthreads();
    
    /* Compute L_i(F)+=K_{i,j}(F,S)*M_j(S), where i=tx */
    BA_comp256(K, kindx, &Lf, sM, iinter);

    /* Ensure that sM[] is no longer used */
    __syncthreads();

  }

  /* Store Lf in device memory (not increment, but substitution) */
  *(L + (bx << 8) + tx) = Lf; // log2(r)=8

}


#define cal32(jp0, jp1)				\
  K1 = *Kptr; Kptr += 32;			\
  *Lf += K0 * sM[jp0];				\
  K0 = *Kptr; Kptr += 32;			\
  *Lf += K1 * sM[jp1]

__device__ void BA_comp32(real *K, int kindx[189], real *Lf, real sM[32], int iinter)
{
  /* Set a ponter to K_{tx,0}(F,S), i.e. the element of tx-th row and
     0th column. Note that K is stored in column major */
  real *Kptr = K + (kindx[iinter] << 10) + tx; // log2(r^2)=10
#if (BA_UNROLLING_r32 == 32) // Prefetch K-data and unroll the column loop 32 times
  real K0, K1;
  K0 = *Kptr; Kptr += 32;
  cal32( 0,  1);
  cal32( 2,  3);
  cal32( 4,  5);
  cal32( 6,  7);
  cal32( 8,  9);
  cal32(10, 11);
  cal32(12, 13);
  cal32(14, 15);
  cal32(16, 17);
  cal32(18, 19);
  cal32(20, 21);
  cal32(22, 23);
  cal32(24, 25);
  cal32(26, 27);
  cal32(28, 29);
  K1 = *Kptr;
  *Lf += K0 * sM[30];
  *Lf += K1 * sM[31];
#elif (BA_UNROLLING_r32 == 1) // Unoptimized version
  for (int j = 0; j < 32; j ++) {
    *Lf += *Kptr * sM[j]; Kptr += 32;
    }
#else
#error Invalid parameter.
#endif
}


__global__ void m2l_kern_basic_r32(real *L, real *K, real *M,
				   int Fsta, int pitch_interaction,
				   int *interaction, int *iinter, int *Kindex)
{
  /* Compute the index of the current observation cell F, to which the
     current thread-block bx is assigned */
  int F = Fsta + bx;
  
  /* Load the number of source cells for F */
  int ninter = iinter[F];

  /* Load the interaction-list of F and the list of base addresses to
     K-matrices with all the threads */
  __shared__ int ilist[189], kindx[189] ;
  if (tx < ninter) {
    int itmp = pitch_interaction * F + tx;
    ilist[tx] = interaction[itmp];
    kindx[tx] = Kindex[itmp];
  }
  if (tx + 32 < ninter) {
    int itmp = pitch_interaction * F + tx + 32;
    ilist[tx + 32] = interaction[itmp];
    kindx[tx + 32] = Kindex[itmp];
  }
  if (tx + 64 < ninter) {
    int itmp = pitch_interaction * F + tx + 64;
    ilist[tx + 64] = interaction[itmp];
    kindx[tx + 64] = Kindex[itmp];
  }
  if (tx + 96 < ninter) {
    int itmp = pitch_interaction * F + tx + 96;
    ilist[tx + 96] = interaction[itmp];
    kindx[tx + 96] = Kindex[itmp];
  }
  if (tx + 128 < ninter) {
    int itmp = pitch_interaction * F + tx + 128;
    ilist[tx + 128] = interaction[itmp];
    kindx[tx + 128] = Kindex[itmp];
  }
  if (tx + 160 < ninter) {
    int itmp = pitch_interaction * F + tx + 160;
    ilist[tx + 160] = interaction[itmp];
    kindx[tx + 160] = Kindex[itmp];
  }
  
  /* Ensure that ilist[] and kindx[] were loaded */
  __syncthreads();
  
  /* Initialize the tx-th element of vector L(F), to which the current
     thread tx is assigned */
  real Lf = ZERO;
  
  /* Loop over source cells S */
  for (int iinter = 0; iinter < ninter; iinter ++) {
    
    /* Load the vector M(S) with all the threads */
    __shared__ real sM[32];
    sM[tx] = *(M + (ilist[iinter] << 5) + tx); // log2(r)=5
    
    /* Ensure that sM[] was loaded */
    __syncthreads();
    
    /* Compute L_i(F)+=K_{i,j}(F,S)*M_j(S), where i=tx */
    BA_comp32(K, kindx, &Lf, sM, iinter);

    /* Ensure that sM[] is no longer used */
    __syncthreads();

  }

  /* Store Lf in device memory (not increment, but substitution) */
  *(L + (bx << 5) + tx) = Lf; // log2(r)=5

}
/**************************************************************************/
#elif defined(CUDA_VER45A)
/**************************************************************************/
#include "real.h"

#define bx blockIdx.x
#define tx threadIdx.x

//110228__global__ void m2l_kern_basic_r256(float *L, float *K, float *M,
//110228				    int Fsta, int pitch_interaction,
//110228				    int *interaction, int *iinter, int *Kindex)
__global__ void m2l_kern_basic_r256(real *L, real *K, real *M,
				    int Fsta, int pitch_interaction,
				    int *interaction, int *iinter, int *Kindex)
{
  /* Compute the index of the current observation cell F, to which the
     current thread-block bx is assigned */
  int F = Fsta + bx;

  /* Load the number of source cells for F */
  int ninter = iinter[F];

  /* Load the interaction-list of F and the list of base addresses to
     K-matrices with all the threads */
  __shared__ int ilist[189], kindx[189] ;
  if (tx < ninter) {
    int itmp = pitch_interaction * F + tx;
    ilist[tx] = interaction[itmp];
    kindx[tx] = Kindex[itmp];
  }

  /* Ensure that ilist[] and kindx[] were loaded */
  __syncthreads();

  /* Initialize the tx-th element of vector L(F), to which the current
     thread tx is assigned */
  //110228  float Lf = 0.0f;
  real Lf = ZERO;

  /* Loop over source cells S */
  for (int iinter = 0; iinter < ninter; iinter ++) {
    
    /* Load the vector M(S) with all the threads */
    //110228    __shared__ float sM[256];
    __shared__ real sM[256];
    sM[tx] = *(M + (ilist[iinter] << 8) + tx); // log2(r)=8
    
    /* Ensure that sM[] was loaded */
    __syncthreads();
    
    /* Set a ponter to K_{tx,0}(F,S), i.e. the element of tx-th row
       and 0th column. Note that K is stored in column major */
    //110228    float *Kptr = K + (kindx[iinter] << 16) + tx; // log2(r^2)=16
    real *Kptr = K + (kindx[iinter] << 16) + tx; // log2(r^2)=16
    
    /* Compute L_i(F)+=K_{i,j}(F,S)*M_j(S), where i=tx */
#if(1)
    /* Optimized version (prefetch K-data and unroll the column loop
       four times) */
    //110228    float K0 = *Kptr; Kptr += 256;     /* fetch j=0 */
    real K0 = *Kptr; Kptr += 256;     /* fetch j=0 */
    for (int j = 0; j < 252; j += 4) { /* loop  j=0,4,...,248 */
      //110228      float K1 = *Kptr; Kptr += 256;   /* fetch j=1,5,...,249 */
      real K1 = *Kptr; Kptr += 256;    /* fetch j=1,5,...,249 */
      Lf += K0 * sM[j];                /* comp  j=0,4,...,248 */
      K0 = *Kptr; Kptr += 256;         /* fetch j=2,6,...,250 */
      Lf += K1 * sM[j + 1];            /* comp  j=1,5,...,249 */
      K1 = *Kptr; Kptr += 256;         /* fetch j=3,7,...,251 */
      Lf += K0 * sM[j + 2];            /* comp  j=2,6,...,250 */
      K0 = *Kptr; Kptr += 256;         /* fetch j=4,8,...,252 */
      Lf += K1 * sM[j + 3];            /* comp  j=3,7,...,251 */
    }
    //110228    float K1 = *Kptr; Kptr += 256;     /* fetch j=253 */
    real K1 = *Kptr; Kptr += 256;      /* fetch j=253 */
    Lf += K0 * sM[252];                /* comp  j=252 */
    K0 = *Kptr; Kptr += 256;           /* fetch j=254 */
    Lf += K1 * sM[253];                /* comp  j=253 */
    K1 = *Kptr;                        /* fetch j=255 */
    Lf += K0 * sM[254];                /* comp  j=254 */
    Lf += K1 * sM[255];                /* comp  j=255 */
#else
    /* Unoptimized version */
    for (int j = 0; j < 256; j ++) {
      Lf += *Kptr * sM[j]; Kptr += 256;
    }
#endif

    /* Ensure that sM[] is no longer used */
    __syncthreads();

  }

  /* Store Lf in device memory (not increment, but substitution) */
  *(L + (bx << 8) + tx) = Lf; // log2(r)=8

}


//110228__global__ void m2l_kern_basic_r32(float *L, float *K, float *M,
//110228				   int Fsta, int pitch_interaction,
//110228				   int *interaction, int *iinter, int *Kindex)
__global__ void m2l_kern_basic_r32(real *L, real *K, real *M,
				   int Fsta, int pitch_interaction,
				   int *interaction, int *iinter, int *Kindex)
{
  /* Compute the index of the current observation cell F, to which the
     current thread-block bx is assigned */
  int F = Fsta + bx;
  
  /* Load the number of source cells for F */
  int ninter = iinter[F];

  /* Load the interaction-list of F and the list of base addresses to
     K-matrices with all the threads */
  __shared__ int ilist[189], kindx[189] ;
  if (tx < ninter) {
    int itmp = pitch_interaction * F + tx;
    ilist[tx] = interaction[itmp];
    kindx[tx] = Kindex[itmp];
  }
  if (tx + 32 < ninter) {
    int itmp = pitch_interaction * F + tx + 32;
    ilist[tx + 32] = interaction[itmp];
    kindx[tx + 32] = Kindex[itmp];
  }
  if (tx + 64 < ninter) {
    int itmp = pitch_interaction * F + tx + 64;
    ilist[tx + 64] = interaction[itmp];
    kindx[tx + 64] = Kindex[itmp];
  }
  if (tx + 96 < ninter) {
    int itmp = pitch_interaction * F + tx + 96;
    ilist[tx + 96] = interaction[itmp];
    kindx[tx + 96] = Kindex[itmp];
  }
  if (tx + 128 < ninter) {
    int itmp = pitch_interaction * F + tx + 128;
    ilist[tx + 128] = interaction[itmp];
    kindx[tx + 128] = Kindex[itmp];
  }
  if (tx + 160 < ninter) {
    int itmp = pitch_interaction * F + tx + 160;
    ilist[tx + 160] = interaction[itmp];
    kindx[tx + 160] = Kindex[itmp];
  }
  
  /* Ensure that ilist[] and kindx[] were loaded */
  __syncthreads();
  
  /* Initialize the tx-th element of vector L(F), to which the current
     thread tx is assigned */
  //110228  float Lf = 0.0f;
  real Lf = ZERO;
  
  /* Loop over source cells S */
  for (int iinter = 0; iinter < ninter; iinter++) {
    
    /* Load the vector M(S) with all the threads */
    //110228    __shared__ float sM[32];
    __shared__ real sM[32];
    sM[tx] = *(M + (ilist[iinter] << 5) + tx); // log2(r)=5
    
    /* Ensure that sM[] was loaded */
    __syncthreads();
    
    /* Set a ponter to K_{tx,0}(F,S), i.e. the element of tx-th row
       and 0th column. Note that K is stored in column major */
    //110228    float *Kptr = K + (kindx[iinter] << 10) + tx; // log2(r^2)=10
    real *Kptr = K + (kindx[iinter] << 10) + tx; // log2(r^2)=10
    
    /* Compute L_i(F)+=K_{i,j}(F,S)*M_j(S), where i=tx */
#if(1)
    /* Optimized version (prefetch K-data and unroll the column loop
       fully) */
#define cal32(jp0, jp1)				\
    K1 = *Kptr; Kptr += 32;			\
    Lf += K0 * sM[jp0];				\
    K0 = *Kptr; Kptr += 32;			\
    Lf += K1 * sM[jp1]

    //110228    float K0, K1;
    real K0, K1;
    K0 = *Kptr; Kptr += 32;
    cal32( 0,  1);
    cal32( 2,  3);
    cal32( 4,  5);
    cal32( 6,  7);
    cal32( 8,  9);
    cal32(10, 11);
    cal32(12, 13);
    cal32(14, 15);
    cal32(16, 17);
    cal32(18, 19);
    cal32(20, 21);
    cal32(22, 23);
    cal32(24, 25);
    cal32(26, 27);
    cal32(28, 29);
    K1 = *Kptr;
    Lf += K0 * sM[30];
    Lf += K1 * sM[31];
#else
    /* Unoptimized version */
    for (int j = 0; j < 32; j ++) {
      Lf += *Kptr * sM[j]; Kptr += 32;
    }
#endif

    /* Ensure that sM[] is no longer used */
    __syncthreads();

  }

  /* Store Lf in device memory (not increment, but substitution) */
  *(L + (bx << 5) + tx) = Lf; // log2(r)=5

}
/**************************************************************************/
#else
/**************************************************************************/
#error Any minor version was not specified.
/**************************************************************************/
#endif
/**************************************************************************/
/**************************************************************************/
#endif
/**************************************************************************/
#endif /* M2L_KERN_BASIC_CU */
