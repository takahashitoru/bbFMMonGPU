#ifndef M2L_KERN_CLUSTER_BLOCKING_CU
#define M2L_KERN_CLUSTER_BLOCKING_CU

/**************************************************************************/
#if defined(CUDA_VER45)
/**************************************************************************/
/**************************************************************************/
#if defined(CUDA_VER45B) || defined(CUDA_VER45C) || defined(CUDA_VER45D) || defined(CUDA_VER45E) || defined(CUDA_VER45F) || defined(CUDA_VER45G) || defined(CUDA_VER45H) || defined(CUDA_VER45I)
/**************************************************************************/
#include "real.h"

#define bx blockIdx.x  // assgined to the bx-th chunk of size B (0<=bx<8^{B-2})
#define ty threadIdx.y // assigned to the ty-th field-cluster (0<=ty<B^3)
#define tx threadIdx.x // assigned to the tx-th row (0<=tx<r/P)

#ifndef NULLCELL
#define NULLCELL - 1
#endif

#ifndef NULL_CELL
#define NULL_CELL NULLCELL
#endif

#ifndef NULL_KINDEX
#define NULL_KINDEX - 1
#endif

#define F0 (0)
#define F1 (1)
#define F2 (2)
#define F3 (3)
#define F4 (4)
#define F5 (5)
#define F6 (6)
#define F7 (7)
#define S0 (0)
#define S1 (1)
#define S2 (2)
#define S3 (3)
#define S4 (4)
#define S5 (5)
#define S6 (6)
#define S7 (7)

#define CL_load_Kpq256(p, q, kindex)					\
  {									\
    /* Set a pointer to Kpq (K is stored in double column-major) */	\
    int id = (ty << 5) + tx;  /* log2(r/P)=5          */		\
    int itmp = (q << 3) + p;  /* log2(P)=3            */		\
    itmp = (itmp << 10) + id; /* log2((r/P)*(r/Q))=10 */		\
    itmp += (kindex << 16);   /* log2(r^2)=16         */		\
    /*110228    float *Kptr = (float *)K + itmp;*/			\
    real *Kptr = (real *)K + itmp;					\
    /* Set a pointer to Kpq (column-major) */				\
    /*110228    float *Kpqptr = (float *)Kpq + id;*/			\
    real *Kpqptr = (real *)Kpq + id;					\
    /* Load the 1st 256 elements to Kpq[0:31][ 0: 7] */			\
    *Kpqptr = *Kptr; Kpqptr += 256; Kptr += 256;			\
    /* Load the 2nd 256 elements to Kpq[0:31][ 8:15] */			\
    *Kpqptr = *Kptr; Kpqptr += 256; Kptr += 256;			\
    /* Load the 3rd 256 elements to Kpq[0:31][16:23] */			\
    *Kpqptr = *Kptr; Kpqptr += 256; Kptr += 256;			\
    /* Load the 4th 256 elements to Kpq[0:31][24:31] */			\
    *Kpqptr = *Kptr;							\
  }

/* K-index is determined from source-cluster index and
   interaction-kind. Namely, the index is given by
   Ktable[kindex0[source-cluster index 0:26]+kindex1[interaction-kind
   0:26]] */
#ifndef load_Kindex0_Kindex1
#define load_Kindex0_Kindex1(i)						\
  {									\
    if      (i ==  0) {Kindex0[i] =  57; Kindex1[i] =  57;}		\
    else if (i ==  1) {Kindex0[i] = 155; Kindex1[i] =  55;}		\
    else if (i ==  2) {Kindex0[i] = 253; Kindex1[i] =  43;}		\
    else if (i ==  3) {Kindex0[i] =  71; Kindex1[i] =  41;}		\
    else if (i ==  4) {Kindex0[i] = 169; Kindex1[i] = -41;}		\
    else if (i ==  5) {Kindex0[i] = 267; Kindex1[i] = -43;}		\
    else if (i ==  6) {Kindex0[i] =  85; Kindex1[i] = -55;}		\
    else if (i ==  7) {Kindex0[i] = 183; Kindex1[i] = -57;}		\
    else if (i ==  8) {Kindex0[i] = 281; Kindex1[i] =  56;}		\
    else if (i ==  9) {Kindex0[i] =  59; Kindex1[i] =  50;}		\
    else if (i == 10) {Kindex0[i] = 157; Kindex1[i] =   8;}		\
    else if (i == 11) {Kindex0[i] = 255; Kindex1[i] =  48;}		\
    else if (i == 12) {Kindex0[i] =  73; Kindex1[i] =   6;}		\
    else if (i == 13) {Kindex0[i] = 171; Kindex1[i] =  42;}		\
    else if (i == 14) {Kindex0[i] = 269; Kindex1[i] =  -6;}		\
    else if (i == 15) {Kindex0[i] =  87; Kindex1[i] =  -8;}		\
    else if (i == 16) {Kindex0[i] = 185; Kindex1[i] = -42;}		\
    else if (i == 17) {Kindex0[i] = 283; Kindex1[i] = -48;}		\
    else if (i == 18) {Kindex0[i] =  61; Kindex1[i] = -50;}		\
    else if (i == 19) {Kindex0[i] = 159; Kindex1[i] = -56;}		\
    else if (i == 20) {Kindex0[i] = 257; Kindex1[i] =  49;}		\
    else if (i == 21) {Kindex0[i] =  75; Kindex1[i] =   7;}		\
    else if (i == 22) {Kindex0[i] = 173; Kindex1[i] =   1;}		\
    else if (i == 23) {Kindex0[i] = 271; Kindex1[i] =  -1;}		\
    else if (i == 24) {Kindex0[i] =  89; Kindex1[i] =  -7;}		\
    else if (i == 25) {Kindex0[i] = 187; Kindex1[i] = -49;}		\
    else if (i == 26) {Kindex0[i] = 285; Kindex1[i] =   0;}		\
    else if (i <  32) {Kindex0[i] =   0; Kindex1[i] =   0;} /* dummy */	\
  }
#endif

#define LKM(f, s, j)				\
  Lp[f] += Kpq[j][tx] * Mq[ty][s][j]
#define CL_core256x1(f, s)						\
  LKM(f, s,  0); LKM(f, s,  1); LKM(f, s,  2); LKM(f, s,  3);		\
  LKM(f, s,  4); LKM(f, s,  5); LKM(f, s,  6); LKM(f, s,  7);		\
  LKM(f, s,  8); LKM(f, s,  9); LKM(f, s, 10); LKM(f, s, 11);		\
  LKM(f, s, 12); LKM(f, s, 13); LKM(f, s, 14); LKM(f, s, 15);		\
  LKM(f, s, 16); LKM(f, s, 17); LKM(f, s, 18); LKM(f, s, 19);		\
  LKM(f, s, 20); LKM(f, s, 21); LKM(f, s, 22); LKM(f, s, 23);		\
  LKM(f, s, 24); LKM(f, s, 25); LKM(f, s, 26); LKM(f, s, 27);		\
  LKM(f, s, 28); LKM(f, s, 29); LKM(f, s, 30); LKM(f, s, 31)
#define CL_comp256x1(kind, f, s)			\
  /* Skip if s is a near neighbor of f */		\
  if (Kindex[kind] != NULL_KINDEX) {			\
    /* Load Kpq */					\
    CL_load_Kpq256(p, q, Kindex[kind]);		\
    /* Ensure that Kpq was loaded */			\
    __syncthreads();					\
    /* Compute Lp(f)=Kpq(f,s)*Mq(s) for a pair (f,s) */	\
    CL_core256x1(f, s);					\
    /* Ensure that Kpq is no longer referenced */	\
    __syncthreads();					\
  }

#define LKM2(fa, fb, sa, sb, j)			\
  Lp[fa] += Kpq[j][tx] * Mq[ty][sa][j];		\
  Lp[fb] += Kpq[j][tx] * Mq[ty][sb][j]
#define CL_core256x2(fa, fb, sa, sb)			\
  LKM2(fa, fb, sa, sb,  0); LKM2(fa, fb, sa, sb,  1);	\
  LKM2(fa, fb, sa, sb,  2); LKM2(fa, fb, sa, sb,  3);	\
  LKM2(fa, fb, sa, sb,  4); LKM2(fa, fb, sa, sb,  5);	\
  LKM2(fa, fb, sa, sb,  6); LKM2(fa, fb, sa, sb,  7);	\
  LKM2(fa, fb, sa, sb,  8); LKM2(fa, fb, sa, sb,  9);	\
  LKM2(fa, fb, sa, sb, 10); LKM2(fa, fb, sa, sb, 11);	\
  LKM2(fa, fb, sa, sb, 12); LKM2(fa, fb, sa, sb, 13);	\
  LKM2(fa, fb, sa, sb, 14); LKM2(fa, fb, sa, sb, 15);	\
  LKM2(fa, fb, sa, sb, 16); LKM2(fa, fb, sa, sb, 17);	\
  LKM2(fa, fb, sa, sb, 18); LKM2(fa, fb, sa, sb, 19);	\
  LKM2(fa, fb, sa, sb, 20); LKM2(fa, fb, sa, sb, 21);	\
  LKM2(fa, fb, sa, sb, 22); LKM2(fa, fb, sa, sb, 23);	\
  LKM2(fa, fb, sa, sb, 24); LKM2(fa, fb, sa, sb, 25);	\
  LKM2(fa, fb, sa, sb, 26); LKM2(fa, fb, sa, sb, 27);	\
  LKM2(fa, fb, sa, sb, 28); LKM2(fa, fb, sa, sb, 29);	\
  LKM2(fa, fb, sa, sb, 30); LKM2(fa, fb, sa, sb, 31)
#define CL_comp256x2(kind, fa, fb, sa, sb)			\
  /* Skip if s is a near neighbor of f */			\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /* Load Kpq */						\
    CL_load_Kpq256(p, q, Kindex[kind]);				\
    /* Ensure that Kpq was loaded */				\
    __syncthreads();						\
    /* Compute Lp(f)=Kpq(f,s)*Mq(s) for two pairs (f,s) */	\
    CL_core256x2(fa, fb, sa, sb);				\
    /* Ensure that Kpq is no longer referenced */		\
    __syncthreads();						\
  }

#define LKM4(fa, fb, fc, fd, sa, sb, sc, sd, j) \
  Lp[fa] += Kpq[j][tx] * Mq[ty][sa][j];		\
  Lp[fb] += Kpq[j][tx] * Mq[ty][sb][j];		\
  Lp[fc] += Kpq[j][tx] * Mq[ty][sc][j];		\
  Lp[fd] += Kpq[j][tx] * Mq[ty][sd][j]
#define CL_core256x4(fa, fb, fc, fd, sa, sb, sc, sd) \
  for (int j = 0; j < 32; j += 4) {		     \
    LKM4(fa, fb, fc, fd, sa, sb, sc, sd, j    );     \
    LKM4(fa, fb, fc, fd, sa, sb, sc, sd, j + 1);     \
    LKM4(fa, fb, fc, fd, sa, sb, sc, sd, j + 2);     \
    LKM4(fa, fb, fc, fd, sa, sb, sc, sd, j + 3);     \
  }
#define CL_comp256x4(kind, fa, fb, fc, fd, sa, sb, sc, sd)	\
  /* Skip if s is a near neighbor of f */			\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /* Load Kpq */						\
    CL_load_Kpq256(p, q, Kindex[kind]);				\
    /* Ensure that Kpq was loaded */				\
    __syncthreads();						\
    /* Compute Lp(f)=Kpq(f,s)*Mq(s) for four pairs (f,s) */	\
    CL_core256x4(fa, fb, fc, fd, sa, sb, sc, sd);		\
    /* Ensure that Kpq is no longer referenced */		\
    __syncthreads();						\
  }

#define LKM8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh, j) \
  Lp[0] += Kpq[j][tx] * Mq[ty][0][j];					\
  Lp[1] += Kpq[j][tx] * Mq[ty][1][j];					\
  Lp[2] += Kpq[j][tx] * Mq[ty][2][j];					\
  Lp[3] += Kpq[j][tx] * Mq[ty][3][j];					\
  Lp[4] += Kpq[j][tx] * Mq[ty][4][j];					\
  Lp[5] += Kpq[j][tx] * Mq[ty][5][j];					\
  Lp[6] += Kpq[j][tx] * Mq[ty][6][j];					\
  Lp[7] += Kpq[j][tx] * Mq[ty][7][j]
#define CL_core256x8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh) \
  for (int j = 0; j < 32; j += 4) {					\
    LKM8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh, j    ); \
    LKM8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh, j + 1); \
    LKM8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh, j + 2); \
    LKM8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh, j + 3); \
  }
#define CL_comp256x8(kind, fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh) \
  /* Skip if s is a near neighbor of f */				\
  if (Kindex[kind] != NULL_KINDEX) {					\
    /* Load Kpq */							\
    CL_load_Kpq256(p, q, Kindex[kind]);					\
    /* Ensure that Kpq was loaded */					\
    __syncthreads();							\
    /* Compute Lp(f)=Kpq(f,s)*Mq(s) for eight pairs (f,s) */		\
    CL_core256x8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh); \
    /* Ensure that Kpq is no longer referenced */			\
    __syncthreads();							\
  }

#define CL_load_M256(q, d)				\
  if (ilist[ty][d] != NULL_CELL) {			\
    int itmp = (q << 5) + tx;    /* log2(r/Q)=5 */	\
    itmp += (ilist[ty][d] << 8); /* log2(r)=8 */	\
    /*110228    float *Mptr = M + itmp;*/		\
    real *Mptr = M + itmp;				\
    Mq[ty][0][tx] = *Mptr;				\
    Mq[ty][1][tx] = *(Mptr +  256);			\
    Mq[ty][2][tx] = *(Mptr +  512);			\
    Mq[ty][3][tx] = *(Mptr +  768);			\
    Mq[ty][4][tx] = *(Mptr + 1024);			\
    Mq[ty][5][tx] = *(Mptr + 1280);			\
    Mq[ty][6][tx] = *(Mptr + 1536);			\
    Mq[ty][7][tx] = *(Mptr + 1792);			\
  } else {						\
    /*110228    Mq[ty][0][tx] = 0.0f;*/					\
    /*110228    Mq[ty][1][tx] = 0.0f;*/					\
    /*110228    Mq[ty][2][tx] = 0.0f;*/					\
    /*110228    Mq[ty][3][tx] = 0.0f;*/					\
    /*110228    Mq[ty][4][tx] = 0.0f;*/					\
    /*110228    Mq[ty][5][tx] = 0.0f;*/					\
    /*110228    Mq[ty][6][tx] = 0.0f;*/					\
    /*110228    Mq[ty][7][tx] = 0.0f;*/					\
    Mq[ty][0][tx] = ZERO;				\
    Mq[ty][1][tx] = ZERO;				\
    Mq[ty][2][tx] = ZERO;				\
    Mq[ty][3][tx] = ZERO;				\
    Mq[ty][4][tx] = ZERO;				\
    Mq[ty][5][tx] = ZERO;				\
    Mq[ty][6][tx] = ZERO;				\
    Mq[ty][7][tx] = ZERO;				\
  }

//110228__global__ void m2l_kern_cluster_blocking_r256b2p8q8(float *L, float *K, float *M, int *sourceclusters, int *Ktable, int FCHsta)
__global__ void m2l_kern_cluster_blocking_r256b2p8q8(real *L, real *K, real *M, int *sourceclusters, int *Ktable, int FCHsta)
{
  /* Load lists of source-clusters for B^3 field-clusters in the
     current chunk of size B(=2), to which the current (bx-th)
     thread-block is assigned */
  __shared__ int ilist[8][32]; // ilist[0:B^3-1][27 or more]; 1024 bytes
  {
    int id = (ty << 5) + tx; // log2(r/P)=5
    ilist[ty][tx] = sourceclusters[((FCHsta + bx) << 8) + id]; // log2(r)=8
  }

  /* Copy Ktable into shared memory with 256 threads */
  __shared__ int sKtable[343]; // 1372 bytes
  {
    int id = (ty << 5) + tx; // log2(r/P)=5
    sKtable[id] = Ktable[id];
    if (id < 87) {
      sKtable[id + 256] = Ktable[id + 256];
    }
  }

  /* Load Kindex0[] and Kindex1[] */
  __shared__ int Kindex0[32], Kindex1[32], Kindex[32]; // 27 or more; 384 bytes
  {
    int id = (ty << 5) + tx; // log2(r/P)=5
    load_Kindex0_Kindex1(id);
  }

  /* Ensure that ilist[], sKtable[], Kindex0[], and Kindex1[] were loaded */
  __syncthreads();

  /* Loop over row-tiles */
  for (int p = 0; p < 8; p ++) { // P=8
    
    /* Initialize the tx-th element of the tiled-vector Lp for the
       ty-th field cluster (FC) in the current chunk */
    //110228    float Lp[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // log2(B^3)=8
    real Lp[8] = {ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO}; // log2(B^3)=8

    /* Loop over column-tiles */
    for (int q = 0; q < 8; q ++) { // Q=8

      /* Loop over source-cluster indices */
      for (int d = 0; d < 27; d ++) {
      
	/* Allocate tiled-vector Mq and tiled-matrix Kpq */
	//110228	__shared__ float Mq[8][8][32]; // Mq[B^3][8][r/Q]; 8192 bytes
	//110228	__shared__ float Kpq[32][32]; // Kpq[r/Q][r/P]; 4096 bytes
	__shared__ real Mq[8][8][32]; // Mq[B^3][8][r/Q]; 8192 bytes in single
	__shared__ real Kpq[32][32]; // Kpq[r/Q][r/P]; 4096 bytes in single

	/* Load all the r/Q elements of Mq for eight source cells in
	   FC */
	CL_load_M256(q, d);
    
	/* Load K-indices for 27 interaction-kinds */
	if (ty == 0) {
	  Kindex[tx] = sKtable[Kindex0[d] + Kindex1[tx]];
	}

	/* Ensure that Mq[][][] and Kindex[] were loaded */
	__syncthreads();

	/* Compute Lp=Kpq*Mq for 27 interaction-kinds */
	CL_comp256x1(0, F0, S7);
 	CL_comp256x1(1, F1, S6);
 	CL_comp256x1(2, F2, S5);
 	CL_comp256x1(3, F3, S4);
 	CL_comp256x1(4, F4, S3);
 	CL_comp256x1(5, F5, S2);
 	CL_comp256x1(6, F6, S1);
 	CL_comp256x1(7, F7, S0);

 	CL_comp256x2( 8, F0, F1, S6, S7);
 	CL_comp256x2( 9, F0, F2, S5, S7);
 	CL_comp256x2(10, F0, F4, S3, S7);
 	CL_comp256x2(11, F1, F3, S4, S6);
 	CL_comp256x2(12, F1, F5, S2, S6);
 	CL_comp256x2(13, F2, F3, S4, S5);
 	CL_comp256x2(14, F2, F6, S1, S5);
 	CL_comp256x2(15, F3, F7, S0, S4);
 	CL_comp256x2(16, F4, F5, S2, S3);
 	CL_comp256x2(17, F4, F6, S1, S3);
 	CL_comp256x2(18, F5, F7, S0, S2);
 	CL_comp256x2(19, F6, F7, S0, S1);

 	CL_comp256x4(20, F0, F1, F2, F3, S4, S5, S6, S7);
 	CL_comp256x4(21, F0, F1, F4, F5, S2, S3, S6, S7);
 	CL_comp256x4(22, F0, F2, F4, F6, S1, S3, S5, S7);
 	CL_comp256x4(23, F1, F3, F5, F7, S0, S2, S4, S6);
 	CL_comp256x4(24, F2, F3, F6, F7, S0, S1, S4, S5);
 	CL_comp256x4(25, F4, F5, F6, F7, S0, S1, S2, S3);

  	CL_comp256x8(26, F0, F1, F2, F3, F4, F5, F6, F7, S0, S1, S2, S3, S4, S5, S6, S7);

      } // end loop over d
    } // end loop over q

    /* Store Lp in device-memory */
    //110228    float *Lptr = L + (bx << 14) + (ty << 11) + (p << 5) + tx; // log2(8*B^3*r)=14, log2(B^3*r)=11, log2(r/P)=5
    real *Lptr = L + (bx << 14) + (ty << 11) + (p << 5) + tx; // log2(8*B^3*r)=14, log2(B^3*r)=11, log2(r/P)=5
    *Lptr          += Lp[0];
    *(Lptr +  256) += Lp[1];
    *(Lptr +  512) += Lp[2];
    *(Lptr +  768) += Lp[3];
    *(Lptr + 1024) += Lp[4];
    *(Lptr + 1280) += Lp[5];
    *(Lptr + 1536) += Lp[6];
    *(Lptr + 1792) += Lp[7];

  } // end loop over p
}


#define CL_load_Kpq32(kindex)						\
  {									\
    /* Set a pointer to Kpq (K is stored in double column-major) */	\
    int id = (ty << 5) + tx; /* log2(r)=5 */				\
    /*110228    float *Kptr = (float *)K + (kindex << 10) + id;*/ /* log2(r^2)=10 */ \
    real *Kptr = (real *)K + (kindex << 10) + id; /* log2(r^2)=10 */	\
    /* Set a pointer to Kpq (column-major) */				\
    /*110228    float *Kpqptr = (float *)Kpq + id;*/			\
    real *Kpqptr = (real *)Kpq + id;					\
    /* Load the 1st 256 elements to Kpq[0:31][ 0: 7] */			\
    *Kpqptr = *Kptr; Kpqptr += 256; Kptr += 256;			\
    /* Load the 2nd 256 elements to Kpq[0:31][ 8:15] */			\
    *Kpqptr = *Kptr; Kpqptr += 256; Kptr += 256;			\
    /* Load the 3rd 256 elements to Kpq[0:31][16:23] */			\
    *Kpqptr = *Kptr; Kpqptr += 256; Kptr += 256;			\
    /* Load the 4th 256 elements to Kpq[0:31][24:31] */			\
    *Kpqptr = *Kptr;							\
  }

#define CL_core32x1(f,s) \
  CL_core256x1(f, s)
#define CL_comp32x1(kind, f, s)				\
  /* Skip if s is a near neighbor of f */			\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /* Load Kpq */						\
    CL_load_Kpq32(Kindex[kind]);				\
    /* Ensure that Kpq was loaded */				\
    __syncthreads();						\
    /* Compute Lp(f)=Kpq(f,s)*Mq(s) for a pair (f,s) */		\
    CL_core32x1(f, s);						\
    /* Ensure that Kpq is no longer referenced */		\
    __syncthreads();						\
  }

#define CL_core32x2(fa, fb, sa, sb) \
  CL_core256x2(fa, fb, sa, sb)
#define CL_comp32x2(kind, fa, fb, sa, sb)			\
  /* Skip if s is a near neighbor of f */			\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /* Load Kpq */						\
    CL_load_Kpq32(Kindex[kind]);				\
    /* Ensure that Kpq was loaded */				\
    __syncthreads();						\
    /* Compute Lp(f)=Kpq(f,s)*Mq(s) for two pairs (f,s) */	\
    CL_core32x2(fa, fb, sa, sb);				\
    /* Ensure that Kpq is no longer referenced */		\
    __syncthreads();						\
  }

#define CL_core32x4(fa, fb, fc, fd, sa, sb, sc, sd) \
  CL_core256x4(fa, fb, fc, fd, sa, sb, sc, sd)
#define CL_comp32x4(kind, fa, fb, fc, fd, sa, sb, sc, sd)	\
  /* Skip if s is a near neighbor of f */			\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /* Load Kpq */						\
    CL_load_Kpq32(Kindex[kind]);				\
    /* Ensure that Kpq was loaded */				\
    __syncthreads();						\
    /* Compute Lp(f)=Kpq(f,s)*Mq(s) for four pairs (f,s) */	\
    CL_core32x4(fa, fb, fc, fd, sa, sb, sc, sd);		\
    /* Ensure that Kpq is no longer referenced */		\
    __syncthreads();						\
  }

#define CL_core32x8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh) \
  CL_core256x8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh)
#define CL_comp32x8(kind, fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh) \
  /* Skip if s is a near neighbor of f */				\
  if (Kindex[kind] != NULL_KINDEX) {					\
    /* Load Kpq */							\
    CL_load_Kpq32(Kindex[kind]);					\
    /* Ensure that Kpq was loaded */					\
    __syncthreads();							\
    /* Compute Lp(f)=Kpq(f,s)*Mq(s) for eight pairs (f,s) */		\
    CL_core32x8(fa, fb, fc, fd, fe, ff, fg, fh, sa, sb, sc, sd, se, sf, sg, sh); \
    /* Ensure that Kpq is no longer referenced */			\
    __syncthreads();							\
  }

#define CL_load_M32(d)						\
  if (ilist[ty][d] != NULL_CELL) {				\
    /*110228    float *Mptr = M + (ilist[ty][d] << 5) + tx;*/	/* log2(r)=5 */	\
    real *Mptr = M + (ilist[ty][d] << 5) + tx;	/* log2(r)=5 */		\
    Mq[ty][0][tx] = *Mptr;        				\
    Mq[ty][1][tx] = *(Mptr +  32);				\
    Mq[ty][2][tx] = *(Mptr +  64);				\
    Mq[ty][3][tx] = *(Mptr +  96);				\
    Mq[ty][4][tx] = *(Mptr + 128);				\
    Mq[ty][5][tx] = *(Mptr + 160);				\
    Mq[ty][6][tx] = *(Mptr + 192);				\
    Mq[ty][7][tx] = *(Mptr + 224);				\
  } else {							\
    /*110228    Mq[ty][0][tx] = 0.0f;*/					\
    /*110228    Mq[ty][1][tx] = 0.0f;*/					\
    /*110228    Mq[ty][2][tx] = 0.0f;*/					\
    /*110228    Mq[ty][3][tx] = 0.0f;*/					\
    /*110228    Mq[ty][4][tx] = 0.0f;*/					\
    /*110228    Mq[ty][5][tx] = 0.0f;*/					\
    /*110228    Mq[ty][6][tx] = 0.0f;*/					\
    /*110228    Mq[ty][7][tx] = 0.0f;*/					\
    Mq[ty][0][tx] = ZERO;				\
    Mq[ty][1][tx] = ZERO;				\
    Mq[ty][2][tx] = ZERO;				\
    Mq[ty][3][tx] = ZERO;				\
    Mq[ty][4][tx] = ZERO;				\
    Mq[ty][5][tx] = ZERO;				\
    Mq[ty][6][tx] = ZERO;				\
    Mq[ty][7][tx] = ZERO;				\
  }


//110228__global__ void m2l_kern_cluster_blocking_r32b2p1q1(float *L, float *K, float *M, int *sourceclusters, int *Ktable, int FCHsta)
__global__ void m2l_kern_cluster_blocking_r32b2p1q1(real *L, real *K, real *M, int *sourceclusters, int *Ktable, int FCHsta)
{
  /* Load lists of source-clusters for B^3 field-clusters in the
     current chunk of size B(=2), to which the current (bx-th)
     thread-block is assigned */
  __shared__ int ilist[8][32]; // ilist[0:B^3-1][27 or more]; 1024 bytes
  {
    int id = (ty << 5) + tx; // log2(r/P)=5
    ilist[ty][tx] = sourceclusters[((FCHsta + bx) << 8) + id]; // log2(r)=8
  }

  /* Copy Ktable into shared memory with 256 threads */
  /* Load the indexes to K */
  __shared__ int sKtable[343]; // 1372 bytes
  {
    int id = (ty << 5) + tx;
    sKtable[id] = Ktable[id];
    if (id < 87) {
      sKtable[id + 256] = Ktable[id + 256];
    }
  }

  /* Load Kindex0[] and Kindex1[] */
  __shared__ int Kindex0[32], Kindex1[32], Kindex[32]; // 27 or more; 384 bytes
  {
    int id = (ty << 5) + tx; // log2(r/P)=5
    load_Kindex0_Kindex1(id);
  }

  /* Ensure that ilist[], sKtable[], Kindex0[], and Kindex1[] were loaded */
  __syncthreads();

  /* 
     Note that P=Q=1, thus p=q=0
  */
    
  /* Initialize the tx-th element of tiled-vector Lp for the ty-th
     field cluster (FC) in the current chunk */
  //110228  float Lp[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // log2(B^3)=8
  real Lp[8] = {ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO}; // log2(B^3)=8
  
  /* Loop over source-cluster indices */
  for (int d = 0; d < 27; d ++) {
      
    /* Allocate tiled-vector Mq and tiled-matrix Kpq */
    //110228    __shared__ float Mq[8][8][32]; // Mq[B^3][8][r/Q]; 8192 bytes
    //110228    __shared__ float Kpq[32][32]; // Kpq[r/Q][r/P]; 4096 bytes
    __shared__ real Mq[8][8][32]; // Mq[B^3][8][r/Q]; 8192 bytes in single
    __shared__ real Kpq[32][32]; // Kpq[r/Q][r/P]; 4096 bytes in single

    /* Load all the r/Q elements of Mq for the eight source cells in
       FC */
    CL_load_M32(d);
    
    /* Load K-indices for 27 interaction-kinds */
    if (ty == 0) {
      Kindex[tx] = sKtable[Kindex0[d] + Kindex1[tx]];
    }

    /* Ensure that Mq[][][] and Kindex[] were loaded */
    __syncthreads();

    /* Compute Lp=Kpq*Mq for 27 interaction-kinds */
    CL_comp32x1(0, F0, S7);
    CL_comp32x1(1, F1, S6);
    CL_comp32x1(2, F2, S5);
    CL_comp32x1(3, F3, S4);
    CL_comp32x1(4, F4, S3);
    CL_comp32x1(5, F5, S2);
    CL_comp32x1(6, F6, S1);
    CL_comp32x1(7, F7, S0);
    
    CL_comp32x2( 8, F0, F1, S6, S7);
    CL_comp32x2( 9, F0, F2, S5, S7);
    CL_comp32x2(10, F0, F4, S3, S7);
    CL_comp32x2(11, F1, F3, S4, S6);
    CL_comp32x2(12, F1, F5, S2, S6);
    CL_comp32x2(13, F2, F3, S4, S5);
    CL_comp32x2(14, F2, F6, S1, S5);
    CL_comp32x2(15, F3, F7, S0, S4);
    CL_comp32x2(16, F4, F5, S2, S3);
    CL_comp32x2(17, F4, F6, S1, S3);
    CL_comp32x2(18, F5, F7, S0, S2);
    CL_comp32x2(19, F6, F7, S0, S1);
    
    CL_comp32x4(20, F0, F1, F2, F3, S4, S5, S6, S7);
    CL_comp32x4(21, F0, F1, F4, F5, S2, S3, S6, S7);
    CL_comp32x4(22, F0, F2, F4, F6, S1, S3, S5, S7);
    CL_comp32x4(23, F1, F3, F5, F7, S0, S2, S4, S6);
    CL_comp32x4(24, F2, F3, F6, F7, S0, S1, S4, S5);
    CL_comp32x4(25, F4, F5, F6, F7, S0, S1, S2, S3);
    
    CL_comp32x8(26, F0, F1, F2, F3, F4, F5, F6, F7, S0, S1, S2, S3, S4, S5, S6, S7);
    
  } // end loop over d

  /* Store Lp in device-memory */
  //110228  float *Lptr = L + (bx << 11) + (ty << 8) + tx; // log2(8*B^3*r), log2(B^3*r)=8
  real *Lptr = L + (bx << 11) + (ty << 8) + tx; // log2(8*B^3*r), log2(B^3*r)=8
  *Lptr         += Lp[0];
  *(Lptr +  32) += Lp[1];
  *(Lptr +  64) += Lp[2];
  *(Lptr +  96) += Lp[3];
  *(Lptr + 128) += Lp[4];
  *(Lptr + 160) += Lp[5];
  *(Lptr + 192) += Lp[6];
  *(Lptr + 224) += Lp[7];
}
/**************************************************************************/
#elif defined(CUDA_VER45A)
/**************************************************************************/
#error This version does not exist.
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
#endif /* M2L_KERN_CLUSTER_BLOCKING_CU */
