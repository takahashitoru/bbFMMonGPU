#ifndef M2L_KERN_SIBLING_BLOCKING_CU
#define M2L_KERN_SIBLING_BLOCKING_CU

/**************************************************************************/
#if defined(CUDA_VER45)
/**************************************************************************/
/**************************************************************************/
#if defined(CUDA_VER45B) || defined(CUDA_VER45C) || defined(CUDA_VER45D) || defined(CUDA_VER45E) || defined(CUDA_VER45F) || defined(CUDA_VER45G) || defined(CUDA_VER45H) || defined(CUDA_VER45I)
/**************************************************************************/
/* Based on CUDA_VER45A */
#include "real.h"

#define bx blockIdx.x
#define tx threadIdx.x

#ifndef NULLCELL
#define NULLCELL - 1
#endif

#ifndef NULL_CELL
#define NULL_CELL NULLCELL
#endif

#ifndef NULL_KINDEX
#define NULL_KINDEX - 1
#endif

/* Setup for internal parameters */
#if !defined(SI_UNROLLING_R256)
#define SI_UNROLLING_R256 (4) // 1, 2, 4, 8, or 32; 4 is the best for C2050+SDK3.2
#endif
#if !defined(SI_UNROLLING_R32)
#define SI_UNROLLING_R32  (1) // 1, 2, 4, or 32; 1 is the best for C2050+SDK3.2
#endif

#if !defined(SINGLE) // if double-precision
#if (SI_UNROLLING_R256 == 32)
#error This does not work for C2050 with SDK 3.2.
#endif
#endif

/* Define 27 interction-kinds. See Table 4 (left) in the paper */
#define F0S7  0 /* F0 vs S7 */
#define F1S6  1 /* F1 vs S6 */
#define F2S5  2 /* F2 vs S5 */
#define F3S4  3 /* F3 vs S4 */
#define F4S3  4 /* F4 vs S3 */
#define F5S2  5 /* F5 vs S2 */
#define F6S1  6 /* F6 vs S1 */
#define F7S0  7 /* F7 vs S0 */
#define F0S6  8 /* F0,F1 vs S6,S7 */
#define F0S5  9 /* F0,F2 vs S5,S7 */
#define F0S3 10 /* F0,F4 vs S3,S7 */
#define F1S4 11 /* F1,F3 vs S4,S6 */
#define F1S2 12 /* F1,F5 vs S2,S6 */
#define F2S4 13 /* F2,F3 vs S4,S5 */
#define F2S1 14 /* F2,F6 vs S1,S5 */
#define F3S0 15 /* F3,F7 vs S0,S4 */
#define F4S2 16 /* F4,F5 vs S2,S3 */
#define F4S1 17 /* F4,F6 vs S1,S3 */
#define F5S0 18 /* F5,F7 vs S0,S2 */
#define F6S0 19 /* F6,F7 vs S0,S1 */
#define F0S4 20 /* F0,F1,F2,F3 vs S4,S5,S6,S7 */
#define F0S2 21 /* F0,F1,F4,F5 vs S2,S3,S6,S7 */
#define F0S1 22 /* F0,F2,F4,F6 vs S1,S3,S5,S7 */
#define F1S0 23 /* F1,F3,F5,F7 vs S0,S2,S4,S6 */
#define F2S0 24 /* F2,F3,F6,F7 vs S0,S1,S4,S5 */
#define F4S0 25 /* F4,F5,F6,F7 vs S0,S1,S2,S3 */
#define F0S0 26 /* F0,F1,F2,F3,F4,F5,F6,F7 vs S0,S1,S2,S3,S4,S5,S6,S7 */


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

#define I0 0
#define I1 256
#define I2 512
#define I3 768
#define I4 1024
#define I5 1280
#define I6 1536
#define I7 1792
#define J0 0
#define J1 1
#define J2 2
#define J3 3
#define J4 4
#define J5 5
#define J6 6
#define J7 7

#define cal256x1(s0, jp0, jp1)			\
  Ksave1 = *Kptr; Kptr += 256;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  Ksave0 = *Kptr; Kptr += 256;			\
  L0 += Ksave1 * sM[s0][jp1]

__device__ void SI_comp256x1(real *K, int Kindex[32], real *Lptr, real sM[4][256], int kind, int f0, int s0)
{
  if (Kindex[kind] != NULL_KINDEX) {
#if (SI_UNROLLING_R256 == 32) // Prefetch K-data and unroll the column loop 32 times
#error Not implemented.
#elif (SI_UNROLLING_R256 == 8) // Prefetch K-data and unroll the column loop eight times
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    for (int j = 0; j < 248; j += 8) {
      real Ksave1;
      cal256x1(s0, j    , j + 1);
      cal256x1(s0, j + 2, j + 3);
      cal256x1(s0, j + 4, j + 5);
      cal256x1(s0, j + 6, j + 7);
    }
    real Ksave1;
    cal256x1(s0, 248, 249);
    cal256x1(s0, 250, 251);
    cal256x1(s0, 252, 253);
    Ksave1 = *Kptr; //Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L0 += Ksave1 * sM[s0][255];
    *(Lptr + f0) += L0;
#elif (SI_UNROLLING_R256 == 4) // Prefetch K-data and unroll the column loop four times
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    for (int j = 0; j < 252; j += 4) {
      real Ksave1;
      cal256x1(s0, j    , j + 1);
      cal256x1(s0, j + 2, j + 3);
    }
    real Ksave1;
    cal256x1(s0, 252, 253);
    Ksave1 = *Kptr; //Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L0 += Ksave1 * sM[s0][255];
    *(Lptr + f0) += L0;
#elif (SI_UNROLLING_R256 == 2) // Prefetch K-data and unroll the column loop twice
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    for (int j = 0; j < 254; j += 2) {
      real Ksave1;
      cal256x1(s0, j    , j + 1);
    }
    real Ksave1;
    Ksave1 = *Kptr; //Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L0 += Ksave1 * sM[s0][255];
    *(Lptr + f0) += L0;
#elif (SI_UNROLLING_R256 == 1) // unoptimised version
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real L0 = ZERO;
    for (int j = 0; j < 256; j ++) {
      L0 += *Kptr * sM[s0][j]; Kptr += 256;
    }
    *(Lptr + f0) += L0;
#else
#error Invalid parameter.
#endif
  }
}

#define cal256x2(s0, s1, jp0, jp1)		\
  Ksave1 = *Kptr; Kptr += 256;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  L1 += Ksave0 * sM[s1][jp0];			\
  Ksave0 = *Kptr; Kptr += 256;			\
  L0 += Ksave1 * sM[s0][jp1];			\
  L1 += Ksave1 * sM[s1][jp1]

__device__ void SI_comp256x2(real *K, int Kindex[32], real *Lptr, real sM[4][256], int kind, int f0, int f1, int s0, int s1)
{
  if (Kindex[kind] != NULL_KINDEX) {
#if (SI_UNROLLING_R256 == 32) // Prefetch K-data and unroll the column loop 32 times
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    real L1 = ZERO;
    for (int j = 0; j < 224; j += 32) {
      real Ksave1;
      cal256x2(s0, s1, j     , j +  1);
      cal256x2(s0, s1, j +  2, j +  3);
      cal256x2(s0, s1, j +  4, j +  5);
      cal256x2(s0, s1, j +  6, j +  7);
      cal256x2(s0, s1, j +  8, j +  9);
      cal256x2(s0, s1, j + 10, j + 11);
      cal256x2(s0, s1, j + 12, j + 13);
      cal256x2(s0, s1, j + 14, j + 15);
      cal256x2(s0, s1, j + 16, j + 17);
      cal256x2(s0, s1, j + 18, j + 19);
      cal256x2(s0, s1, j + 20, j + 21);
      cal256x2(s0, s1, j + 22, j + 23);
      cal256x2(s0, s1, j + 24, j + 25);
      cal256x2(s0, s1, j + 26, j + 27);
      cal256x2(s0, s1, j + 28, j + 29);
      cal256x2(s0, s1, j + 30, j + 31);
    }
    real Ksave1;
    cal256x2(s0, s1, 224, 225);
    cal256x2(s0, s1, 226, 227);
    cal256x2(s0, s1, 228, 229);
    cal256x2(s0, s1, 230, 231);
    cal256x2(s0, s1, 232, 233);
    cal256x2(s0, s1, 234, 235);
    cal256x2(s0, s1, 236, 237);
    cal256x2(s0, s1, 238, 239);
    cal256x2(s0, s1, 240, 241);
    cal256x2(s0, s1, 242, 243);
    cal256x2(s0, s1, 244, 245);
    cal256x2(s0, s1, 246, 247);
    cal256x2(s0, s1, 248, 249);
    cal256x2(s0, s1, 250, 251);
    cal256x2(s0, s1, 252, 253);
    Ksave1 = *Kptr; Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L1 += Ksave0 * sM[s1][254];
    L0 += Ksave1 * sM[s0][255];
    L1 += Ksave1 * sM[s1][255];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
#elif (SI_UNROLLING_R256 == 8) // Prefetch K-data and unroll the column loop eight times
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    real L1 = ZERO;
    for (int j = 0; j < 248; j += 8) {
      real Ksave1;
      cal256x2(s0, s1, j    , j + 1);
      cal256x2(s0, s1, j + 2, j + 3);
      cal256x2(s0, s1, j + 4, j + 5);
      cal256x2(s0, s1, j + 6, j + 7);
    }
    real Ksave1;
    cal256x2(s0, s1, 248, 249);
    cal256x2(s0, s1, 250, 251);
    cal256x2(s0, s1, 252, 253);
    Ksave1 = *Kptr; Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L1 += Ksave0 * sM[s1][254];
    L0 += Ksave1 * sM[s0][255];
    L1 += Ksave1 * sM[s1][255];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
#elif (SI_UNROLLING_R256 == 4) // Prefetch K-data and unroll the column loop four times
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    real L1 = ZERO;
    for (int j = 0; j < 252; j += 4) {
      real Ksave1;
      cal256x2(s0, s1, j    , j + 1);
      cal256x2(s0, s1, j + 2, j + 3);
    }
    real Ksave1;
    cal256x2(s0, s1, 252, 253);
    Ksave1 = *Kptr; Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L1 += Ksave0 * sM[s1][254];
    L0 += Ksave1 * sM[s0][255];
    L1 += Ksave1 * sM[s1][255];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
#elif (SI_UNROLLING_R256 == 2) // Prefetch K-data and unroll the column loop twice
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    real L1 = ZERO;
    for (int j = 0; j < 254; j += 2) {
      real Ksave1;
      cal256x2(s0, s1, j    , j + 1);
    }
    real Ksave1;
    Ksave1 = *Kptr; Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L1 += Ksave0 * sM[s1][254];
    L0 += Ksave1 * sM[s0][255];
    L1 += Ksave1 * sM[s1][255];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
#elif (SI_UNROLLING_R256 == 1) // unoptimised version
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real L0 = ZERO;
    real L1 = ZERO;
    for (int j = 0; j < 256; j ++) {
      real Ksave = *Kptr; Kptr += 256;
      L0 += Ksave * sM[s0][j];
      L1 += Ksave * sM[s1][j];
    }
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
#else
#error Invalid parameter.
#endif
  }
}

#define cal256x4(s0, s1, s2, s3, jp0, jp1)	\
  Ksave1 = *Kptr; Kptr += 256;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  L1 += Ksave0 * sM[s1][jp0];			\
  L2 += Ksave0 * sM[s2][jp0];			\
  L3 += Ksave0 * sM[s3][jp0];			\
  Ksave0 = *Kptr; Kptr += 256;			\
  L0 += Ksave1 * sM[s0][jp1];			\
  L1 += Ksave1 * sM[s1][jp1];			\
  L2 += Ksave1 * sM[s2][jp1];			\
  L3 += Ksave1 * sM[s3][jp1]

__device__ void SI_comp256x4(real *K, int Kindex[32], real *Lptr, real sM[4][256], int kind, int f0, int f1, int f2, int f3, int s0, int s1, int s2, int s3)
{
  if (Kindex[kind] != NULL_KINDEX) {
#if (SI_UNROLLING_R256 == 32) // Prefetch K-data and unroll the column loop 32 times
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    real L1 = ZERO;
    real L2 = ZERO;
    real L3 = ZERO;
    for (int j = 0; j < 224; j += 32) {
      real Ksave1;
      cal256x4(s0, s1, s2, s3, j     , j +  1);
      cal256x4(s0, s1, s2, s3, j +  2, j +  3);
      cal256x4(s0, s1, s2, s3, j +  4, j +  5);
      cal256x4(s0, s1, s2, s3, j +  6, j +  7);
      cal256x4(s0, s1, s2, s3, j +  8, j +  9);
      cal256x4(s0, s1, s2, s3, j + 10, j + 11);
      cal256x4(s0, s1, s2, s3, j + 12, j + 13);
      cal256x4(s0, s1, s2, s3, j + 14, j + 15);
      cal256x4(s0, s1, s2, s3, j + 16, j + 17);
      cal256x4(s0, s1, s2, s3, j + 18, j + 19);
      cal256x4(s0, s1, s2, s3, j + 20, j + 21);
      cal256x4(s0, s1, s2, s3, j + 22, j + 23);
      cal256x4(s0, s1, s2, s3, j + 24, j + 25);
      cal256x4(s0, s1, s2, s3, j + 26, j + 27);
      cal256x4(s0, s1, s2, s3, j + 28, j + 29);
      cal256x4(s0, s1, s2, s3, j + 30, j + 31);
    }
    real Ksave1;
    cal256x4(s0, s1, s2, s3, 224, 225);
    cal256x4(s0, s1, s2, s3, 226, 227);
    cal256x4(s0, s1, s2, s3, 228, 229);
    cal256x4(s0, s1, s2, s3, 230, 231);
    cal256x4(s0, s1, s2, s3, 232, 233);
    cal256x4(s0, s1, s2, s3, 234, 235);
    cal256x4(s0, s1, s2, s3, 236, 237);
    cal256x4(s0, s1, s2, s3, 238, 239);
    cal256x4(s0, s1, s2, s3, 240, 241);
    cal256x4(s0, s1, s2, s3, 242, 243);
    cal256x4(s0, s1, s2, s3, 244, 245);
    cal256x4(s0, s1, s2, s3, 246, 247);
    cal256x4(s0, s1, s2, s3, 248, 249);
    cal256x4(s0, s1, s2, s3, 250, 251);
    cal256x4(s0, s1, s2, s3, 252, 253);
    Ksave1 = *Kptr; Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L1 += Ksave0 * sM[s1][254];
    L2 += Ksave0 * sM[s2][254];
    L3 += Ksave0 * sM[s3][254];
    L0 += Ksave1 * sM[s0][255];
    L1 += Ksave1 * sM[s1][255];
    L2 += Ksave1 * sM[s2][255];
    L3 += Ksave1 * sM[s3][255];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
    *(Lptr + f2) += L2;
    *(Lptr + f3) += L3;
#elif (SI_UNROLLING_R256 == 8) // Prefetch K-data and unroll the column loop eight times
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    real L1 = ZERO;
    real L2 = ZERO;
    real L3 = ZERO;
    for (int j = 0; j < 248; j += 8) {
      real Ksave1;
      cal256x4(s0, s1, s2, s3, j    , j + 1);
      cal256x4(s0, s1, s2, s3, j + 2, j + 3);
      cal256x4(s0, s1, s2, s3, j + 4, j + 5);
      cal256x4(s0, s1, s2, s3, j + 6, j + 7);
    }
    real Ksave1;
    cal256x4(s0, s1, s2, s3, 248, 249);
    cal256x4(s0, s1, s2, s3, 250, 251);
    cal256x4(s0, s1, s2, s3, 252, 253);
    Ksave1 = *Kptr; Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L1 += Ksave0 * sM[s1][254];
    L2 += Ksave0 * sM[s2][254];
    L3 += Ksave0 * sM[s3][254];
    L0 += Ksave1 * sM[s0][255];
    L1 += Ksave1 * sM[s1][255];
    L2 += Ksave1 * sM[s2][255];
    L3 += Ksave1 * sM[s3][255];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
    *(Lptr + f2) += L2;
    *(Lptr + f3) += L3;
#elif (SI_UNROLLING_R256 == 4) // Prefetch K-data and unroll the column loop four times
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    real L1 = ZERO;
    real L2 = ZERO;
    real L3 = ZERO;
    for (int j = 0; j < 252; j += 4) {
      real Ksave1;
      cal256x4(s0, s1, s2, s3, j    , j + 1);
      cal256x4(s0, s1, s2, s3, j + 2, j + 3);
    }
    real Ksave1;
    cal256x4(s0, s1, s2, s3, 252, 253);
    Ksave1 = *Kptr; Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L1 += Ksave0 * sM[s1][254];
    L2 += Ksave0 * sM[s2][254];
    L3 += Ksave0 * sM[s3][254];
    L0 += Ksave1 * sM[s0][255];
    L1 += Ksave1 * sM[s1][255];
    L2 += Ksave1 * sM[s2][255];
    L3 += Ksave1 * sM[s3][255];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
    *(Lptr + f2) += L2;
    *(Lptr + f3) += L3;
#elif (SI_UNROLLING_R256 == 2) // Prefetch K-data and unroll the column loop twice
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real Ksave0 = *Kptr; Kptr += 256;
    real L0 = ZERO;
    real L1 = ZERO;
    real L2 = ZERO;
    real L3 = ZERO;
    for (int j = 0; j < 254; j += 2) {
      real Ksave1;
      cal256x4(s0, s1, s2, s3, j    , j + 1);
    }
    real Ksave1;
    Ksave1 = *Kptr; Kptr += 256;
    L0 += Ksave0 * sM[s0][254];
    L1 += Ksave0 * sM[s1][254];
    L2 += Ksave0 * sM[s2][254];
    L3 += Ksave0 * sM[s3][254];
    L0 += Ksave1 * sM[s0][255];
    L1 += Ksave1 * sM[s1][255];
    L2 += Ksave1 * sM[s2][255];
    L3 += Ksave1 * sM[s3][255];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
    *(Lptr + f2) += L2;
    *(Lptr + f3) += L3;
#elif (SI_UNROLLING_R256 == 1) // unoptimised version
    real *Kptr = K + (Kindex[kind] << 16) + tx;
    real L0 = ZERO;
    real L1 = ZERO;
    real L2 = ZERO;
    real L3 = ZERO;
    for (int j = 0; j < 256; j ++) {
      real Ksave = *Kptr; Kptr += 256;
      L0 += Ksave * sM[s0][j];
      L1 += Ksave * sM[s1][j];
      L2 += Ksave * sM[s2][j];
      L3 += Ksave * sM[s3][j];
    }
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
    *(Lptr + f2) += L2;
    *(Lptr + f3) += L3;
#else
#error Invalid parameter.
#endif
  }
}

#define SI_comp256x8(kind, f0, f1, f2, f3, s0, s1, s2, s3)	\
  SI_comp256x4(kind, f0, f1, f2, f3, s0, s1, s2, s3)


__global__ void m2l_kern_sibling_blocking_r256(real *L, real *K, real *M, int *sourceclusters, int *Ktable, int FCsta)
{

  /* Set the pointer to L for the 0th sibling in the current
     observation cluster OC, to which bx-th thread-block is
     assigned */
  real *Lptr = L + (bx << 11) + tx; // log2(r*8)=11
  
  /* Load 0th siblings of 27 source clusters (SCs) for OC;
     ilist[27:31] are unused */
  __shared__ int ilist[32]; // 27 or more; 128 bytes
  if (tx < 32) {
    ilist[tx] = sourceclusters[((FCsta + bx) << 5) + tx]; // log2(32)=5
  }

  /* Copy Ktable in shared memory */
  __shared__ int sKtable[343]; // 1372 bytes
  sKtable[tx] = Ktable[tx];
  if (tx < 87) {
    sKtable[tx + 256] = Ktable[tx + 256];
  }
  
  /* Load Kindex0[] and Kindex1[] */
  __shared__ int Kindex0[32], Kindex1[32], Kindex[32]; // 27 or more; 384 bytes
  if (tx < 32) {
    load_Kindex0_Kindex1(tx);
  }

  /* Ensure that ilist[], sKtable[], Kindex0[], and Kindex1[] were loaded */
  __syncthreads();
  
  /* Loop over source-cluster indices */
  for (int d = 0; d < 27; d ++) {
    
    /* If the 0th sibling, which is the representative of SC, does not
       exist in the hierachy (then, the other seven siblings in SC
       neither exist) or SC coincides with OC, we can skip the
       computaion for SC */
    if (ilist[d] != NULL_CELL) {

      /* Load K-indices for all the 27 interaction-kinds */
      if (tx < 32) {
	Kindex[tx] = sKtable[Kindex0[d] + Kindex1[tx]];
      }
      
      /* Ensure that Kindex[] was loaded */
      __syncthreads();
      
      /* Allocate the memory for ONLY FOUR M-vectors in SC because of
	 the shortage of shared memory */
      __shared__ real sM[4][256]; // 4096 bytes for single

      {
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[0][tx] = *(Mptr + I0); /* S0 */
	sM[1][tx] = *(Mptr + I1); /* S1 */
	sM[2][tx] = *(Mptr + I2); /* S2 */
	sM[3][tx] = *(Mptr + I3); /* S3 */
      } /* sM[]={S0,S1,S2,S3} */
      __syncthreads();
    
      SI_comp256x1(K, Kindex, Lptr, sM, F4S3, I4, J3);
      SI_comp256x1(K, Kindex, Lptr, sM, F5S2, I5, J2);
      SI_comp256x1(K, Kindex, Lptr, sM, F6S1, I6, J1);
      SI_comp256x1(K, Kindex, Lptr, sM, F7S0, I7, J0);
      SI_comp256x2(K, Kindex, Lptr, sM, F5S0, I5, I7, J0, J2);
      SI_comp256x2(K, Kindex, Lptr, sM, F6S0, I6, I7, J0, J1);
      SI_comp256x4(K, Kindex, Lptr, sM, F4S0, I4, I5, I6, I7, J0, J1, J2, J3);
      SI_comp256x4(K, Kindex, Lptr, sM, F0S0, I0, I1, I2, I3, J0, J1, J2, J3); // x8
      __syncthreads();
      
      {
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[1][tx] = *(Mptr + I4); /* S4 */
	sM[3][tx] = *(Mptr + I6); /* S6 */
      } /* sM[]={S0,S4,S2,S6} */
      __syncthreads();
      
      SI_comp256x2(K, Kindex, Lptr, sM, F1S4, I1, I3, J1, J3);
      SI_comp256x2(K, Kindex, Lptr, sM, F1S2, I1, I5, J2, J3);
      SI_comp256x4(K, Kindex, Lptr, sM, F1S0, I1, I3, I5, I7, J0, J2, J1, J3);
      __syncthreads();
      
      {
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[0][tx] = *(Mptr + I5); /* S5 */
	sM[2][tx] = *(Mptr + I7); /* S7 */
      } /* sM[]={S5,S4,S7,S6} */
      __syncthreads();
      
      SI_comp256x1(K, Kindex, Lptr, sM, F0S7, I0, J2);
      SI_comp256x1(K, Kindex, Lptr, sM, F1S6, I1, J3);
      SI_comp256x1(K, Kindex, Lptr, sM, F2S5, I2, J0);
      SI_comp256x1(K, Kindex, Lptr, sM, F3S4, I3, J1);
      SI_comp256x2(K, Kindex, Lptr, sM, F0S5, I0, I2, J0, J2);
      SI_comp256x2(K, Kindex, Lptr, sM, F2S4, I2, I3, J1, J0);
      SI_comp256x4(K, Kindex, Lptr, sM, F0S4, I0, I1, I2, I3, J1, J0, J3, J2);
      SI_comp256x4(K, Kindex, Lptr, sM, F0S0, I4, I5, I6, I7, J1, J0, J3, J2); // x8
      __syncthreads();
      
      {
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[0][tx] = *(Mptr + I2); /* S2 */
	sM[1][tx] = *(Mptr + I3); /* S3 */
      } /* sM[]={S2,S3,S7,S6} */
      __syncthreads();
      
      SI_comp256x2(K, Kindex, Lptr, sM, F0S6, I0, I1, J3, J2);
      SI_comp256x2(K, Kindex, Lptr, sM, F4S2, I4, I5, J0, J1);
      SI_comp256x4(K, Kindex, Lptr, sM, F0S2, I0, I1, I4, I5, J0, J1, J3, J2);
      __syncthreads();
      
      {
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[0][tx] = *(Mptr + I1); /* S1 */
	sM[3][tx] = *(Mptr + I5); /* S5 */
      } /* sM[]={S1,S3,S7,S5} */
      __syncthreads();
      
      SI_comp256x2(K, Kindex, Lptr, sM, F0S3, I0, I4, J1, J2);
      SI_comp256x2(K, Kindex, Lptr, sM, F4S1, I4, I6, J0, J1);
      SI_comp256x4(K, Kindex, Lptr, sM, F0S1, I0, I2, I4, I6, J0, J1, J3, J2);
      __syncthreads();
      
      {
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[1][tx] = *(Mptr + I0); /* S0 */
	sM[2][tx] = *(Mptr + I4); /* S4 */
      } /* sM[]={S1,S0,S4,S5} */
      __syncthreads();
      
      SI_comp256x2(K, Kindex, Lptr, sM, F2S1, I2, I6, J0, J3);
      SI_comp256x2(K, Kindex, Lptr, sM, F3S0, I3, I7, J1, J2);
      SI_comp256x4(K, Kindex, Lptr, sM, F2S0, I2, I3, I6, I7, J1, J0, J2, J3);
      __syncthreads();
    }
  }
}

#define cal32x1(s0, jp0, jp1)			\
  Ksave1 = *Kptr; Kptr += 32;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  Ksave0 = *Kptr; Kptr += 32;			\
  L0 += Ksave1 * sM[s0][jp1]

__device__ void SI_comp32x1(real *K, int Kindex[32], real *Lptr, real sM[4][32], int kind, int f0, int s0)
{
  if (Kindex[kind] != NULL_KINDEX) {
#if (SI_UNROLLING_R32 == 32) // Prefetch K-data and unroll the column loop 32 times
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real Ksave0 = *Kptr; Kptr += 32;
    real L0 = ZERO;
    real Ksave1;
    cal32x1(s0,  0,  1);
    cal32x1(s0,  2,  3);
    cal32x1(s0,  4,  5);
    cal32x1(s0,  6,  7);
    cal32x1(s0,  8,  9);
    cal32x1(s0, 10, 11);
    cal32x1(s0, 12, 13);
    cal32x1(s0, 14, 15);
    cal32x1(s0, 16, 17);
    cal32x1(s0, 18, 19);
    cal32x1(s0, 20, 21);
    cal32x1(s0, 22, 23);
    cal32x1(s0, 24, 25);
    cal32x1(s0, 26, 27);
    cal32x1(s0, 28, 29);
    Ksave1 = *Kptr; //Kptr += 32;
    L0 += Ksave0 * sM[s0][30];
    L0 += Ksave1 * sM[s0][31];
    *(Lptr + f0) += L0;
#elif (SI_UNROLLING_R32 == 4) // Prefetch K-data and unroll the column loop four times
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real Ksave0 = *Kptr; Kptr += 32;
    real L0 = ZERO;
    for (int j = 0; j < 28; j += 4) {
      real Ksave1;
      cal32x1(s0, j    , j + 1);
      cal32x1(s0, j + 2, j + 3);
    }
    real Ksave1;
    cal32x1(s0, 28, 29);
    Ksave1 = *Kptr; //Kptr += 32;
    L0 += Ksave0 * sM[s0][30];
    L0 += Ksave1 * sM[s0][31];
    *(Lptr + f0) += L0;
#elif (SI_UNROLLING_R32 == 2) // Prefetch K-data and unroll the column loop twice
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real Ksave0 = *Kptr; Kptr += 32;
    real L0 = ZERO;
    for (int j = 0; j < 30; j += 2) {
      real Ksave1;
      cal32x1(s0, j    , j + 1);
    }
    real Ksave1;
    Ksave1 = *Kptr; //Kptr += 32;
    L0 += Ksave0 * sM[s0][30];
    L0 += Ksave1 * sM[s0][31];
    *(Lptr + f0) += L0;
#elif (SI_UNROLLING_R32 == 1) // unoptimized version
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real L0 = ZERO;
    for (int j = 0; j < 32; j ++) {
      L0 += *Kptr * sM[s0][j]; Kptr += 32;
    }
    *(Lptr + f0) += L0;
#else
#error Invalid parameter.
#endif
  }
}

#define cal32x2(s0, s1, jp0, jp1)		\
  Ksave1 = *Kptr; Kptr += 32;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  L1 += Ksave0 * sM[s1][jp0];			\
  Ksave0 = *Kptr; Kptr += 32;			\
  L0 += Ksave1 * sM[s0][jp1];			\
  L1 += Ksave1 * sM[s1][jp1]

__device__ void SI_comp32x2(real *K, int Kindex[32], real *Lptr, real sM[4][32], int kind, int f0, int f1, int s0, int s1)
{
  if (Kindex[kind] != NULL_KINDEX) {
#if (SI_UNROLLING_R32 == 32) // Prefetch K-data and unroll the column loop 32 times
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real Ksave0 = *Kptr; Kptr += 32;
    real L0 = ZERO;
    real L1 = ZERO;
    real Ksave1;
    cal32x2(s0, s1,  0,  1);
    cal32x2(s0, s1,  2,  3);
    cal32x2(s0, s1,  4,  5);
    cal32x2(s0, s1,  6,  7);
    cal32x2(s0, s1,  8,  9);
    cal32x2(s0, s1, 10, 11);
    cal32x2(s0, s1, 12, 13);
    cal32x2(s0, s1, 14, 15);
    cal32x2(s0, s1, 16, 17);
    cal32x2(s0, s1, 18, 19);
    cal32x2(s0, s1, 20, 21);
    cal32x2(s0, s1, 22, 23);
    cal32x2(s0, s1, 24, 25);
    cal32x2(s0, s1, 26, 27);
    cal32x2(s0, s1, 28, 29);
    Ksave1 = *Kptr; Kptr += 32;
    L0 += Ksave0 * sM[s0][30];
    L1 += Ksave0 * sM[s1][30];
    L0 += Ksave1 * sM[s0][31];
    L1 += Ksave1 * sM[s1][31];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
#elif (SI_UNROLLING_R32 == 4) // Prefetch K-data and unroll the column loop four times
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real Ksave0 = *Kptr; Kptr += 32;
    real L0 = ZERO;
    real L1 = ZERO;
    for (int j = 0; j < 28; j += 4) {
      real Ksave1;
      cal32x2(s0, s1, j    , j + 1);
      cal32x2(s0, s1, j + 2, j + 3);
    }
    real Ksave1;
    cal32x2(s0, s1, 28, 29);
    Ksave1 = *Kptr; Kptr += 32;
    L0 += Ksave0 * sM[s0][30];
    L1 += Ksave0 * sM[s1][30];
    L0 += Ksave1 * sM[s0][31];
    L1 += Ksave1 * sM[s1][31];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
#elif (SI_UNROLLING_R32 == 2) // Prefetch K-data and unroll the column loop twice
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real Ksave0 = *Kptr; Kptr += 32;
    real L0 = ZERO;
    real L1 = ZERO;
    for (int j = 0; j < 30; j += 2) {
      real Ksave1;
      cal32x2(s0, s1, j    , j + 1);
    }
    real Ksave1;
    Ksave1 = *Kptr; Kptr += 32;
    L0 += Ksave0 * sM[s0][30];
    L1 += Ksave0 * sM[s1][30];
    L0 += Ksave1 * sM[s0][31];
    L1 += Ksave1 * sM[s1][31];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
#elif (SI_UNROLLING_R32 == 1) // unoptimised version
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real L0 = ZERO;
    real L1 = ZERO;
    for (int j = 0; j < 32; j ++) {
      real Ksave = *Kptr; Kptr += 32;
      L0 += Ksave * sM[s0][j];
      L1 += Ksave * sM[s1][j];
    }
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
#else
#error Invalid parameter.
#endif
  }
}

#define cal32x4(s0, s1, s2, s3, jp0, jp1)	\
  Ksave1 = *Kptr; Kptr += 32;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  L1 += Ksave0 * sM[s1][jp0];			\
  L2 += Ksave0 * sM[s2][jp0];			\
  L3 += Ksave0 * sM[s3][jp0];			\
  Ksave0 = *Kptr; Kptr += 32;			\
  L0 += Ksave1 * sM[s0][jp1];			\
  L1 += Ksave1 * sM[s1][jp1];			\
  L2 += Ksave1 * sM[s2][jp1];			\
  L3 += Ksave1 * sM[s3][jp1]

__device__ void SI_comp32x4(real *K, int Kindex[32], real *Lptr, real sM[4][32], int kind, int f0, int f1, int f2, int f3, int s0, int s1, int s2, int s3)
{
  if (Kindex[kind] != NULL_KINDEX) {
#if (SI_UNROLLING_R32 == 32) // Prefetch K-data and unroll the column loop 32 times
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real Ksave0 = *Kptr; Kptr += 32;
    real L0 = ZERO;
    real L1 = ZERO;
    real L2 = ZERO;
    real L3 = ZERO;
    real Ksave1;
    cal32x4(s0, s1, s2, s3,  0,  1);
    cal32x4(s0, s1, s2, s3,  2,  3);
    cal32x4(s0, s1, s2, s3,  4,  5);
    cal32x4(s0, s1, s2, s3,  6,  7);
    cal32x4(s0, s1, s2, s3,  8,  9);
    cal32x4(s0, s1, s2, s3, 10, 11);
    cal32x4(s0, s1, s2, s3, 12, 13);
    cal32x4(s0, s1, s2, s3, 14, 15);
    cal32x4(s0, s1, s2, s3, 16, 17);
    cal32x4(s0, s1, s2, s3, 18, 19);
    cal32x4(s0, s1, s2, s3, 20, 21);
    cal32x4(s0, s1, s2, s3, 22, 23);
    cal32x4(s0, s1, s2, s3, 24, 25);
    cal32x4(s0, s1, s2, s3, 26, 27);
    cal32x4(s0, s1, s2, s3, 28, 29);
    Ksave1 = *Kptr; Kptr += 32;
    L0 += Ksave0 * sM[s0][30];
    L1 += Ksave0 * sM[s1][30];
    L2 += Ksave0 * sM[s2][30];
    L3 += Ksave0 * sM[s3][30];
    L0 += Ksave1 * sM[s0][31];
    L1 += Ksave1 * sM[s1][31];
    L2 += Ksave1 * sM[s2][31];
    L3 += Ksave1 * sM[s3][31];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
    *(Lptr + f2) += L2;
    *(Lptr + f3) += L3;
#elif (SI_UNROLLING_R32 == 4) // Prefetch K-data and unroll the column loop four times
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real Ksave0 = *Kptr; Kptr += 32;
    real L0 = ZERO;
    real L1 = ZERO;
    real L2 = ZERO;
    real L3 = ZERO;
    for (int j = 0; j < 28; j += 4) {
      real Ksave1;
      cal32x4(s0, s1, s2, s3, j    , j + 1);
      cal32x4(s0, s1, s2, s3, j + 2, j + 3);
    }
    real Ksave1;
    cal32x4(s0, s1, s2, s3, 28, 29);
    Ksave1 = *Kptr; Kptr += 32;
    L0 += Ksave0 * sM[s0][30];
    L1 += Ksave0 * sM[s1][30];
    L2 += Ksave0 * sM[s2][30];
    L3 += Ksave0 * sM[s3][30];
    L0 += Ksave1 * sM[s0][31];
    L1 += Ksave1 * sM[s1][31];
    L2 += Ksave1 * sM[s2][31];
    L3 += Ksave1 * sM[s3][31];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
    *(Lptr + f2) += L2;
    *(Lptr + f3) += L3;
#elif (SI_UNROLLING_R32 == 2) // Prefetch K-data and unroll the column loop twice
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real Ksave0 = *Kptr; Kptr += 32;
    real L0 = ZERO;
    real L1 = ZERO;
    real L2 = ZERO;
    real L3 = ZERO;
    for (int j = 0; j < 30; j += 2) {
      real Ksave1;
      cal32x4(s0, s1, s2, s3, j    , j + 1);
    }
    real Ksave1;
    Ksave1 = *Kptr; Kptr += 32;
    L0 += Ksave0 * sM[s0][30];
    L1 += Ksave0 * sM[s1][30];
    L2 += Ksave0 * sM[s2][30];
    L3 += Ksave0 * sM[s3][30];
    L0 += Ksave1 * sM[s0][31];
    L1 += Ksave1 * sM[s1][31];
    L2 += Ksave1 * sM[s2][31];
    L3 += Ksave1 * sM[s3][31];
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
    *(Lptr + f2) += L2;
    *(Lptr + f3) += L3;
#elif (SI_UNROLLING_R32 == 1) // unoptimised version
    real *Kptr = K + (Kindex[kind] << 10) + tx;
    real L0 = ZERO;
    real L1 = ZERO;
    real L2 = ZERO;
    real L3 = ZERO;
    for (int j = 0; j < 32; j ++) {
      real Ksave = *Kptr; Kptr += 32;
      L0 += Ksave * sM[s0][j];
      L1 += Ksave * sM[s1][j];
      L2 += Ksave * sM[s2][j];
      L3 += Ksave * sM[s3][j];
    }
    *(Lptr + f0) += L0;
    *(Lptr + f1) += L1;
    *(Lptr + f2) += L2;
    *(Lptr + f3) += L3;
#else
#error Invalid parameter.
#endif
  }
}

#define cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, jp0, jp1)	\
  Ksave1 = *Kptr; Kptr += 32;					\
  L0 += Ksave0 * sM[s0][jp0];					\
  L1 += Ksave0 * sM[s1][jp0];					\
  L2 += Ksave0 * sM[s2][jp0];					\
  L3 += Ksave0 * sM[s3][jp0];					\
  L4 += Ksave0 * sM[s4][jp0];					\
  L5 += Ksave0 * sM[s5][jp0];					\
  L6 += Ksave0 * sM[s6][jp0];					\
  L7 += Ksave0 * sM[s7][jp0];					\
  Ksave0 = *Kptr; Kptr += 32;					\
  L0 += Ksave1 * sM[s0][jp1];					\
  L1 += Ksave1 * sM[s1][jp1];					\
  L2 += Ksave1 * sM[s2][jp1];					\
  L3 += Ksave1 * sM[s3][jp1];					\
  L4 += Ksave1 * sM[s4][jp1];					\
  L5 += Ksave1 * sM[s5][jp1];					\
  L6 += Ksave1 * sM[s6][jp1];					\
  L7 += Ksave1 * sM[s7][jp1]

__device__ void SI_comp32x8(real *K, int Kindex[32], real *Lptr, real sM[4][32], int kind, int f0, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7)
{
#if (SI_UNROLLING_R32 == 32) // Prefetch K-data and unroll the column loop 32 times
  real *Kptr = K + (Kindex[kind] << 10) + tx;
  real Ksave0 = *Kptr; Kptr += 32;
  real L0 = ZERO;
  real L1 = ZERO;
  real L2 = ZERO;
  real L3 = ZERO;
  real L4 = ZERO;
  real L5 = ZERO;
  real L6 = ZERO;
  real L7 = ZERO;
  real Ksave1;
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  0,  1);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  2,  3);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  4,  5);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  6,  7);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  8,  9);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 10, 11);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 12, 13);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 14, 15);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 16, 17);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 18, 19);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 20, 21);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 22, 23);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 24, 25);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 26, 27);
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 28, 29);
  Ksave1 = *Kptr; Kptr += 32;
  L0 += Ksave0 * sM[s0][30];
  L1 += Ksave0 * sM[s1][30];
  L2 += Ksave0 * sM[s2][30];
  L3 += Ksave0 * sM[s3][30];
  L4 += Ksave0 * sM[s4][30];
  L5 += Ksave0 * sM[s5][30];
  L6 += Ksave0 * sM[s6][30];
  L7 += Ksave0 * sM[s7][30];
  L0 += Ksave1 * sM[s0][31];
  L1 += Ksave1 * sM[s1][31];
  L2 += Ksave1 * sM[s2][31];
  L3 += Ksave1 * sM[s3][31];
  L4 += Ksave1 * sM[s4][31];
  L5 += Ksave1 * sM[s5][31];
  L6 += Ksave1 * sM[s6][31];
  L7 += Ksave1 * sM[s7][31];
  *(Lptr + f0) += L0;
  *(Lptr + f1) += L1;
  *(Lptr + f2) += L2;
  *(Lptr + f3) += L3;
  *(Lptr + f4) += L4;
  *(Lptr + f5) += L5;
  *(Lptr + f6) += L6;
  *(Lptr + f7) += L7;
#elif (SI_UNROLLING_R32 == 4) // Prefetch K-data and unroll the column loop four times
  real *Kptr = K + (Kindex[kind] << 10) + tx;
  real Ksave0 = *Kptr; Kptr += 32;
  real L0 = ZERO;
  real L1 = ZERO;
  real L2 = ZERO;
  real L3 = ZERO;
  real L4 = ZERO;
  real L5 = ZERO;
  real L6 = ZERO;
  real L7 = ZERO;
  for (int j = 0; j < 28; j += 4) {
    real Ksave1;
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, j    , j + 1);
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, j + 2, j + 3);
  }
  real Ksave1;
  cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 28, 29);
  Ksave1 = *Kptr; Kptr += 32;
  L0 += Ksave0 * sM[s0][30];
  L1 += Ksave0 * sM[s1][30];
  L2 += Ksave0 * sM[s2][30];
  L3 += Ksave0 * sM[s3][30];
  L4 += Ksave0 * sM[s4][30];
  L5 += Ksave0 * sM[s5][30];
  L6 += Ksave0 * sM[s6][30];
  L7 += Ksave0 * sM[s7][30];
  L0 += Ksave1 * sM[s0][31];
  L1 += Ksave1 * sM[s1][31];
  L2 += Ksave1 * sM[s2][31];
  L3 += Ksave1 * sM[s3][31];
  L4 += Ksave1 * sM[s4][31];
  L5 += Ksave1 * sM[s5][31];
  L6 += Ksave1 * sM[s6][31];
  L7 += Ksave1 * sM[s7][31];
  *(Lptr + f0) += L0;
  *(Lptr + f1) += L1;
  *(Lptr + f2) += L2;
  *(Lptr + f3) += L3;
  *(Lptr + f4) += L4;
  *(Lptr + f5) += L5;
  *(Lptr + f6) += L6;
  *(Lptr + f7) += L7;
#elif (SI_UNROLLING_R32 == 2) // Prefetch K-data and unroll the column loop twice
  real *Kptr = K + (Kindex[kind] << 10) + tx;
  real Ksave0 = *Kptr; Kptr += 32;
  real L0 = ZERO;
  real L1 = ZERO;
  real L2 = ZERO;
  real L3 = ZERO;
  real L4 = ZERO;
  real L5 = ZERO;
  real L6 = ZERO;
  real L7 = ZERO;
  for (int j = 0; j < 30; j += 2) {
    real Ksave1;
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, j    , j + 1);
  }
  real Ksave1;
  Ksave1 = *Kptr; Kptr += 32;
  L0 += Ksave0 * sM[s0][30];
  L1 += Ksave0 * sM[s1][30];
  L2 += Ksave0 * sM[s2][30];
  L3 += Ksave0 * sM[s3][30];
  L4 += Ksave0 * sM[s4][30];
  L5 += Ksave0 * sM[s5][30];
  L6 += Ksave0 * sM[s6][30];
  L7 += Ksave0 * sM[s7][30];
  L0 += Ksave1 * sM[s0][31];
  L1 += Ksave1 * sM[s1][31];
  L2 += Ksave1 * sM[s2][31];
  L3 += Ksave1 * sM[s3][31];
  L4 += Ksave1 * sM[s4][31];
  L5 += Ksave1 * sM[s5][31];
  L6 += Ksave1 * sM[s6][31];
  L7 += Ksave1 * sM[s7][31];
  *(Lptr + f0) += L0;
  *(Lptr + f1) += L1;
  *(Lptr + f2) += L2;
  *(Lptr + f3) += L3;
  *(Lptr + f4) += L4;
  *(Lptr + f5) += L5;
  *(Lptr + f6) += L6;
  *(Lptr + f7) += L7;
#elif (SI_UNROLLING_R32 == 1) // unoptimised version
  real *Kptr = K + (Kindex[kind] << 10) + tx;
  real L0 = ZERO;
  real L1 = ZERO;
  real L2 = ZERO;
  real L3 = ZERO;
  real L4 = ZERO;
  real L5 = ZERO;
  real L6 = ZERO;
  real L7 = ZERO;
  for (int j = 0; j < 32; j ++) {
    real Ksave = *Kptr; Kptr += 32;
    L0 += Ksave * sM[s0][j];
    L1 += Ksave * sM[s1][j];
    L2 += Ksave * sM[s2][j];
    L3 += Ksave * sM[s3][j];
    L4 += Ksave * sM[s4][j];
    L5 += Ksave * sM[s5][j];
    L6 += Ksave * sM[s6][j];
    L7 += Ksave * sM[s7][j];
  }
  *(Lptr + f0) += L0;
  *(Lptr + f1) += L1;
  *(Lptr + f2) += L2;
  *(Lptr + f3) += L3;
  *(Lptr + f4) += L4;
  *(Lptr + f5) += L5;
  *(Lptr + f6) += L6;
  *(Lptr + f7) += L7;
#else
#error Invalid parameter.
#endif
}

#define II0 0
#define II1 32
#define II2 64
#define II3 96
#define II4 128
#define II5 160
#define II6 192
#define II7 224

__global__ void m2l_kern_sibling_blocking_r32(real *L, real *K, real *M, int *sourceclusters, int *Ktable, int FCsta)
{
  /* Set the pointer to L for the 0th sibling in the current
     observation cluster OC, to which bx-th thread-block is
     assigned */
  real *Lptr = L + (bx << 8) + tx; // log2(8*r)=8
  
  /* Load 0th siblings of 27 source clusters (SCs) for OC;
     ilist[27:31] are unused */
  __shared__ int ilist[32]; // 27 or more; 128 bytes
  ilist[tx] = sourceclusters[((FCsta + bx) << 5) + tx]; // log2(32)=5

  /* Copy Ktable in shared memory */
  __shared__ int sKtable[343]; // 1372 bytes
  sKtable[tx      ] = Ktable[tx      ];
  sKtable[tx +  32] = Ktable[tx +  32];
  sKtable[tx +  64] = Ktable[tx +  64];
  sKtable[tx +  96] = Ktable[tx +  96];
  sKtable[tx + 128] = Ktable[tx + 128];
  sKtable[tx + 160] = Ktable[tx + 160];
  sKtable[tx + 192] = Ktable[tx + 192];
  sKtable[tx + 224] = Ktable[tx + 224];
  sKtable[tx + 256] = Ktable[tx + 256];
  sKtable[tx + 288] = Ktable[tx + 288];
  if (tx < 23) {
    sKtable[tx + 320] = Ktable[tx + 320];
  }
  
  /* Load Kindex0[] and Kindex1[] */
  __shared__ int Kindex0[32], Kindex1[32], Kindex[32]; // 27 or more; 384 bytes
  load_Kindex0_Kindex1(tx);

  /* Ensure that ilist[], sKtable[], Kindex0[], and Kindex1[] were loaded */
  __syncthreads();

  /* Loop over source-cluster indices */
  for (int d = 0; d < 27; d ++) {
    
    /* If the 0th sibling, which is the representative of SC, does not
       exist in the hierachy (then, the other seven siblings in SC
       neither exist) or SC coincides with OC, we can skip the
       computaion for SC */
    if (ilist[d] != NULL_CELL) {

      /* Load K-indices for all the 27 interaction-kinds */
      Kindex[tx] = sKtable[Kindex0[d] + Kindex1[tx]];

      /* Ensure that Kindex[] was loaded */
      __syncthreads();
      
      /* Allocate the memory for all eight M-vectors in SC */
      __shared__ real sM[8][32]; // 1024 bytes in single

      {
	real *Mptr = M + (ilist[d] << 5) + tx; // log2(r)=5
	sM[0][tx] = *(Mptr + II0); /* S0 */
	sM[1][tx] = *(Mptr + II1); /* S1 */
	sM[2][tx] = *(Mptr + II2); /* S2 */
	sM[3][tx] = *(Mptr + II3); /* S3 */
	sM[4][tx] = *(Mptr + II4); /* S4 */
	sM[5][tx] = *(Mptr + II5); /* S5 */
	sM[6][tx] = *(Mptr + II6); /* S6 */
	sM[7][tx] = *(Mptr + II7); /* S7 */
      }
      __syncthreads();

      SI_comp32x1(K, Kindex, Lptr, sM, F0S7, II0, J7);
      SI_comp32x1(K, Kindex, Lptr, sM, F1S6, II1, J6);
      SI_comp32x1(K, Kindex, Lptr, sM, F2S5, II2, J5);
      SI_comp32x1(K, Kindex, Lptr, sM, F3S4, II3, J4);
      SI_comp32x1(K, Kindex, Lptr, sM, F4S3, II4, J3);
      SI_comp32x1(K, Kindex, Lptr, sM, F5S2, II5, J2);
      SI_comp32x1(K, Kindex, Lptr, sM, F6S1, II6, J1);
      SI_comp32x1(K, Kindex, Lptr, sM, F7S0, II7, J0);
      SI_comp32x2(K, Kindex, Lptr, sM, F0S3, II0, II4, J3, J7);
      SI_comp32x2(K, Kindex, Lptr, sM, F0S5, II0, II2, J5, J7);
      SI_comp32x2(K, Kindex, Lptr, sM, F0S6, II0, II1, J6, J7);
      SI_comp32x2(K, Kindex, Lptr, sM, F1S2, II1, II5, J2, J6);
      SI_comp32x2(K, Kindex, Lptr, sM, F1S4, II1, II3, J4, J6);
      SI_comp32x2(K, Kindex, Lptr, sM, F2S1, II2, II6, J1, J5);
      SI_comp32x2(K, Kindex, Lptr, sM, F2S4, II2, II3, J4, J5);
      SI_comp32x2(K, Kindex, Lptr, sM, F3S0, II3, II7, J0, J4);
      SI_comp32x2(K, Kindex, Lptr, sM, F4S1, II4, II6, J1, J3);
      SI_comp32x2(K, Kindex, Lptr, sM, F4S2, II4, II5, J2, J3);
      SI_comp32x2(K, Kindex, Lptr, sM, F5S0, II5, II7, J0, J2);
      SI_comp32x2(K, Kindex, Lptr, sM, F6S0, II6, II7, J0, J1);
      SI_comp32x4(K, Kindex, Lptr, sM, F0S4, II0, II1, II2, II3, J4, J5, J6, J7);
      SI_comp32x4(K, Kindex, Lptr, sM, F0S2, II0, II1, II4, II5, J2, J3, J6, J7);
      SI_comp32x4(K, Kindex, Lptr, sM, F0S1, II0, II2, II4, II6, J1, J3, J5, J7);
      SI_comp32x4(K, Kindex, Lptr, sM, F1S0, II1, II3, II5, II7, J0, J2, J4, J6);
      SI_comp32x4(K, Kindex, Lptr, sM, F2S0, II2, II3, II6, II7, J0, J1, J4, J5);
      SI_comp32x4(K, Kindex, Lptr, sM, F4S0, II4, II5, II6, II7, J0, J1, J2, J3);
      SI_comp32x8(K, Kindex, Lptr, sM, F0S0, II0, II1, II2, II3, II4, II5, II6, II7, J0, J1, J2, J3, J4, J5, J6, J7);

      __syncthreads();
    }
  }
}
/**************************************************************************/
#elif defined(CUDA_VER45A)
/**************************************************************************/
#error This version does not work. Never use this.

#include "real.h"

#define bx blockIdx.x
#define tx threadIdx.x

#ifndef NULLCELL
#define NULLCELL - 1
#endif

#ifndef NULL_CELL
#define NULL_CELL NULLCELL
#endif

#ifndef NULL_KINDEX
#define NULL_KINDEX - 1
#endif

/* Define 27 interction-kinds. See Table 4 (left) in the paper */
#define F0S7  0 /* F0 vs S7 */
#define F1S6  1 /* F1 vs S6 */
#define F2S5  2 /* F2 vs S5 */
#define F3S4  3 /* F3 vs S4 */
#define F4S3  4 /* F4 vs S3 */
#define F5S2  5 /* F5 vs S2 */
#define F6S1  6 /* F6 vs S1 */
#define F7S0  7 /* F7 vs S0 */
#define F0S6  8 /* F0,F1 vs S6,S7 */
#define F0S5  9 /* F0,F2 vs S5,S7 */
#define F0S3 10 /* F0,F4 vs S3,S7 */
#define F1S4 11 /* F1,F3 vs S4,S6 */
#define F1S2 12 /* F1,F5 vs S2,S6 */
#define F2S4 13 /* F2,F3 vs S4,S5 */
#define F2S1 14 /* F2,F6 vs S1,S5 */
#define F3S0 15 /* F3,F7 vs S0,S4 */
#define F4S2 16 /* F4,F5 vs S2,S3 */
#define F4S1 17 /* F4,F6 vs S1,S3 */
#define F5S0 18 /* F5,F7 vs S0,S2 */
#define F6S0 19 /* F6,F7 vs S0,S1 */
#define F0S4 20 /* F0,F1,F2,F3 vs S4,S5,S6,S7 */
#define F0S2 21 /* F0,F1,F4,F5 vs S2,S3,S6,S7 */
#define F0S1 22 /* F0,F2,F4,F6 vs S1,S3,S5,S7 */
#define F1S0 23 /* F1,F3,F5,F7 vs S0,S2,S4,S6 */
#define F2S0 24 /* F2,F3,F6,F7 vs S0,S1,S4,S5 */
#define F4S0 25 /* F4,F5,F6,F7 vs S0,S1,S2,S3 */
#define F0S0 26 /* F0,F1,F2,F3,F4,F5,F6,F7 vs S0,S1,S2,S3,S4,S5,S6,S7 */


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

#define I0 0
#define I1 256
#define I2 512
#define I3 768
#define I4 1024
#define I5 1280
#define I6 1536
#define I7 1792
#define J0 0
#define J1 1
#define J2 2
#define J3 3
#define J4 4
#define J5 5
#define J6 6
#define J7 7


#define cal256x1(s0, jp0, jp1)			\
  Ksave1 = *Kptr; Kptr += 256;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  Ksave0 = *Kptr; Kptr += 256;			\
  L0 += Ksave1 * sM[s0][jp1]

/* Prefetch K-data and unroll the column loop four times */
#define SI_comp256x1(kind, f0, s0)				\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /*110228    float *Kptr = K + (Kindex[kind] << 16) + tx;*/	\
    /*110228    float Ksave0 = *Kptr; Kptr += 256;*/		\
    /*110228    float L0 = 0.0f;*/				\
    real *Kptr = K + (Kindex[kind] << 16) + tx;			\
    real Ksave0 = *Kptr; Kptr += 256;				\
    real L0 = ZERO;						\
    for (int j = 0; j < 252; j += 4) {				\
      /*110228      float Ksave1;*/				\
      real Ksave1;						\
      cal256x1(s0, j    , j + 1);				\
      cal256x1(s0, j + 2, j + 3);				\
    }								\
    /*110228    float Ksave1;*/					\
    real Ksave1;						\
    cal256x1(s0, 252, 253);					\
    Ksave1 = *Kptr; Kptr += 256;				\
    L0 += Ksave0 * sM[s0][254];					\
    L0 += Ksave1 * sM[s0][255];					\
    *(Lptr + f0) += L0;						\
  }

#define cal256x2(s0, s1, jp0, jp1)		\
  Ksave1 = *Kptr; Kptr += 256;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  L1 += Ksave0 * sM[s1][jp0];			\
  Ksave0 = *Kptr; Kptr += 256;			\
  L0 += Ksave1 * sM[s0][jp1];			\
  L1 += Ksave1 * sM[s1][jp1]

/* Prefetch K-data and unroll column loop 32 times */
#define SI_comp256x2(kind, f0, f1, s0, s1)			\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /*110228    float *Kptr = K + (Kindex[kind] << 16) + tx;*/	\
    /*110228    float Ksave0 = *Kptr; Kptr += 256;*/		\
    /*110228    float L0 = 0.0f;*/				\
    /*110228    float L1 = 0.0f*/;				\
    real *Kptr = K + (Kindex[kind] << 16) + tx;			\
    real Ksave0 = *Kptr; Kptr += 256;				\
    real L0 = ZERO;						\
    real L1 = ZERO;						\
    for (int j = 0; j < 224; j += 32) {				\
      /*110228      float Ksave1;*/				\
      real Ksave1;						\
      cal256x2(s0, s1, j     , j +  1);				\
      cal256x2(s0, s1, j +  2, j +  3);				\
      cal256x2(s0, s1, j +  4, j +  5);				\
      cal256x2(s0, s1, j +  6, j +  7);				\
      cal256x2(s0, s1, j +  8, j +  9);				\
      cal256x2(s0, s1, j + 10, j + 11);				\
      cal256x2(s0, s1, j + 12, j + 13);				\
      cal256x2(s0, s1, j + 14, j + 15);				\
      cal256x2(s0, s1, j + 16, j + 17);				\
      cal256x2(s0, s1, j + 18, j + 19);				\
      cal256x2(s0, s1, j + 20, j + 21);				\
      cal256x2(s0, s1, j + 22, j + 23);				\
      cal256x2(s0, s1, j + 24, j + 25);				\
      cal256x2(s0, s1, j + 26, j + 27);				\
      cal256x2(s0, s1, j + 28, j + 29);				\
      cal256x2(s0, s1, j + 30, j + 31);				\
    }								\
    /*110228    float Ksave1;*/					\
    real Ksave1;						\
    cal256x2(s0, s1, 224, 225);					\
    cal256x2(s0, s1, 226, 227);					\
    cal256x2(s0, s1, 228, 229);					\
    cal256x2(s0, s1, 230, 231);					\
    cal256x2(s0, s1, 232, 233);					\
    cal256x2(s0, s1, 234, 235);					\
    cal256x2(s0, s1, 236, 237);					\
    cal256x2(s0, s1, 238, 239);					\
    cal256x2(s0, s1, 240, 241);					\
    cal256x2(s0, s1, 242, 243);					\
    cal256x2(s0, s1, 244, 245);					\
    cal256x2(s0, s1, 246, 247);					\
    cal256x2(s0, s1, 248, 249);					\
    cal256x2(s0, s1, 250, 251);					\
    cal256x2(s0, s1, 252, 253);					\
    Ksave1 = *Kptr; Kptr += 256;				\
    L0 += Ksave0 * sM[s0][254];					\
    L1 += Ksave0 * sM[s1][254];					\
    L0 += Ksave1 * sM[s0][255];					\
    L1 += Ksave1 * sM[s1][255];					\
    *(Lptr + f0) += L0;						\
    *(Lptr + f1) += L1;						\
  }

#define cal256x4(s0, s1, s2, s3, jp0, jp1)	\
  Ksave1 = *Kptr; Kptr += 256;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  L1 += Ksave0 * sM[s1][jp0];			\
  L2 += Ksave0 * sM[s2][jp0];			\
  L3 += Ksave0 * sM[s3][jp0];			\
  Ksave0 = *Kptr; Kptr += 256;			\
  L0 += Ksave1 * sM[s0][jp1];			\
  L1 += Ksave1 * sM[s1][jp1];			\
  L2 += Ksave1 * sM[s2][jp1];			\
  L3 += Ksave1 * sM[s3][jp1]

/* Prefetch K-data and unroll column loop 32 times */
#define SI_comp256x4(kind, f0, f1, f2, f3, s0, s1, s2, s3)	\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /*110228    float *Kptr = K + (Kindex[kind] << 16) + tx;*/	\
    /*110228    float Ksave0 = *Kptr; Kptr += 256;*/		\
    /*110228    float L0 = 0.0f;*/				\
    /*110228    float L1 = 0.0f;*/				\
    /*110228    float L2 = 0.0f;*/				\
    /*110228    float L3 = 0.0f;*/				\
    real *Kptr = K + (Kindex[kind] << 16) + tx;			\
    real Ksave0 = *Kptr; Kptr += 256;				\
    real L0 = ZERO;						\
    real L1 = ZERO;						\
    real L2 = ZERO;						\
    real L3 = ZERO;						\
    for (int j = 0; j < 224; j += 32) {				\
      /*110228      float Ksave1;*/				\
      real Ksave1;						\
      cal256x4(s0, s1, s2, s3, j     , j +  1);			\
      cal256x4(s0, s1, s2, s3, j +  2, j +  3);			\
      cal256x4(s0, s1, s2, s3, j +  4, j +  5);			\
      cal256x4(s0, s1, s2, s3, j +  6, j +  7);			\
      cal256x4(s0, s1, s2, s3, j +  8, j +  9);			\
      cal256x4(s0, s1, s2, s3, j + 10, j + 11);			\
      cal256x4(s0, s1, s2, s3, j + 12, j + 13);			\
      cal256x4(s0, s1, s2, s3, j + 14, j + 15);			\
      cal256x4(s0, s1, s2, s3, j + 16, j + 17);			\
      cal256x4(s0, s1, s2, s3, j + 18, j + 19);			\
      cal256x4(s0, s1, s2, s3, j + 20, j + 21);			\
      cal256x4(s0, s1, s2, s3, j + 22, j + 23);			\
      cal256x4(s0, s1, s2, s3, j + 24, j + 25);			\
      cal256x4(s0, s1, s2, s3, j + 26, j + 27);			\
      cal256x4(s0, s1, s2, s3, j + 28, j + 29);			\
      cal256x4(s0, s1, s2, s3, j + 30, j + 31);			\
    }								\
    /*110228    float Ksave1;*/					\
    real Ksave1;						\
    cal256x4(s0, s1, s2, s3, 224, 225);				\
    cal256x4(s0, s1, s2, s3, 226, 227);				\
    cal256x4(s0, s1, s2, s3, 228, 229);				\
    cal256x4(s0, s1, s2, s3, 230, 231);				\
    cal256x4(s0, s1, s2, s3, 232, 233);				\
    cal256x4(s0, s1, s2, s3, 234, 235);				\
    cal256x4(s0, s1, s2, s3, 236, 237);				\
    cal256x4(s0, s1, s2, s3, 238, 239);				\
    cal256x4(s0, s1, s2, s3, 240, 241);				\
    cal256x4(s0, s1, s2, s3, 242, 243);				\
    cal256x4(s0, s1, s2, s3, 244, 245);				\
    cal256x4(s0, s1, s2, s3, 246, 247);				\
    cal256x4(s0, s1, s2, s3, 248, 249);				\
    cal256x4(s0, s1, s2, s3, 250, 251);				\
    cal256x4(s0, s1, s2, s3, 252, 253);				\
    Ksave1 = *Kptr; Kptr += 256;				\
    L0 += Ksave0 * sM[s0][254];					\
    L1 += Ksave0 * sM[s1][254];					\
    L2 += Ksave0 * sM[s2][254];					\
    L3 += Ksave0 * sM[s3][254];					\
    L0 += Ksave1 * sM[s0][255];					\
    L1 += Ksave1 * sM[s1][255];					\
    L2 += Ksave1 * sM[s2][255];					\
    L3 += Ksave1 * sM[s3][255];					\
    *(Lptr + f0) += L0;						\
    *(Lptr + f1) += L1;						\
    *(Lptr + f2) += L2;						\
    *(Lptr + f3) += L3;						\
  }

#define SI_comp256x8(kind, f0, f1, f2, f3, s0, s1, s2, s3)	\
  SI_comp256x4(kind, f0, f1, f2, f3, s0, s1, s2, s3)


//110228__global__ void m2l_kern_sibling_blocking_r256(float *L, float *K, float *M, int *sourceclusters, int *Ktable, int FCsta)
__global__ void m2l_kern_sibling_blocking_r256(real *L, real *K, real *M, int *sourceclusters, int *Ktable, int FCsta)
{

  /* Set the pointer to L for the 0th sibling in the current
     observation cluster OC, to which bx-th thread-block is
     assigned */
  //110228  float *Lptr = L + (bx << 11) + tx; // log2(r*8)=11
  real *Lptr = L + (bx << 11) + tx; // log2(r*8)=11
  
  /* Load 0th siblings of 27 source clusters (SCs) for OC;
     ilist[27:31] are unused */
  __shared__ int ilist[32]; // 27 or more; 128 bytes
  if (tx < 32) {
    ilist[tx] = sourceclusters[((FCsta + bx) << 5) + tx]; // log2(32)=5
  }

  /* Copy Ktable in shared memory */
  __shared__ int sKtable[343]; // 1372 bytes
  sKtable[tx] = Ktable[tx];
  if (tx < 87) {
    sKtable[tx + 256] = Ktable[tx + 256];
  }
  
  /* Load Kindex0[] and Kindex1[] */
  __shared__ int Kindex0[32], Kindex1[32], Kindex[32]; // 27 or more; 384 bytes
  if (tx < 32) {
    load_Kindex0_Kindex1(tx);
  }

  /* Ensure that ilist[], sKtable[], Kindex0[], and Kindex1[] were loaded */
  __syncthreads();
  
  /* Loop over source-cluster indices */
  for (int d = 0; d < 27; d ++) {
    
    /* If the 0th sibling, which is the representative of SC, does not
       exist in the hierachy (then, the other seven siblings in SC
       neither exist) or SC coincides with OC, we can skip the
       computaion for SC */
    if (ilist[d] != NULL_CELL) {

      /* Load K-indices for all the 27 interaction-kinds */
      if (tx < 32) {
	Kindex[tx] = sKtable[Kindex0[d] + Kindex1[tx]];
      }
      
      /* Ensure that Kindex[] was loaded */
      __syncthreads();
      
      /* Allocate the memory for ONLY FOUR M-vectors in SC because of
	 the shortage of shared memory */
      //110228      __shared__ float sM[4][256]; // 4096 bytes
      __shared__ real sM[4][256]; // 4096 bytes for single

      {
	//110228	float *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[0][tx] = *(Mptr + I0); /* S0 */
	sM[1][tx] = *(Mptr + I1); /* S1 */
	sM[2][tx] = *(Mptr + I2); /* S2 */
	sM[3][tx] = *(Mptr + I3); /* S3 */
      } /* sM[]={S0,S1,S2,S3} */
      __syncthreads();
    
      SI_comp256x1(F4S3, I4, J3);
      SI_comp256x1(F5S2, I5, J2);
      SI_comp256x1(F6S1, I6, J1);
      SI_comp256x1(F7S0, I7, J0);
      SI_comp256x2(F5S0, I5, I7, J0, J2);
      SI_comp256x2(F6S0, I6, I7, J0, J1);
      SI_comp256x4(F4S0, I4, I5, I6, I7, J0, J1, J2, J3);
      SI_comp256x8(F0S0, I0, I1, I2, I3, J0, J1, J2, J3);
      __syncthreads();
      
      {
	//110228	float *Mptr = M + (ilist[d] << 8) + tx;
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[1][tx] = *(Mptr + I4); /* S4 */
	sM[3][tx] = *(Mptr + I6); /* S6 */
      } /* sM[]={S0,S4,S2,S6} */
      __syncthreads();
      
      SI_comp256x2(F1S4, I1, I3, J1, J3);
      SI_comp256x2(F1S2, I1, I5, J2, J3);
      SI_comp256x4(F1S0, I1, I3, I5, I7, J0, J2, J1, J3);
      __syncthreads();
      
      {
	//110228	float *Mptr = M + (ilist[d] << 8) + tx;
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[0][tx] = *(Mptr + I5); /* S5 */
	sM[2][tx] = *(Mptr + I7); /* S7 */
      } /* sM[]={S5,S4,S7,S6} */
      __syncthreads();
      
      SI_comp256x1(F0S7, I0, J2);
      SI_comp256x1(F1S6, I1, J3);
      SI_comp256x1(F2S5, I2, J0);
      SI_comp256x1(F3S4, I3, J1);
      SI_comp256x2(F0S5, I0, I2, J0, J2);
      SI_comp256x2(F2S4, I2, I3, J1, J0);
      SI_comp256x4(F0S4, I0, I1, I2, I3, J1, J0, J3, J2);
      SI_comp256x8(F0S0, I4, I5, I6, I7, J1, J0, J3, J2);
      __syncthreads();
      
      {
	//110228	float *Mptr = M + (ilist[d] << 8) + tx;
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[0][tx] = *(Mptr + I2); /* S2 */
	sM[1][tx] = *(Mptr + I3); /* S3 */
      } /* sM[]={S2,S3,S7,S6} */
      __syncthreads();
      
      SI_comp256x2(F0S6, I0, I1, J3, J2);
      SI_comp256x2(F4S2, I4, I5, J0, J1);
      SI_comp256x4(F0S2, I0, I1, I4, I5, J0, J1, J3, J2);
      __syncthreads();
      
      {
	//110228	float *Mptr = M + (ilist[d] << 8) + tx;
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[0][tx] = *(Mptr + I1); /* S1 */
	sM[3][tx] = *(Mptr + I5); /* S5 */
      } /* sM[]={S1,S3,S7,S5} */
      __syncthreads();
      
      SI_comp256x2(F0S3, I0, I4, J1, J2);
      SI_comp256x2(F4S1, I4, I6, J0, J1);
      SI_comp256x4(F0S1, I0, I2, I4, I6, J0, J1, J3, J2);
      __syncthreads();
      
      {
	//110228	float *Mptr = M + (ilist[d] << 8) + tx;
	real *Mptr = M + (ilist[d] << 8) + tx; // log2(r)=8
	sM[1][tx] = *(Mptr + I0); /* S0 */
	sM[2][tx] = *(Mptr + I4); /* S4 */
      } /* sM[]={S1,S0,S4,S5} */
      __syncthreads();
      
      SI_comp256x2(F2S1, I2, I6, J0, J3);
      SI_comp256x2(F3S0, I3, I7, J1, J2);
      SI_comp256x4(F2S0, I2, I3, I6, I7, J1, J0, J2, J3);
      __syncthreads();
    }
  }
}

#define cal32x1(s0, jp0, jp1)			\
  Ksave1 = *Kptr; Kptr += 32;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  Ksave0 = *Kptr; Kptr += 32;			\
  L0 += Ksave1 * sM[s0][jp1]

/* Prefetch K-data and unroll column loop completely */
#define SI_comp32x1(kind, f0, s0)				\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /*110228    float *Kptr = K + (Kindex[kind] << 10) + tx;*/	\
    /*110228    float Ksave0 = *Kptr; Kptr += 32;*/		\
    /*110228    float L0 = 0.0f;*/				\
    /*110228    float Ksave1;*/					\
    real *Kptr = K + (Kindex[kind] << 10) + tx;			\
    real Ksave0 = *Kptr; Kptr += 32;				\
    real L0 = ZERO;						\
    real Ksave1;						\
    cal32x1(s0,  0,  1);					\
    cal32x1(s0,  2,  3);					\
    cal32x1(s0,  4,  5);					\
    cal32x1(s0,  6,  7);					\
    cal32x1(s0,  8,  9);					\
    cal32x1(s0, 10, 11);					\
    cal32x1(s0, 12, 13);					\
    cal32x1(s0, 14, 15);					\
    cal32x1(s0, 16, 17);					\
    cal32x1(s0, 18, 19);					\
    cal32x1(s0, 20, 21);					\
    cal32x1(s0, 22, 23);					\
    cal32x1(s0, 24, 25);					\
    cal32x1(s0, 26, 27);					\
    cal32x1(s0, 28, 29);					\
    Ksave1 = *Kptr; Kptr += 32;					\
    L0 += Ksave0 * sM[s0][30];					\
    L0 += Ksave1 * sM[s0][31];					\
    *(Lptr + f0) += L0;						\
  }

#define cal32x2(s0, s1, jp0, jp1)		\
  Ksave1 = *Kptr; Kptr += 32;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  L1 += Ksave0 * sM[s1][jp0];			\
  Ksave0 = *Kptr; Kptr += 32;			\
  L0 += Ksave1 * sM[s0][jp1];			\
  L1 += Ksave1 * sM[s1][jp1]

/* Prefetch K-data and unroll column loop completely */
#define SI_comp32x2(kind, f0, f1, s0, s1)			\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /*110228    float *Kptr = K + (Kindex[kind] << 10) + tx;*/	\
    /*110228    float Ksave0 = *Kptr; Kptr += 32;*/		\
    /*110228    float L0 = 0.0f;*/				\
    /*110228    float L1 = 0.0f;*/				\
    /*110228    float Ksave1;*/					\
    real *Kptr = K + (Kindex[kind] << 10) + tx;			\
    real Ksave0 = *Kptr; Kptr += 32;				\
    real L0 = ZERO;						\
    real L1 = ZERO;						\
    real Ksave1;						\
    cal32x2(s0, s1,  0,  1);					\
    cal32x2(s0, s1,  2,  3);					\
    cal32x2(s0, s1,  4,  5);					\
    cal32x2(s0, s1,  6,  7);					\
    cal32x2(s0, s1,  8,  9);					\
    cal32x2(s0, s1, 10, 11);					\
    cal32x2(s0, s1, 12, 13);					\
    cal32x2(s0, s1, 14, 15);					\
    cal32x2(s0, s1, 16, 17);					\
    cal32x2(s0, s1, 18, 19);					\
    cal32x2(s0, s1, 20, 21);					\
    cal32x2(s0, s1, 22, 23);					\
    cal32x2(s0, s1, 24, 25);					\
    cal32x2(s0, s1, 26, 27);					\
    cal32x2(s0, s1, 28, 29);					\
    Ksave1 = *Kptr; Kptr += 32;					\
    L0 += Ksave0 * sM[s0][30];					\
    L1 += Ksave0 * sM[s1][30];					\
    L0 += Ksave1 * sM[s0][31];					\
    L1 += Ksave1 * sM[s1][31];					\
    *(Lptr + f0) += L0;						\
    *(Lptr + f1) += L1;						\
  }

#define cal32x4(s0, s1, s2, s3, jp0, jp1)	\
  Ksave1 = *Kptr; Kptr += 32;			\
  L0 += Ksave0 * sM[s0][jp0];			\
  L1 += Ksave0 * sM[s1][jp0];			\
  L2 += Ksave0 * sM[s2][jp0];			\
  L3 += Ksave0 * sM[s3][jp0];			\
  Ksave0 = *Kptr; Kptr += 32;			\
  L0 += Ksave1 * sM[s0][jp1];			\
  L1 += Ksave1 * sM[s1][jp1];			\
  L2 += Ksave1 * sM[s2][jp1];			\
  L3 += Ksave1 * sM[s3][jp1]

/* Prefetch K-data and unroll column loop completely */
#define SI_comp32x4(kind, f0, f1, f2, f3, s0, s1, s2, s3)	\
  if (Kindex[kind] != NULL_KINDEX) {				\
    /*110228    float *Kptr = K + (Kindex[kind] << 10) + tx;*/	\
    /*110228    float Ksave0 = *Kptr; Kptr += 32;*/		\
    /*110228    float L0 = 0.0f;*/				\
    /*110228    float L1 = 0.0f;*/				\
    /*110228    float L2 = 0.0f;*/				\
    /*110228    float L3 = 0.0f;*/				\
    /*110228    float Ksave1;*/					\
    real *Kptr = K + (Kindex[kind] << 10) + tx;			\
    real Ksave0 = *Kptr; Kptr += 32;				\
    real L0 = ZERO;						\
    real L1 = ZERO;						\
    real L2 = ZERO;						\
    real L3 = ZERO;						\
    real Ksave1;						\
    cal32x4(s0, s1, s2, s3,  0,  1);				\
    cal32x4(s0, s1, s2, s3,  2,  3);				\
    cal32x4(s0, s1, s2, s3,  4,  5);				\
    cal32x4(s0, s1, s2, s3,  6,  7);				\
    cal32x4(s0, s1, s2, s3,  8,  9);				\
    cal32x4(s0, s1, s2, s3, 10, 11);				\
    cal32x4(s0, s1, s2, s3, 12, 13);				\
    cal32x4(s0, s1, s2, s3, 14, 15);				\
    cal32x4(s0, s1, s2, s3, 16, 17);				\
    cal32x4(s0, s1, s2, s3, 18, 19);				\
    cal32x4(s0, s1, s2, s3, 20, 21);				\
    cal32x4(s0, s1, s2, s3, 22, 23);				\
    cal32x4(s0, s1, s2, s3, 24, 25);				\
    cal32x4(s0, s1, s2, s3, 26, 27);				\
    cal32x4(s0, s1, s2, s3, 28, 29);				\
    Ksave1 = *Kptr; Kptr += 32;					\
    L0 += Ksave0 * sM[s0][30];					\
    L1 += Ksave0 * sM[s1][30];					\
    L2 += Ksave0 * sM[s2][30];					\
    L3 += Ksave0 * sM[s3][30];					\
    L0 += Ksave1 * sM[s0][31];					\
    L1 += Ksave1 * sM[s1][31];					\
    L2 += Ksave1 * sM[s2][31];					\
    L3 += Ksave1 * sM[s3][31];					\
    *(Lptr + f0) += L0;						\
    *(Lptr + f1) += L1;						\
    *(Lptr + f2) += L2;						\
    *(Lptr + f3) += L3;						\
  }

#define cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, jp0, jp1)	\
  Ksave1 = *Kptr; Kptr += 32;					\
  L0 += Ksave0 * sM[s0][jp0];					\
  L1 += Ksave0 * sM[s1][jp0];					\
  L2 += Ksave0 * sM[s2][jp0];					\
  L3 += Ksave0 * sM[s3][jp0];					\
  L4 += Ksave0 * sM[s4][jp0];					\
  L5 += Ksave0 * sM[s5][jp0];					\
  L6 += Ksave0 * sM[s6][jp0];					\
  L7 += Ksave0 * sM[s7][jp0];					\
  Ksave0 = *Kptr; Kptr += 32;					\
  L0 += Ksave1 * sM[s0][jp1];					\
  L1 += Ksave1 * sM[s1][jp1];					\
  L2 += Ksave1 * sM[s2][jp1];					\
  L3 += Ksave1 * sM[s3][jp1];					\
  L4 += Ksave1 * sM[s4][jp1];					\
  L5 += Ksave1 * sM[s5][jp1];					\
  L6 += Ksave1 * sM[s6][jp1];					\
  L7 += Ksave1 * sM[s7][jp1]

/* Prefetch K-data and unroll column loop completely */
#define SI_comp32x8(kind, f0, f1, f2, f3, f4, f5, f6, f7, s0, s1, s2, s3, s4, s5, s6, s7) \
  {									\
    /*110228    float *Kptr = K + (Kindex[kind] << 10) + tx;*/		\
    /*110228    float Ksave0 = *Kptr; Kptr += 32;*/			\
    /*110228    float L0 = 0.0f;*/					\
    /*110228    float L1 = 0.0f;*/					\
    /*110228    float L2 = 0.0f;*/					\
    /*110228    float L3 = 0.0f;*/					\
    /*110228    float L4 = 0.0f;*/					\
    /*110228    float L5 = 0.0f;*/					\
    /*110228    float L6 = 0.0f;*/					\
    /*110228    float L7 = 0.0f;*/					\
    /*110228    float Ksave1;*/						\
    real *Kptr = K + (Kindex[kind] << 10) + tx;				\
    real Ksave0 = *Kptr; Kptr += 32;					\
    real L0 = ZERO;							\
    real L1 = ZERO;							\
    real L2 = ZERO;							\
    real L3 = ZERO;							\
    real L4 = ZERO;							\
    real L5 = ZERO;							\
    real L6 = ZERO;							\
    real L7 = ZERO;							\
    real Ksave1;							\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  0,  1);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  2,  3);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  4,  5);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  6,  7);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7,  8,  9);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 10, 11);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 12, 13);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 14, 15);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 16, 17);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 18, 19);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 20, 21);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 22, 23);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 24, 25);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 26, 27);			\
    cal32x8(s0, s1, s2, s3, s4, s5, s6, s7, 28, 29);			\
    Ksave1 = *Kptr; Kptr += 32;						\
    L0 += Ksave0 * sM[s0][30];						\
    L1 += Ksave0 * sM[s1][30];						\
    L2 += Ksave0 * sM[s2][30];						\
    L3 += Ksave0 * sM[s3][30];						\
    L4 += Ksave0 * sM[s4][30];						\
    L5 += Ksave0 * sM[s5][30];						\
    L6 += Ksave0 * sM[s6][30];						\
    L7 += Ksave0 * sM[s7][30];						\
    L0 += Ksave1 * sM[s0][31];						\
    L1 += Ksave1 * sM[s1][31];						\
    L2 += Ksave1 * sM[s2][31];						\
    L3 += Ksave1 * sM[s3][31];						\
    L4 += Ksave1 * sM[s4][31];						\
    L5 += Ksave1 * sM[s5][31];						\
    L6 += Ksave1 * sM[s6][31];						\
    L7 += Ksave1 * sM[s7][31];						\
    *(Lptr + f0) += L0;							\
    *(Lptr + f1) += L1;							\
    *(Lptr + f2) += L2;							\
    *(Lptr + f3) += L3;							\
    *(Lptr + f4) += L4;							\
    *(Lptr + f5) += L5;							\
    *(Lptr + f6) += L6;							\
    *(Lptr + f7) += L7;							\
  }

#define II0 0
#define II1 32
#define II2 64
#define II3 96
#define II4 128
#define II5 160
#define II6 192
#define II7 224

//110228__global__ void m2l_kern_sibling_blocking_r32(float *L, float *K, float *M, int *sourceclusters, int *Ktable, int FCsta)
__global__ void m2l_kern_sibling_blocking_r32(real *L, real *K, real *M, int *sourceclusters, int *Ktable, int FCsta)
{
  /* Set the pointer to L for the 0th sibling in the current
     observation cluster OC, to which bx-th thread-block is
     assigned */
  //110228  float *Lptr = L + (bx << 8) + tx; // log2(8*r)=8
  real *Lptr = L + (bx << 8) + tx; // log2(8*r)=8
  
  /* Load 0th siblings of 27 source clusters (SCs) for OC;
     ilist[27:31] are unused */
  __shared__ int ilist[32]; // 27 or more; 128 bytes
  ilist[tx] = sourceclusters[((FCsta + bx) << 5) + tx]; // log2(32)=5

  /* Copy Ktable in shared memory */
  __shared__ int sKtable[343]; // 1372 bytes
  sKtable[tx      ] = Ktable[tx      ];
  sKtable[tx +  32] = Ktable[tx +  32];
  sKtable[tx +  64] = Ktable[tx +  64];
  sKtable[tx +  96] = Ktable[tx +  96];
  sKtable[tx + 128] = Ktable[tx + 128];
  sKtable[tx + 160] = Ktable[tx + 160];
  sKtable[tx + 192] = Ktable[tx + 192];
  sKtable[tx + 224] = Ktable[tx + 224];
  sKtable[tx + 256] = Ktable[tx + 256];
  sKtable[tx + 288] = Ktable[tx + 288];
  if (tx < 23) {
    sKtable[tx + 320] = Ktable[tx + 320];
  }
  
  /* Load Kindex0[] and Kindex1[] */
  __shared__ int Kindex0[32], Kindex1[32], Kindex[32]; // 27 or more; 384 bytes
  load_Kindex0_Kindex1(tx);

  /* Ensure that ilist[], sKtable[], Kindex0[], and Kindex1[] were loaded */
  __syncthreads();

  /* Loop over source-cluster indices */
  for (int d = 0; d < 27; d ++) {
    
    /* If the 0th sibling, which is the representative of SC, does not
       exist in the hierachy (then, the other seven siblings in SC
       neither exist) or SC coincides with OC, we can skip the
       computaion for SC */
    if (ilist[d] != NULL_CELL) {

      /* Load K-indices for all the 27 interaction-kinds */
      Kindex[tx] = sKtable[Kindex0[d] + Kindex1[tx]];

      /* Ensure that Kindex[] was loaded */
      __syncthreads();
      
      /* Allocate the memory for all eight M-vectors in SC */
      //110228      __shared__ float sM[8][32]; // 1024 bytes
      __shared__ real sM[8][32]; // 1024 bytes in single

      {
	//110228	float *Mptr = M + (ilist[d] << 5) + tx; // log2(r)=5
	real *Mptr = M + (ilist[d] << 5) + tx; // log2(r)=5
	sM[0][tx] = *(Mptr + II0); /* S0 */
	sM[1][tx] = *(Mptr + II1); /* S1 */
	sM[2][tx] = *(Mptr + II2); /* S2 */
	sM[3][tx] = *(Mptr + II3); /* S3 */
	sM[4][tx] = *(Mptr + II4); /* S4 */
	sM[5][tx] = *(Mptr + II5); /* S5 */
	sM[6][tx] = *(Mptr + II6); /* S6 */
	sM[7][tx] = *(Mptr + II7); /* S7 */
      }
      __syncthreads();

      SI_comp32x1(F0S7, II0, J7);
      SI_comp32x1(F1S6, II1, J6);
      SI_comp32x1(F2S5, II2, J5);
      SI_comp32x1(F3S4, II3, J4);
      SI_comp32x1(F4S3, II4, J3);
      SI_comp32x1(F5S2, II5, J2);
      SI_comp32x1(F6S1, II6, J1);
      SI_comp32x1(F7S0, II7, J0);
      
      SI_comp32x2(F0S3, II0, II4, J3, J7);
      SI_comp32x2(F0S5, II0, II2, J5, J7);
      SI_comp32x2(F0S6, II0, II1, J6, J7);
      SI_comp32x2(F1S2, II1, II5, J2, J6);
      SI_comp32x2(F1S4, II1, II3, J4, J6);
      SI_comp32x2(F2S1, II2, II6, J1, J5);
      SI_comp32x2(F2S4, II2, II3, J4, J5);
      SI_comp32x2(F3S0, II3, II7, J0, J4);
      SI_comp32x2(F4S1, II4, II6, J1, J3);
      SI_comp32x2(F4S2, II4, II5, J2, J3);
      SI_comp32x2(F5S0, II5, II7, J0, J2);
      SI_comp32x2(F6S0, II6, II7, J0, J1);
      
      SI_comp32x4(F0S4, II0, II1, II2, II3, J4, J5, J6, J7);
      SI_comp32x4(F0S2, II0, II1, II4, II5, J2, J3, J6, J7);
      SI_comp32x4(F0S1, II0, II2, II4, II6, J1, J3, J5, J7);
      SI_comp32x4(F1S0, II1, II3, II5, II7, J0, J2, J4, J6);
      SI_comp32x4(F2S0, II2, II3, II6, II7, J0, J1, J4, J5);
      SI_comp32x4(F4S0, II4, II5, II6, II7, J0, J1, J2, J3);
      
      SI_comp32x8(F0S0, II0, II1, II2, II3, II4, II5, II6, II7, J0, J1, J2, J3, J4, J5, J6, J7);
      __syncthreads();
    }
  }
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
#endif /* M2L_KERN_SIBLING_BLOCKING_CU */
