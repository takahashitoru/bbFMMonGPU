#include "bbfmm.h"

//120214#ifndef POW8
//120214#define POW8(i) (1 << (3 * (i)))
//120214#endif

#if defined(CPU8) // See anotherDownwardPassNearField.c
#define CPU7
#define CPU7A
#endif

/**************************************************************************/
#if defined(CPU9)
/**************************************************************************/
/**************************************************************************/
#if defined(CPU9U)
/**************************************************************************/
/* Based on CPU9T */

#if !defined(NINTER_B2) // 1,2,4,16
#define NINTER_B2 8
#endif

#if !defined(NINTER_B4) // 1,2,4,16
#define NINTER_B4 8
#endif

#if !defined(NINTER_B8) // 1,2,4,16
#define NINTER_B8 8
#endif

#if !defined(UNROLL_FACTOR_B2) // 0,4,8,16...
#define UNROLL_FACTOR_B2 0
#endif

#if !defined(UNROLL_FACTOR_B4) // 0,4,8,16...
#define UNROLL_FACTOR_B4 8
#endif

#if !defined(UNROLL_FACTOR_B8)
#define UNROLL_FACTOR_B8 8
#endif

#if !defined(LEVEL_TO_SWITCH_B2_TO_B4) // 3 or more
#define LEVEL_TO_SWITCH_B2_TO_B4 5
#endif

#if !defined(LEVEL_TO_SWITCH_B4_TO_B8) // 4 or more
#define LEVEL_TO_SWITCH_B4_TO_B8 6
//#define LEVEL_TO_SWITCH_B4_TO_B8 7
//#define LEVEL_TO_SWITCH_B4_TO_B8 9999
#endif

static void cmp21(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp22(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp24(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp28(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    //    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    const real Ltmp = K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    Lptr[k] += Ltmp;
  }
}

static void cmp216(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}

static void cmp41(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp42(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp44(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

//#include <pmmintrin.h> // SSE3
#include <smmintrin.h> // SSE4

static void cmp48(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=4
{
#if(0)
#if defined DOUBLE
#error DOUBLE is not supported yet
#endif
#if(0) // works, but slow
  const float K0 = Kijptr[Kijoff0];
  const float K1 = Kijptr[Kijoff1];
  const float K2 = Kijptr[Kijoff2];
  const float K3 = Kijptr[Kijoff3];
  const float K4 = Kijptr[Kijoff4];
  const float K5 = Kijptr[Kijoff5];
  const float K6 = Kijptr[Kijoff6];
  const float K7 = Kijptr[Kijoff7];
  const __m128 vK0 = _mm_setr_ps(K0, K1, K2, K3);
  const __m128 vK4 = _mm_setr_ps(K4, K5, K6, K7);
  const float *M0 = Mjptr + Mjoff0;
  const float *M1 = Mjptr + Mjoff1;
  const float *M2 = Mjptr + Mjoff2;
  const float *M3 = Mjptr + Mjoff3;
  const float *M4 = Mjptr + Mjoff4;
  const float *M5 = Mjptr + Mjoff5;
  const float *M6 = Mjptr + Mjoff6;
  const float *M7 = Mjptr + Mjoff7;
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) {
    const int itmp = Mjshift[k];
    const __m128 vM0 = _mm_setr_ps(M0[itmp], M1[itmp], M2[itmp], M3[itmp]);
    const __m128 vM4 = _mm_setr_ps(M4[itmp], M5[itmp], M6[itmp], M7[itmp]);
    const __m128 vL0 = _mm_mul_ps(vK0, vM0);
    const __m128 vL4 = _mm_mul_ps(vK4, vM4);
    const __m128 vL = _mm_add_ps(vL0, vL4);
    float sL[4] __attribute__ ((aligned(16)));
    _mm_store_ps(sL, vL);
    const float Ltmp = sL[0] + sL[1] + sL[2] + sL[3]; 
    Lptr[k] += Ltmp;
  }
#else // works, but slow
  const float K0 = Kijptr[Kijoff0];
  const float K1 = Kijptr[Kijoff1];
  const float K2 = Kijptr[Kijoff2];
  const float K3 = Kijptr[Kijoff3];
  const float K4 = Kijptr[Kijoff4];
  const float K5 = Kijptr[Kijoff5];
  const float K6 = Kijptr[Kijoff6];
  const float K7 = Kijptr[Kijoff7];
  const __m128 vK0 = _mm_set1_ps(K0);
  const __m128 vK1 = _mm_set1_ps(K1);
  const __m128 vK2 = _mm_set1_ps(K2);
  const __m128 vK3 = _mm_set1_ps(K3);
  const __m128 vK4 = _mm_set1_ps(K4);
  const __m128 vK5 = _mm_set1_ps(K5);
  const __m128 vK6 = _mm_set1_ps(K6);
  const __m128 vK7 = _mm_set1_ps(K7);
  const float *M0 = Mjptr + Mjoff0;
  const float *M1 = Mjptr + Mjoff1;
  const float *M2 = Mjptr + Mjoff2;
  const float *M3 = Mjptr + Mjoff3;
  const float *M4 = Mjptr + Mjoff4;
  const float *M5 = Mjptr + Mjoff5;
  const float *M6 = Mjptr + Mjoff6;
  const float *M7 = Mjptr + Mjoff7;
  for (int k = 0; k < 64; k += 4) {
    const int i0 = Mjshift[k];
    const int i1 = Mjshift[k + 1];
    const int i2 = Mjshift[k + 2];
    const int i3 = Mjshift[k + 3];
    __m128 vM, vL;
    vM = _mm_setr_ps(M0[i0], M0[i1], M0[i2], M0[i3]);
    vL = _mm_mul_ps(vK0, vM);
    vM = _mm_setr_ps(M1[i0], M1[i1], M1[i2], M1[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK1, vM));
    vM = _mm_setr_ps(M2[i0], M2[i1], M2[i2], M2[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK2, vM));
    vM = _mm_setr_ps(M3[i0], M3[i1], M3[i2], M3[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK3, vM));
    vM = _mm_setr_ps(M4[i0], M4[i1], M4[i2], M4[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK4, vM));
    vM = _mm_setr_ps(M5[i0], M5[i1], M5[i2], M5[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK5, vM));
    vM = _mm_setr_ps(M6[i0], M6[i1], M6[i2], M6[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK6, vM));
    vM = _mm_setr_ps(M7[i0], M7[i1], M7[i2], M7[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK7, vM));
    float sL[4] __attribute__ ((aligned(16)));
    _mm_store_ps(sL, vL);
    Lptr[k] += sL[0];
    Lptr[k + 1] += sL[1];
    Lptr[k + 2] += sL[2];
    Lptr[k + 3] += sL[3];
  }
#endif
#else
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0;
  real *M1 = Mjptr + Mjoff1;
  real *M2 = Mjptr + Mjoff2;
  real *M3 = Mjptr + Mjoff3;
  real *M4 = Mjptr + Mjoff4;
  real *M5 = Mjptr + Mjoff5;
  real *M6 = Mjptr + Mjoff6;
  real *M7 = Mjptr + Mjoff7;
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    const real Ltmp = K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    Lptr[k] += Ltmp;
  }
#endif
}

static void cmp416(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0;
  real *M1 = Mjptr + Mjoff1;
  real *M2 = Mjptr + Mjoff2;
  real *M3 = Mjptr + Mjoff3;
  real *M4 = Mjptr + Mjoff4;
  real *M5 = Mjptr + Mjoff5;
  real *M6 = Mjptr + Mjoff6;
  real *M7 = Mjptr + Mjoff7;
  real *M8 = Mjptr + Mjoff8;
  real *M9 = Mjptr + Mjoff9;
  real *M10 = Mjptr + Mjoff10;
  real *M11 = Mjptr + Mjoff11;
  real *M12 = Mjptr + Mjoff12;
  real *M13 = Mjptr + Mjoff13;
  real *M14 = Mjptr + Mjoff14;
  real *M15 = Mjptr + Mjoff15;
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) {
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}


static void cmp81(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0;
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp82(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0;
  real *M1 = Mjptr + Mjoff1;
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp84(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0;
  real *M1 = Mjptr + Mjoff1;
  real *M2 = Mjptr + Mjoff2;
  real *M3 = Mjptr + Mjoff3;
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp88(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=8
{
#if(0)
#if defined DOUBLE
#error DOUBLE is not supported yet
#endif
#if(1) // works, but slow
  const float K0 = Kijptr[Kijoff0];
  const float K1 = Kijptr[Kijoff1];
  const float K2 = Kijptr[Kijoff2];
  const float K3 = Kijptr[Kijoff3];
  const float K4 = Kijptr[Kijoff4];
  const float K5 = Kijptr[Kijoff5];
  const float K6 = Kijptr[Kijoff6];
  const float K7 = Kijptr[Kijoff7];
  const __m128 vK0 = _mm_setr_ps(K0, K1, K2, K3);
  const __m128 vK4 = _mm_setr_ps(K4, K5, K6, K7);
  const float *M0 = Mjptr + Mjoff0;
  const float *M1 = Mjptr + Mjoff1;
  const float *M2 = Mjptr + Mjoff2;
  const float *M3 = Mjptr + Mjoff3;
  const float *M4 = Mjptr + Mjoff4;
  const float *M5 = Mjptr + Mjoff5;
  const float *M6 = Mjptr + Mjoff6;
  const float *M7 = Mjptr + Mjoff7;
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) {
    const int itmp = Mjshift[k];
    const __m128 vM0 = _mm_setr_ps(M0[itmp], M1[itmp], M2[itmp], M3[itmp]);
    const __m128 vM4 = _mm_setr_ps(M4[itmp], M5[itmp], M6[itmp], M7[itmp]);
    const __m128 vL0 = _mm_mul_ps(vK0, vM0);
    const __m128 vL4 = _mm_mul_ps(vK4, vM4);
    const __m128 vL = _mm_add_ps(vL0, vL4);
    float sL[4] __attribute__ ((aligned(16)));
    _mm_store_ps(sL, vL);
    const float Ltmp = sL[0] + sL[1] + sL[2] + sL[3]; 
    Lptr[k] += Ltmp;
  }
#else // works, but slow
  const float K0 = Kijptr[Kijoff0];
  const float K1 = Kijptr[Kijoff1];
  const float K2 = Kijptr[Kijoff2];
  const float K3 = Kijptr[Kijoff3];
  const float K4 = Kijptr[Kijoff4];
  const float K5 = Kijptr[Kijoff5];
  const float K6 = Kijptr[Kijoff6];
  const float K7 = Kijptr[Kijoff7];
  const __m128 vK0 = _mm_set1_ps(K0);
  const __m128 vK1 = _mm_set1_ps(K1);
  const __m128 vK2 = _mm_set1_ps(K2);
  const __m128 vK3 = _mm_set1_ps(K3);
  const __m128 vK4 = _mm_set1_ps(K4);
  const __m128 vK5 = _mm_set1_ps(K5);
  const __m128 vK6 = _mm_set1_ps(K6);
  const __m128 vK7 = _mm_set1_ps(K7);
  const float *M0 = Mjptr + Mjoff0;
  const float *M1 = Mjptr + Mjoff1;
  const float *M2 = Mjptr + Mjoff2;
  const float *M3 = Mjptr + Mjoff3;
  const float *M4 = Mjptr + Mjoff4;
  const float *M5 = Mjptr + Mjoff5;
  const float *M6 = Mjptr + Mjoff6;
  const float *M7 = Mjptr + Mjoff7;
  for (int k = 0; k < 512; k += 4) {
    const int i0 = Mjshift[k];
    const int i1 = Mjshift[k + 1];
    const int i2 = Mjshift[k + 2];
    const int i3 = Mjshift[k + 3];
    __m128 vM, vL;
    vM = _mm_setr_ps(M0[i0], M0[i1], M0[i2], M0[i3]);
    vL = _mm_mul_ps(vK0, vM);
    vM = _mm_setr_ps(M1[i0], M1[i1], M1[i2], M1[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK1, vM));
    vM = _mm_setr_ps(M2[i0], M2[i1], M2[i2], M2[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK2, vM));
    vM = _mm_setr_ps(M3[i0], M3[i1], M3[i2], M3[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK3, vM));
    vM = _mm_setr_ps(M4[i0], M4[i1], M4[i2], M4[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK4, vM));
    vM = _mm_setr_ps(M5[i0], M5[i1], M5[i2], M5[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK5, vM));
    vM = _mm_setr_ps(M6[i0], M6[i1], M6[i2], M6[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK6, vM));
    vM = _mm_setr_ps(M7[i0], M7[i1], M7[i2], M7[i3]);
    vL = _mm_add_ps(vL, _mm_mul_ps(vK7, vM));
    float sL[4] __attribute__ ((aligned(16)));
    _mm_store_ps(sL, vL);
    Lptr[k] += sL[0];
    Lptr[k + 1] += sL[1];
    Lptr[k + 2] += sL[2];
    Lptr[k + 3] += sL[3];
  }
#endif
#else
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0;
  real *M1 = Mjptr + Mjoff1;
  real *M2 = Mjptr + Mjoff2;
  real *M3 = Mjptr + Mjoff3;
  real *M4 = Mjptr + Mjoff4;
  real *M5 = Mjptr + Mjoff5;
  real *M6 = Mjptr + Mjoff6;
  real *M7 = Mjptr + Mjoff7;
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    const real Ltmp = K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    Lptr[k] += Ltmp;
 }
#endif
}

static void cmp816(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // not vectorized?
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
    //    const real Ltmp = K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
    //    Lptr[k] += Ltmp;
  }
}


#define CMP21(Kijoff0, Mjoff0)						\
  {									\
    cmp21(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP22(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp22(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP24(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp24(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP28(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp28(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP216(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp216(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }


#define CMP41(Kijoff0, Mjoff0)						\
  {									\
    cmp41(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP42(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp42(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP44(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp44(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP48(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp48(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP416(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp416(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }


#define CMP81(Kijoff0, Mjoff0)						\
  {									\
    cmp81(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP82(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp82(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP84(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp84(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP88(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp88(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP816(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp816(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }

#include "aux_CPU9P.h" // Created by aux_CPU9P.c

#if (NINTER_B2 == 1)
#define COMPXYZ_B2_S0 COMPXYZ_B2_I1_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I1_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I1_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I1_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I1_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I1_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I1_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I1_S7
#elif (NINTER_B2 == 2) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I2_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I2_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I2_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I2_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I2_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I2_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I2_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I2_S7
#elif (NINTER_B2 == 4) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I4_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I4_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I4_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I4_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I4_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I4_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I4_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I4_S7
#elif (NINTER_B2 == 8) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I8_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I8_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I8_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I8_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I8_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I8_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I8_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I8_S7
#elif (NINTER_B2 == 16) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I16_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I16_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I16_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I16_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I16_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I16_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I16_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I16_S7
#else
#error Undefined NINTER_B2.
#endif

#if (NINTER_B4 == 1)
#define COMPXYZ_B4_S0 COMPXYZ_B4_I1_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I1_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I1_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I1_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I1_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I1_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I1_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I1_S7
#elif (NINTER_B4 == 2) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I2_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I2_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I2_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I2_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I2_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I2_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I2_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I2_S7
#elif (NINTER_B4 == 4) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I4_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I4_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I4_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I4_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I4_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I4_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I4_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I4_S7
#elif (NINTER_B4 == 8) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I8_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I8_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I8_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I8_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I8_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I8_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I8_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I8_S7
#elif (NINTER_B4 == 16) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I16_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I16_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I16_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I16_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I16_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I16_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I16_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I16_S7
#else
#error Undefined NINTER_B4.
#endif

#if (NINTER_B8 == 1)
#define COMPXYZ_B8_S0 COMPXYZ_B8_I1_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I1_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I1_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I1_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I1_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I1_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I1_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I1_S7
#elif (NINTER_B8 == 2) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I2_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I2_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I2_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I2_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I2_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I2_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I2_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I2_S7
#elif (NINTER_B8 == 4) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I4_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I4_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I4_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I4_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I4_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I4_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I4_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I4_S7
#elif (NINTER_B8 == 8) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I8_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I8_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I8_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I8_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I8_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I8_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I8_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I8_S7
#elif (NINTER_B8 == 16) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I16_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I16_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I16_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I16_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I16_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I16_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I16_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I16_S7
#else
#error Undefined NINTER_B8.
#endif


#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_S##s();				\
  } else if (B == 4) {				\
    COMPXYZ_B4_S##s();				\
  } else if (B == 8) {				\
    COMPXYZ_B8_S##s();				\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}


#define LOAD_M1(n)						\
  for (int iz = 0; iz < n; iz ++) {				\
    for (int iy = 0; iy < n; iy ++) {				\
      const real *Mptr0 = Mptr + (iz * ncpe + iy) * ncpe;	\
      for (int ix = 0; ix < n; ix ++) {				\
	Mj[iz][iy][ix] = Mptr0[ix];				\
      }								\
    }								\
  }

#define LOAD_M2(n)						\
  for (int iz = 0; iz < n; iz ++) {				\
    for (int iy = 0; iy < n; iy += 2) {				\
      const real *Mptr0 = Mptr + (iz * ncpe + (iy + 0)) * ncpe;	\
      const real *Mptr1 = Mptr0 + ncpe;				\
      for (int ix = 0; ix < n; ix ++) {				\
	Mj[iz][iy + 0][ix] = Mptr0[ix];				\
	Mj[iz][iy + 1][ix] = Mptr1[ix];				\
      }								\
    }								\
  }

#define LOAD_M4(n)						\
  for (int iz = 0; iz < n; iz ++) {				\
    for (int iy = 0; iy < n; iy += 4) {				\
      const real *Mptr0 = Mptr + (iz * ncpe + (iy + 0)) * ncpe;	\
      const real *Mptr1 = Mptr0 + ncpe;				\
      const real *Mptr2 = Mptr1 + ncpe;				\
      const real *Mptr3 = Mptr2 + ncpe;				\
      for (int ix = 0; ix < n; ix ++) {				\
	Mj[iz][iy + 0][ix] = Mptr0[ix];				\
	Mj[iz][iy + 1][ix] = Mptr1[ix];				\
	Mj[iz][iy + 2][ix] = Mptr2[ix];				\
	Mj[iz][iy + 3][ix] = Mptr3[ix];				\
      }								\
    }								\
  }

static load_M(const int B, const int ncpe, const real *Mptr, real Mj[2 * B + 4][2 * B + 4][2 * B + 4])
{
  if (B == 2) {
    LOAD_M2(8);
    //    LOAD_M1(8);
  } else if (B == 4) {
    LOAD_M2(12);
    //    LOAD_M1(12);
  } else if (B == 8) {
    LOAD_M4(20); // LOOP WAS VECTORIZED.
    //    LOAD_M2(20); // LOOP WAS VECTORIZED.
    //    LOAD_M1(20);
  } else {
    INFO("Undefined B=%d. Exit.\n", B);
    exit(EXIT_FAILURE);
  }
}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
#if(1)
    load_M(B, ncpe, Mptr, Mj);
#else
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
#endif
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(0); // s=0
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(1); // s=1
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(2); // s=2
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(3); // s=3
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(4); // s=4
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(5); // s=5
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(6); // s=6
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(7); // s=7
      Lptr += B * B * B;

      /* Point to next i */
      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9T)
/**************************************************************************/
/* Based on CPU9S */

#if !defined(NINTER_B2)
//#define NINTER_B2 1
//#define NINTER_B2 2
//#define NINTER_B2 4
#define NINTER_B2 8
//#define NINTER_B2 16
#endif

#if !defined(NINTER_B4)
//#define NINTER_B4 1
//#define NINTER_B4 2
//#define NINTER_B4 4
#define NINTER_B4 8
//#define NINTER_B4 16
#endif

#if !defined(NINTER_B8)
//#define NINTER_B8 1
//#define NINTER_B8 2
//#define NINTER_B8 4
#define NINTER_B8 8
//#define NINTER_B8 16
#endif

#if !defined(UNROLL_FACTOR_B2)
#define UNROLL_FACTOR_B2 0
//#define UNROLL_FACTOR_B2 8
#endif

#if !defined(UNROLL_FACTOR_B4)
//#define UNROLL_FACTOR_B4 0
//#define UNROLL_FACTOR_B4 4
#define UNROLL_FACTOR_B4 8
//#define UNROLL_FACTOR_B4 16
#endif

#if !defined(UNROLL_FACTOR_B8)
//#define UNROLL_FACTOR_B8 0
//#define UNROLL_FACTOR_B8 4
#define UNROLL_FACTOR_B8 8
//#define UNROLL_FACTOR_B8 16 // almost the same as 8
#endif

#if !defined(LEVEL_TO_SWITCH_B2_TO_B4) // 3 or more
//#define LEVEL_TO_SWITCH_B2_TO_B4 3
//#define LEVEL_TO_SWITCH_B2_TO_B4 4
#define LEVEL_TO_SWITCH_B2_TO_B4 5
//#define LEVEL_TO_SWITCH_B2_TO_B4 6
#endif

#if !defined(LEVEL_TO_SWITCH_B4_TO_B8) // 4 or more
//#define LEVEL_TO_SWITCH_B4_TO_B8 4
//#define LEVEL_TO_SWITCH_B4_TO_B8 5
#define LEVEL_TO_SWITCH_B4_TO_B8 6
//#define LEVEL_TO_SWITCH_B4_TO_B8 7
#endif

static void cmp21(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp22(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp24(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp28(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    //    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    const real Ltmp = K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    Lptr[k] += Ltmp;
  }
}

static void cmp216(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}

static void cmp41(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp42(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp44(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp48(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    //    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    const real Ltmp = K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    Lptr[k] += Ltmp;
  }
}

static void cmp416(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) {
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}


static void cmp81(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp82(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp84(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp88(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    //    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    const real Ltmp = K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
    Lptr[k] += Ltmp;
 }
}

static void cmp816(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // not vectorized?
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
    //    const real Ltmp = K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
    //    Lptr[k] += Ltmp;
  }
}


#define CMP21(Kijoff0, Mjoff0)						\
  {									\
    cmp21(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP22(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp22(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP24(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp24(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP28(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp28(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP216(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp216(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }


#define CMP41(Kijoff0, Mjoff0)						\
  {									\
    cmp41(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP42(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp42(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP44(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp44(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP48(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp48(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP416(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp416(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }


#define CMP81(Kijoff0, Mjoff0)						\
  {									\
    cmp81(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP82(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp82(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP84(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp84(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP88(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp88(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP816(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp816(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }

#include "aux_CPU9P.h" // Created by aux_CPU9P.c

#if (NINTER_B2 == 1)
#define COMPXYZ_B2_S0 COMPXYZ_B2_I1_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I1_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I1_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I1_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I1_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I1_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I1_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I1_S7
#elif (NINTER_B2 == 2) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I2_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I2_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I2_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I2_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I2_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I2_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I2_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I2_S7
#elif (NINTER_B2 == 4) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I4_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I4_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I4_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I4_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I4_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I4_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I4_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I4_S7
#elif (NINTER_B2 == 8) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I8_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I8_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I8_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I8_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I8_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I8_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I8_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I8_S7
#elif (NINTER_B2 == 16) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I16_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I16_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I16_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I16_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I16_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I16_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I16_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I16_S7
#else
#error Undefined NINTER_B2.
#endif

#if (NINTER_B4 == 1)
#define COMPXYZ_B4_S0 COMPXYZ_B4_I1_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I1_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I1_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I1_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I1_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I1_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I1_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I1_S7
#elif (NINTER_B4 == 2) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I2_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I2_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I2_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I2_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I2_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I2_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I2_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I2_S7
#elif (NINTER_B4 == 4) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I4_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I4_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I4_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I4_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I4_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I4_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I4_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I4_S7
#elif (NINTER_B4 == 8) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I8_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I8_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I8_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I8_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I8_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I8_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I8_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I8_S7
#elif (NINTER_B4 == 16) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I16_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I16_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I16_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I16_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I16_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I16_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I16_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I16_S7
#else
#error Undefined NINTER_B4.
#endif

#if (NINTER_B8 == 1)
#define COMPXYZ_B8_S0 COMPXYZ_B8_I1_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I1_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I1_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I1_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I1_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I1_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I1_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I1_S7
#elif (NINTER_B8 == 2) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I2_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I2_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I2_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I2_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I2_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I2_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I2_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I2_S7
#elif (NINTER_B8 == 4) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I4_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I4_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I4_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I4_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I4_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I4_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I4_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I4_S7
#elif (NINTER_B8 == 8) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I8_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I8_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I8_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I8_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I8_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I8_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I8_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I8_S7
#elif (NINTER_B8 == 16) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I16_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I16_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I16_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I16_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I16_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I16_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I16_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I16_S7
#else
#error Undefined NINTER_B8.
#endif


#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_S##s();				\
  } else if (B == 4) {				\
    COMPXYZ_B4_S##s();				\
  } else if (B == 8) {				\
    COMPXYZ_B8_S##s();				\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}


#define LOAD_M1(n)						\
  for (int iz = 0; iz < n; iz ++) {				\
    for (int iy = 0; iy < n; iy ++) {				\
      const real *Mptr0 = Mptr + (iz * ncpe + iy) * ncpe;	\
      for (int ix = 0; ix < n; ix ++) {				\
	Mj[iz][iy][ix] = Mptr0[ix];				\
      }								\
    }								\
  }

#define LOAD_M2(n)						\
  for (int iz = 0; iz < n; iz ++) {				\
    for (int iy = 0; iy < n; iy += 2) {				\
      const real *Mptr0 = Mptr + (iz * ncpe + (iy + 0)) * ncpe;	\
      const real *Mptr1 = Mptr0 + ncpe;				\
      for (int ix = 0; ix < n; ix ++) {				\
	Mj[iz][iy + 0][ix] = Mptr0[ix];				\
	Mj[iz][iy + 1][ix] = Mptr1[ix];				\
      }								\
    }								\
  }

#define LOAD_M4(n)						\
  for (int iz = 0; iz < n; iz ++) {				\
    for (int iy = 0; iy < n; iy += 4) {				\
      const real *Mptr0 = Mptr + (iz * ncpe + (iy + 0)) * ncpe;	\
      const real *Mptr1 = Mptr0 + ncpe;				\
      const real *Mptr2 = Mptr1 + ncpe;				\
      const real *Mptr3 = Mptr2 + ncpe;				\
      for (int ix = 0; ix < n; ix ++) {				\
	Mj[iz][iy + 0][ix] = Mptr0[ix];				\
	Mj[iz][iy + 1][ix] = Mptr1[ix];				\
	Mj[iz][iy + 2][ix] = Mptr2[ix];				\
	Mj[iz][iy + 3][ix] = Mptr3[ix];				\
      }								\
    }								\
  }

static load_M(const int B, const int ncpe, const real *Mptr, real Mj[2 * B + 4][2 * B + 4][2 * B + 4])
{
  if (B == 2) {
    LOAD_M2(8);
    //    LOAD_M1(8);
  } else if (B == 4) {
    LOAD_M2(12);
    //    LOAD_M1(12);
  } else if (B == 8) {
    LOAD_M4(20); // LOOP WAS VECTORIZED.
    //    LOAD_M2(20); // LOOP WAS VECTORIZED.
    //    LOAD_M1(20);
  } else {
    INFO("Undefined B=%d. Exit.\n", B);
    exit(EXIT_FAILURE);
  }
}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
#if(1)
    load_M(B, ncpe, Mptr, Mj);
#else
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
#endif
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(0); // s=0
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(1); // s=1
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(2); // s=2
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(3); // s=3
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(4); // s=4
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(5); // s=5
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(6); // s=6
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(7); // s=7
      Lptr += B * B * B;

      /* Point to next i */
      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9S)
/**************************************************************************/
/* Based on CPU9Q */

#if !defined(NINTER_B2)
//#define NINTER_B2 1
//#define NINTER_B2 2
//#define NINTER_B2 4
#define NINTER_B2 8
//#define NINTER_B2 16
#endif

#if !defined(NINTER_B4)
//#define NINTER_B4 1
//#define NINTER_B4 2
//#define NINTER_B4 4
#define NINTER_B4 8
//#define NINTER_B4 16
#endif

#if !defined(NINTER_B8)
//#define NINTER_B8 1
//#define NINTER_B8 2
//#define NINTER_B8 4
#define NINTER_B8 8
//#define NINTER_B8 16
#endif

#if !defined(UNROLL_FACTOR_B2)
#define UNROLL_FACTOR_B2 0
//#define UNROLL_FACTOR_B2 8
#endif

#if !defined(UNROLL_FACTOR_B4)
//#define UNROLL_FACTOR_B4 0
//#define UNROLL_FACTOR_B4 4
#define UNROLL_FACTOR_B4 8
//#define UNROLL_FACTOR_B4 16
#endif

#if !defined(UNROLL_FACTOR_B8)
//#define UNROLL_FACTOR_B8 0
//#define UNROLL_FACTOR_B8 4
#define UNROLL_FACTOR_B8 8
//#define UNROLL_FACTOR_B8 16
#endif

#if !defined(LEVEL_TO_SWITCH_B2_TO_B4) // 3 or more
//#define LEVEL_TO_SWITCH_B2_TO_B4 3
//#define LEVEL_TO_SWITCH_B2_TO_B4 4
#define LEVEL_TO_SWITCH_B2_TO_B4 5
//#define LEVEL_TO_SWITCH_B2_TO_B4 6
#endif

#if !defined(LEVEL_TO_SWITCH_B4_TO_B8) // 4 or more
//#define LEVEL_TO_SWITCH_B4_TO_B8 4
//#define LEVEL_TO_SWITCH_B4_TO_B8 5
#define LEVEL_TO_SWITCH_B4_TO_B8 6
//#define LEVEL_TO_SWITCH_B4_TO_B8 7
#endif

static void cmp21(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp22(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp24(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp28(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
  }
}

static void cmp216(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}

static void cmp41(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp42(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp44(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp48(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
  }
}

static void cmp416(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) {
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}


static void cmp81(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp82(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp84(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp88(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
  }
}

static void cmp816(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=8
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B8)
  for (int k = 0; k < 512; k ++) {
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}


#define CMP21(Kijoff0, Mjoff0)						\
  {									\
    cmp21(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP22(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp22(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP24(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp24(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP28(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp28(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP216(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp216(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }


#define CMP41(Kijoff0, Mjoff0)						\
  {									\
    cmp41(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP42(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp42(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP44(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp44(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP48(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp48(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP416(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp416(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }


#define CMP81(Kijoff0, Mjoff0)						\
  {									\
    cmp81(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP82(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp82(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP84(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp84(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP88(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp88(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP816(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp816(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }

#include "aux_CPU9P.h" // Created by aux_CPU9P.c

#if (NINTER_B2 == 1)
#define COMPXYZ_B2_S0 COMPXYZ_B2_I1_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I1_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I1_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I1_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I1_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I1_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I1_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I1_S7
#elif (NINTER_B2 == 2) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I2_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I2_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I2_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I2_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I2_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I2_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I2_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I2_S7
#elif (NINTER_B2 == 4) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I4_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I4_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I4_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I4_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I4_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I4_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I4_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I4_S7
#elif (NINTER_B2 == 8) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I8_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I8_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I8_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I8_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I8_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I8_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I8_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I8_S7
#elif (NINTER_B2 == 16) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I16_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I16_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I16_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I16_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I16_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I16_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I16_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I16_S7
#else
#error Undefined NINTER_B2.
#endif

#if (NINTER_B4 == 1)
#define COMPXYZ_B4_S0 COMPXYZ_B4_I1_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I1_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I1_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I1_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I1_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I1_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I1_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I1_S7
#elif (NINTER_B4 == 2) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I2_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I2_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I2_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I2_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I2_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I2_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I2_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I2_S7
#elif (NINTER_B4 == 4) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I4_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I4_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I4_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I4_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I4_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I4_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I4_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I4_S7
#elif (NINTER_B4 == 8) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I8_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I8_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I8_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I8_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I8_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I8_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I8_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I8_S7
#elif (NINTER_B4 == 16) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I16_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I16_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I16_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I16_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I16_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I16_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I16_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I16_S7
#else
#error Undefined NINTER_B4.
#endif

#if (NINTER_B8 == 1)
#define COMPXYZ_B8_S0 COMPXYZ_B8_I1_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I1_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I1_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I1_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I1_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I1_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I1_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I1_S7
#elif (NINTER_B8 == 2) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I2_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I2_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I2_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I2_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I2_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I2_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I2_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I2_S7
#elif (NINTER_B8 == 4) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I4_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I4_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I4_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I4_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I4_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I4_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I4_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I4_S7
#elif (NINTER_B8 == 8) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I8_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I8_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I8_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I8_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I8_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I8_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I8_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I8_S7
#elif (NINTER_B8 == 16) 
#define COMPXYZ_B8_S0 COMPXYZ_B8_I16_S0
#define COMPXYZ_B8_S1 COMPXYZ_B8_I16_S1
#define COMPXYZ_B8_S2 COMPXYZ_B8_I16_S2
#define COMPXYZ_B8_S3 COMPXYZ_B8_I16_S3
#define COMPXYZ_B8_S4 COMPXYZ_B8_I16_S4
#define COMPXYZ_B8_S5 COMPXYZ_B8_I16_S5
#define COMPXYZ_B8_S6 COMPXYZ_B8_I16_S6
#define COMPXYZ_B8_S7 COMPXYZ_B8_I16_S7
#else
#error Undefined NINTER_B8.
#endif


#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_S##s();				\
  } else if (B == 4) {				\
    COMPXYZ_B4_S##s();				\
  } else if (B == 8) {				\
    COMPXYZ_B8_S##s();				\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}


#define LOAD_M1(n)						\
  for (int iz = 0; iz < n; iz ++) {				\
    for (int iy = 0; iy < n; iy ++) {				\
      const real *Mptr0 = Mptr + (iz * ncpe + iy) * ncpe;	\
      for (int ix = 0; ix < n; ix ++) {				\
	Mj[iz][iy][ix] = Mptr0[ix];				\
      }								\
    }								\
  }

#define LOAD_M2(n)						\
  for (int iz = 0; iz < n; iz ++) {				\
    for (int iy = 0; iy < n; iy += 2) {				\
      const real *Mptr0 = Mptr + (iz * ncpe + (iy + 0)) * ncpe;	\
      const real *Mptr1 = Mptr0 + ncpe;				\
      for (int ix = 0; ix < n; ix ++) {				\
	Mj[iz][iy + 0][ix] = Mptr0[ix];				\
	Mj[iz][iy + 1][ix] = Mptr1[ix];				\
      }								\
    }								\
  }

#define LOAD_M4(n)						\
  for (int iz = 0; iz < n; iz ++) {				\
    for (int iy = 0; iy < n; iy += 4) {				\
      const real *Mptr0 = Mptr + (iz * ncpe + (iy + 0)) * ncpe;	\
      const real *Mptr1 = Mptr0 + ncpe;				\
      const real *Mptr2 = Mptr1 + ncpe;				\
      const real *Mptr3 = Mptr2 + ncpe;				\
      for (int ix = 0; ix < n; ix ++) {				\
	Mj[iz][iy + 0][ix] = Mptr0[ix];				\
	Mj[iz][iy + 1][ix] = Mptr1[ix];				\
	Mj[iz][iy + 2][ix] = Mptr2[ix];				\
	Mj[iz][iy + 3][ix] = Mptr3[ix];				\
      }								\
    }								\
  }

static load_M(const int B, const int ncpe, const real *Mptr, real Mj[2 * B + 4][2 * B + 4][2 * B + 4])
{
  if (B == 2) {
    LOAD_M2(8);
    //    LOAD_M1(8);
  } else if (B == 4) {
    LOAD_M2(12);
    //    LOAD_M1(12);
  } else if (B == 8) {
    LOAD_M4(20); // LOOP WAS VECTORIZED.
    //    LOAD_M2(20); // LOOP WAS VECTORIZED.
    //    LOAD_M1(20);
  } else {
    INFO("Undefined B=%d. Exit.\n", B);
    exit(EXIT_FAILURE);
  }
}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
#if(1)
    load_M(B, ncpe, Mptr, Mj);
#else
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
#endif
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(0); // s=0
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(1); // s=1
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(2); // s=2
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(3); // s=3
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(4); // s=4
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(5); // s=5
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(6); // s=6
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(7); // s=7
      Lptr += B * B * B;

      /* Point to next i */
      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9R)
/**************************************************************************/
/* Based on CPU9Q */

#if !defined(NINTER_B2)
//#define NINTER_B2 1
//#define NINTER_B2 2
//#define NINTER_B2 4
#define NINTER_B2 8
//#define NINTER_B2 16
#endif

#if !defined(NINTER_B4)
//#define NINTER_B4 1
//#define NINTER_B4 2
//#define NINTER_B4 4
#define NINTER_B4 8
//#define NINTER_B4 16
#endif

#if !defined(UNROLL_FACTOR_B2)
#define UNROLL_FACTOR_B2 0
//#define UNROLL_FACTOR_B2 8 // loop was not vectorized: low trip count.
#endif

#if !defined(UNROLL_FACTOR_B4)
//#define UNROLL_FACTOR_B4 0
//#define UNROLL_FACTOR_B4 4 // LOOP WAS VECTORIZED.
#define UNROLL_FACTOR_B4 8 // LOOP WAS VECTORIZED.
//#define UNROLL_FACTOR_B4 16 // loop was not vectorized: low trip count.
#endif

#if !defined(LEVEL_TO_SWITCH_B2_TO_B4) // 3 or more
//#define LEVEL_TO_SWITCH_B2_TO_B4 3
//#define LEVEL_TO_SWITCH_B2_TO_B4 4
#define LEVEL_TO_SWITCH_B2_TO_B4 5
#endif

static void cmp21(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp22(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp24(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp28(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
  }
}

static void cmp216(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}

static void cmp41(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp42(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp44(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp48(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
  }
}

static void cmp416(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) {
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}

#define CMP21(Kijoff0, Mjoff0)						\
  {									\
    cmp21(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP22(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp22(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP24(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp24(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP28(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp28(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP216(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp216(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }


#define CMP41(Kijoff0, Mjoff0)						\
  {									\
    cmp41(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP42(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp42(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP44(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp44(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP48(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp48(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP416(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp416(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }

#include "aux_CPU9P.h" // Created by aux_CPU9P.c

#if (NINTER_B2 == 1)
#define COMPXYZ_B2_S0 COMPXYZ_B2_I1_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I1_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I1_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I1_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I1_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I1_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I1_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I1_S7
#elif (NINTER_B2 == 2) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I2_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I2_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I2_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I2_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I2_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I2_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I2_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I2_S7
#elif (NINTER_B2 == 4) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I4_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I4_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I4_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I4_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I4_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I4_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I4_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I4_S7
#elif (NINTER_B2 == 8) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I8_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I8_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I8_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I8_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I8_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I8_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I8_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I8_S7
#elif (NINTER_B2 == 16) 
#define COMPXYZ_B2_S0 COMPXYZ_B2_I16_S0
#define COMPXYZ_B2_S1 COMPXYZ_B2_I16_S1
#define COMPXYZ_B2_S2 COMPXYZ_B2_I16_S2
#define COMPXYZ_B2_S3 COMPXYZ_B2_I16_S3
#define COMPXYZ_B2_S4 COMPXYZ_B2_I16_S4
#define COMPXYZ_B2_S5 COMPXYZ_B2_I16_S5
#define COMPXYZ_B2_S6 COMPXYZ_B2_I16_S6
#define COMPXYZ_B2_S7 COMPXYZ_B2_I16_S7
#else
#error Undefined NINTER_B2.
#endif

#if (NINTER_B4 == 1)
#define COMPXYZ_B4_S0 COMPXYZ_B4_I1_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I1_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I1_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I1_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I1_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I1_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I1_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I1_S7
#elif (NINTER_B4 == 2) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I2_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I2_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I2_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I2_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I2_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I2_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I2_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I2_S7
#elif (NINTER_B4 == 4) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I4_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I4_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I4_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I4_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I4_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I4_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I4_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I4_S7
#elif (NINTER_B4 == 8) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I8_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I8_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I8_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I8_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I8_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I8_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I8_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I8_S7
#elif (NINTER_B4 == 16) 
#define COMPXYZ_B4_S0 COMPXYZ_B4_I16_S0
#define COMPXYZ_B4_S1 COMPXYZ_B4_I16_S1
#define COMPXYZ_B4_S2 COMPXYZ_B4_I16_S2
#define COMPXYZ_B4_S3 COMPXYZ_B4_I16_S3
#define COMPXYZ_B4_S4 COMPXYZ_B4_I16_S4
#define COMPXYZ_B4_S5 COMPXYZ_B4_I16_S5
#define COMPXYZ_B4_S6 COMPXYZ_B4_I16_S6
#define COMPXYZ_B4_S7 COMPXYZ_B4_I16_S7
#else
#error Undefined NINTER_B4.
#endif

#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_S##s();				\
  } else if (B == 4) {				\
    COMPXYZ_B4_S##s();				\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  //  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
  //     loaded on memory explicitly like in GPU */
  //  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk; M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    //    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    //    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < cutoff; i ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.

      /* Set a pointer to K; K[j][i][k], where k=0; K will not be
	 loaded on memory explicitly like in GPU */
      real *Kptr = K + (j * cutoff + i) * 316 + 0;

      /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and sib=iz=iy=ix=0 */
      real *Lptr = L + ((((bx * cutoff + i) * 8 + 0) * B + 0) * B + 0) * B + 0;

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(0); // s=0
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(1); // s=1
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(2); // s=2
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(3); // s=3
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(4); // s=4
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(5); // s=5
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(6); // s=6
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(7); // s=7
      Lptr += B * B * B;

      //      /* Point to next i */
      //      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9Q)
/**************************************************************************/
/* Based on CPU9P */

#if !defined(NINTER)
//#define NINTER 1
//#define NINTER 2
//#define NINTER 4
#define NINTER 8
//#define NINTER 16
#endif


#if !defined(UNROLL_FACTOR_B2)
#define UNROLL_FACTOR_B2 0
//#define UNROLL_FACTOR_B2 8 // loop was not vectorized: low trip count.
#endif

#if !defined(UNROLL_FACTOR_B4)
//#define UNROLL_FACTOR_B4 0
//#define UNROLL_FACTOR_B4 4 // LOOP WAS VECTORIZED.
#define UNROLL_FACTOR_B4 8 // LOOP WAS VECTORIZED.
//#define UNROLL_FACTOR_B4 16 // loop was not vectorized: low trip count.
#endif

#if !defined(LEVEL_TO_SWITCH_B2_TO_B4) // 3 or more
#define LEVEL_TO_SWITCH_B2_TO_B4 3 // default
//#define LEVEL_TO_SWITCH_B2_TO_B4 4
//#define LEVEL_TO_SWITCH_B2_TO_B4 5
//#define LEVEL_TO_SWITCH_B2_TO_B4 6
#endif

static void cmp21(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp22(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp24(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp28(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=2
{
#if(0)
  real K[8], *M[8];
  K[0] = Kijptr[Kijoff0];
  K[1] = Kijptr[Kijoff1];
  K[2] = Kijptr[Kijoff2];
  K[3] = Kijptr[Kijoff3];
  K[4] = Kijptr[Kijoff4];
  K[5] = Kijptr[Kijoff5];
  K[6] = Kijptr[Kijoff6];
  K[7] = Kijptr[Kijoff7];
  M[0] = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  M[1] = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  M[2] = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  M[3] = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  M[4] = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  M[5] = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  M[6] = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  M[7] = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) {
    const int itmp = Mjshift[k];
#pragma simd    
    for(int l = 0; l < 8; l ++) { // SIMD LOOP WAS VECTORIZED.
      Lptr[k] += K[l] * (M[l])[itmp];
    }
  }
#else
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
  }
#endif
}

static void cmp216(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=2
{
#if(0)
  real K[16], *M[16];
  K[0] = Kijptr[Kijoff0];
  K[1] = Kijptr[Kijoff1];
  K[2] = Kijptr[Kijoff2];
  K[3] = Kijptr[Kijoff3];
  K[4] = Kijptr[Kijoff4];
  K[5] = Kijptr[Kijoff5];
  K[6] = Kijptr[Kijoff6];
  K[7] = Kijptr[Kijoff7];
  K[8] = Kijptr[Kijoff8];
  K[9] = Kijptr[Kijoff9];
  K[10] = Kijptr[Kijoff10];
  K[11] = Kijptr[Kijoff11];
  K[12] = Kijptr[Kijoff12];
  K[13] = Kijptr[Kijoff13];
  K[14] = Kijptr[Kijoff14];
  K[15] = Kijptr[Kijoff15];
  M[0] = Mjptr + Mjoff0;
  M[1] = Mjptr + Mjoff1;
  M[2] = Mjptr + Mjoff2;
  M[3] = Mjptr + Mjoff3;
  M[4] = Mjptr + Mjoff4;
  M[5] = Mjptr + Mjoff5;
  M[6] = Mjptr + Mjoff6;
  M[7] = Mjptr + Mjoff7;
  M[8] = Mjptr + Mjoff8;
  M[9] = Mjptr + Mjoff9;
  M[10] = Mjptr + Mjoff10;
  M[11] = Mjptr + Mjoff11;
  M[12] = Mjptr + Mjoff12;
  M[13] = Mjptr + Mjoff13;
  M[14] = Mjptr + Mjoff14;
  M[15] = Mjptr + Mjoff15;
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) {
    const int itmp = Mjshift[k];
#pragma simd
    for (int l = 0; l < 16; l ++) { // SIMD LOOP WAS VECTORIZED.
      Lptr[k] += K[l] * (M[l])[itmp];
    }
  }
#else
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B2)
  for (int k = 0; k < 8; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
#endif
}

static void cmp41(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp42(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
  }
}

static void cmp44(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
  }
}

static void cmp48(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=4
{
#if(0)//slow
  real K[8], *M[8];
  K[0] = Kijptr[Kijoff0];
  K[1] = Kijptr[Kijoff1];
  K[2] = Kijptr[Kijoff2];
  K[3] = Kijptr[Kijoff3];
  K[4] = Kijptr[Kijoff4];
  K[5] = Kijptr[Kijoff5];
  K[6] = Kijptr[Kijoff6];
  K[7] = Kijptr[Kijoff7];
  M[0] = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  M[1] = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  M[2] = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  M[3] = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  M[4] = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  M[5] = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  M[6] = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  M[7] = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) {
    const int itmp = Mjshift[k];
#pragma simd    
    for(int l = 0; l < 8; l ++) { // SIMD LOOP WAS VECTORIZED.
      Lptr[k] += K[l] * (M[l])[itmp];
    }
  }
#else
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) { // LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
  }
#endif
}

static void cmp416(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=4
{
#if(0)
  real K[16], *M[16];
  K[0] = Kijptr[Kijoff0];
  K[1] = Kijptr[Kijoff1];
  K[2] = Kijptr[Kijoff2];
  K[3] = Kijptr[Kijoff3];
  K[4] = Kijptr[Kijoff4];
  K[5] = Kijptr[Kijoff5];
  K[6] = Kijptr[Kijoff6];
  K[7] = Kijptr[Kijoff7];
  K[8] = Kijptr[Kijoff8];
  K[9] = Kijptr[Kijoff9];
  K[10] = Kijptr[Kijoff10];
  K[11] = Kijptr[Kijoff11];
  K[12] = Kijptr[Kijoff12];
  K[13] = Kijptr[Kijoff13];
  K[14] = Kijptr[Kijoff14];
  K[15] = Kijptr[Kijoff15];
  M[0] = Mjptr + Mjoff0;
  M[1] = Mjptr + Mjoff1;
  M[2] = Mjptr + Mjoff2;
  M[3] = Mjptr + Mjoff3;
  M[4] = Mjptr + Mjoff4;
  M[5] = Mjptr + Mjoff5;
  M[6] = Mjptr + Mjoff6;
  M[7] = Mjptr + Mjoff7;
  M[8] = Mjptr + Mjoff8;
  M[9] = Mjptr + Mjoff9;
  M[10] = Mjptr + Mjoff10;
  M[11] = Mjptr + Mjoff11;
  M[12] = Mjptr + Mjoff12;
  M[13] = Mjptr + Mjoff13;
  M[14] = Mjptr + Mjoff14;
  M[15] = Mjptr + Mjoff15;
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) {
    const int itmp = Mjshift[k];
#pragma simd
    for (int l = 0; l < 16; l ++) { // SIMD LOOP WAS VECTORIZED.
      Lptr[k] += K[l] * (M[l])[itmp];
    }
  }
#else
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma unroll(UNROLL_FACTOR_B4)
  for (int k = 0; k < 64; k ++) {
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
#endif
}

#define CMP21(Kijoff0, Mjoff0)						\
  {									\
    cmp21(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP22(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp22(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP24(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp24(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP28(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp28(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP216(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp216(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }


#define CMP41(Kijoff0, Mjoff0)						\
  {									\
    cmp41(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP42(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp42(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP44(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp44(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP48(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp48(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP416(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp416(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }

#include "aux_CPU9P.h" // Created by aux_CPU9P.c

#if (NINTER == 1)
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I1_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I1_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#elif (NINTER == 2) 
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I2_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I2_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#elif (NINTER == 4) 
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I4_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I4_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#elif (NINTER == 8) 
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I8_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I8_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#elif (NINTER == 16) 
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I16_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I16_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#else
#error Undefined NINTER.
#endif


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(0); // s=0
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(1); // s=1
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(2); // s=2
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(3); // s=3
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(4); // s=4
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(5); // s=5
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(6); // s=6
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(7); // s=7
      Lptr += B * B * B;

      /* Point to next i */
      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9P)
/**************************************************************************/
/* Based on CPU9N */

#if !defined(NINTER)
//#define NINTER 1
//#define NINTER 2
//#define NINTER 4
#define NINTER 8
//#define NINTER 16
#endif

static void cmp21(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma simd
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp22(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma simd
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
    //    const int itmp = Mjshift[k];
    //    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp];
  }
}

static void cmp24(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma simd
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
    //    const int itmp = Mjshift[k];
    //    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp];
  }
}

static void cmp28(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma simd
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    //    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]] + K4 * M4[Mjshift[k]] + K5 * M5[Mjshift[k]] + K6 * M6[Mjshift[k]] + K7 * M7[Mjshift[k]];
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
  }
}

static void cmp216(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=2
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma simd
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}

static void cmp41(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0,
		  const int Mjoff0) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
#pragma simd
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]];
  }
}

static void cmp42(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1,
		  const int Mjoff0, const int Mjoff1) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
#pragma simd
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]];
    //    const int itmp = Mjshift[k];
    //    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp];
  }
}

static void cmp44(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
#pragma simd
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]];
    //    const int itmp = Mjshift[k];
    //    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp];
  }
}

static void cmp48(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		  const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7,
		  const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
#pragma simd
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    //    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]] + K4 * M4[Mjshift[k]] + K5 * M5[Mjshift[k]] + K6 * M6[Mjshift[k]] + K7 * M7[Mjshift[k]];
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp];
  }
}

static void cmp416(real *Lptr, real *Kijptr, real *Mjptr, const int *Mjshift,
		   const int Kijoff0, const int Kijoff1, const int Kijoff2, const int Kijoff3, const int Kijoff4, const int Kijoff5, const int Kijoff6, const int Kijoff7, const int Kijoff8, const int Kijoff9, const int Kijoff10, const int Kijoff11, const int Kijoff12, const int Kijoff13, const int Kijoff14, const int Kijoff15,
		   const int Mjoff0, const int Mjoff1, const int Mjoff2, const int Mjoff3, const int Mjoff4, const int Mjoff5, const int Mjoff6, const int Mjoff7, const int Mjoff8, const int Mjoff9, const int Mjoff10, const int Mjoff11, const int Mjoff12, const int Mjoff13, const int Mjoff14, const int Mjoff15) // B=4
{
  real K0 = Kijptr[Kijoff0];
  real K1 = Kijptr[Kijoff1];
  real K2 = Kijptr[Kijoff2];
  real K3 = Kijptr[Kijoff3];
  real K4 = Kijptr[Kijoff4];
  real K5 = Kijptr[Kijoff5];
  real K6 = Kijptr[Kijoff6];
  real K7 = Kijptr[Kijoff7];
  real K8 = Kijptr[Kijoff8];
  real K9 = Kijptr[Kijoff9];
  real K10 = Kijptr[Kijoff10];
  real K11 = Kijptr[Kijoff11];
  real K12 = Kijptr[Kijoff12];
  real K13 = Kijptr[Kijoff13];
  real K14 = Kijptr[Kijoff14];
  real K15 = Kijptr[Kijoff15];
  real *M0 = Mjptr + Mjoff0; // &Mjptr[Mjoff0]
  real *M1 = Mjptr + Mjoff1; // &Mjptr[Mjoff1]
  real *M2 = Mjptr + Mjoff2; // &Mjptr[Mjoff2]
  real *M3 = Mjptr + Mjoff3; // &Mjptr[Mjoff3]
  real *M4 = Mjptr + Mjoff4; // &Mjptr[Mjoff4]
  real *M5 = Mjptr + Mjoff5; // &Mjptr[Mjoff5]
  real *M6 = Mjptr + Mjoff6; // &Mjptr[Mjoff6]
  real *M7 = Mjptr + Mjoff7; // &Mjptr[Mjoff7]
  real *M8 = Mjptr + Mjoff8; // &Mjptr[Mjoff8]
  real *M9 = Mjptr + Mjoff9; // &Mjptr[Mjoff9]
  real *M10 = Mjptr + Mjoff10; // &Mjptr[Mjoff10]
  real *M11 = Mjptr + Mjoff11; // &Mjptr[Mjoff11]
  real *M12 = Mjptr + Mjoff12; // &Mjptr[Mjoff12]
  real *M13 = Mjptr + Mjoff13; // &Mjptr[Mjoff13]
  real *M14 = Mjptr + Mjoff14; // &Mjptr[Mjoff14]
  real *M15 = Mjptr + Mjoff15; // &Mjptr[Mjoff15]
#pragma simd
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    //    Lptr[k] += K0 * M0[Mjshift[k]] + K1 * M1[Mjshift[k]] + K2 * M2[Mjshift[k]] + K3 * M3[Mjshift[k]] + K4 * M4[Mjshift[k]] + K5 * M5[Mjshift[k]] + K6 * M6[Mjshift[k]] + K7 * M7[Mjshift[k]] + K8 * M8[Mjshift[k]] + K9 * M9[Mjshift[k]] + K10 * M10[Mjshift[k]] + K11 * M11[Mjshift[k]] + K12 * M12[Mjshift[k]] + K13 * M13[Mjshift[k]] + K14 * M14[Mjshift[k]] + K15 * M15[Mjshift[k]];
    const int itmp = Mjshift[k];
    Lptr[k] += K0 * M0[itmp] + K1 * M1[itmp] + K2 * M2[itmp] + K3 * M3[itmp] + K4 * M4[itmp] + K5 * M5[itmp] + K6 * M6[itmp] + K7 * M7[itmp] + K8 * M8[itmp] + K9 * M9[itmp] + K10 * M10[itmp] + K11 * M11[itmp] + K12 * M12[itmp] + K13 * M13[itmp] + K14 * M14[itmp] + K15 * M15[itmp];
  }
}

#define CMP21(Kijoff0, Mjoff0)						\
  {									\
    cmp21(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP22(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp22(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP24(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp24(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP28(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp28(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP216(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp216(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }


#define CMP41(Kijoff0, Mjoff0)						\
  {									\
    cmp41(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Mjoff0);		\
  }
#define CMP42(Kijoff0, Kijoff1, Mjoff0, Mjoff1)				\
  {									\
    cmp42(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Mjoff0, Mjoff1); \
  }
#define CMP44(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3) \
  {									\
    cmp44(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Mjoff0, Mjoff1, Mjoff2, Mjoff3); \
  }
#define CMP48(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7) \
  {									\
    cmp48(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7); \
  }
#define CMP416(Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15)\
  {									\
    cmp416(Lptr, Kijptr, Mjptr, Mjshift, Kijoff0, Kijoff1, Kijoff2, Kijoff3, Kijoff4, Kijoff5, Kijoff6, Kijoff7, Kijoff8, Kijoff9, Kijoff10, Kijoff11, Kijoff12, Kijoff13, Kijoff14, Kijoff15, Mjoff0, Mjoff1, Mjoff2, Mjoff3, Mjoff4, Mjoff5, Mjoff6, Mjoff7, Mjoff8, Mjoff9, Mjoff10, Mjoff11, Mjoff12, Mjoff13, Mjoff14, Mjoff15); \
  }

#include "aux_CPU9P.h" // Created by aux_CPU9P.c

#if (NINTER == 1)
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I1_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I1_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#elif (NINTER == 2) 
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I2_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I2_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#elif (NINTER == 4) 
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I4_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I4_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#elif (NINTER == 8) 
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I8_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I8_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#elif (NINTER == 16) 
#define COMPXYZ(s)				\
  if (B == 2) {					\
    COMPXYZ_B2_I16_S##s();			\
  } else if (B == 4) {				\
    COMPXYZ_B4_I16_S##s();			\
  } else {					\
    INFO("Undefined B=%d. Exit.\n", B);		\
    exit(EXIT_FAILURE);				\
  }

#else
#error Undefined NINTER.
#endif


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(0); // s=0
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(1); // s=1
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(2); // s=2
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(3); // s=3
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(4); // s=4
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(5); // s=5
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(6); // s=6
      Lptr += B * B * B;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      COMPXYZ(7); // s=7
      Lptr += B * B * B;

      /* Point to next i */
      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9O)
/**************************************************************************/
/* Based on CPU9N */

static void cmp2(real *Lptr, const real Ktmp, real *Mjptr, const int *Mjshift) // B=2
{
#pragma simd
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
  }
}

static void cmp4(real *Lptr, const real Ktmp, real *Mjptr, const int *Mjshift) // B=4
{
#pragma simd
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
  }
}

#define CMP2(Kijoff_diff, Mjoff_diff)					\
  {									\
    Kijptr += Kijoff_diff;						\
    Mjptr += Mjoff_diff;						\
    cmp2(Lptr, *Kijptr, Mjptr, Mjshift);				\
  }

#define CMP4(Kijoff_diff, Mjoff_diff)					\
  {									\
    Kijptr += Kijoff_diff;						\
    Mjptr += Mjoff_diff;						\
    cmp4(Lptr, *Kijptr, Mjptr, Mjshift);				\
  }


/* Created by aux_CPU9N.c */
#define B2_COMPXYZ0() CMP2(57, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ1() CMP2(8, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ2() CMP2(50, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ3() CMP2(1, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ4() CMP2(56, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ5() CMP2(7, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ6() CMP2(49, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ7() CMP2(0, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B4_COMPXYZ0() CMP4(57, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ1() CMP4(8, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ2() CMP4(50, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ3() CMP4(1, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ4() CMP4(56, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ5() CMP4(7, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ6() CMP4(49, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ7() CMP4(0, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)



static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  //  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
  //     loaded on memory explicitly like in GPU */
  //  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    //    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    //    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < cutoff; i ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.

      /* Set a pointer to K; K[j][i][k], where k=0 ; K will not be
	 loaded on memory explicitly like in GPU */
      real *Kptr = K + (j * cutoff + i) * 316 + 0;

      /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and sib=iz=iy=ix=0 */
      real *Lptr = L + ((((bx * cutoff + i) * 8 + 0) * B + 0) * B + 0) * B + 0;

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ0();
      } else {
	B2_COMPXYZ0();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ1();
      } else {
	B2_COMPXYZ1();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ2();
      } else {
	B2_COMPXYZ2();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ3();
      } else {
	B2_COMPXYZ3();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ4();
      } else {
	B2_COMPXYZ4();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ5();
      } else {
	B2_COMPXYZ5();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ6();
      } else {
	B2_COMPXYZ6();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ7();
      } else {
	B2_COMPXYZ7();
      }
      //      Lptr += B * B * B; // next sibling index

      //      /* Point to next i */
      //      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9N)
/**************************************************************************/
/* Based on CPU9I */

static void cmp2(real *Lptr, const real Ktmp, real *Mjptr, const int *Mjshift) // B=2
{
#pragma simd
  for (int k = 0; k < 8; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
  }
}

static void cmp4(real *Lptr, const real Ktmp, real *Mjptr, const int *Mjshift) // B=4
{
#pragma simd
  for (int k = 0; k < 64; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
  }
}

#define CMP2(Kijoff_diff, Mjoff_diff)					\
  {									\
    Kijptr += Kijoff_diff;						\
    Mjptr += Mjoff_diff;						\
    cmp2(Lptr, *Kijptr, Mjptr, Mjshift);				\
  }

#define CMP4(Kijoff_diff, Mjoff_diff)					\
  {									\
    Kijptr += Kijoff_diff;						\
    Mjptr += Mjoff_diff;						\
    cmp4(Lptr, *Kijptr, Mjptr, Mjshift);				\
  }


/* Created by aux_CPU9N.c */
#define B2_COMPXYZ0() CMP2(57, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ1() CMP2(8, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ2() CMP2(50, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ3() CMP2(1, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 4); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ4() CMP2(56, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ5() CMP2(7, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ6() CMP2(49, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B2_COMPXYZ7() CMP2(0, 0); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 4); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(9, 19); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(2, 3); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1); CMP2(1, 1)
#define B4_COMPXYZ0() CMP4(57, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ1() CMP4(8, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ2() CMP4(50, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ3() CMP4(1, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 4); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ4() CMP4(56, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ5() CMP4(7, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ6() CMP4(49, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)
#define B4_COMPXYZ7() CMP4(0, 0); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 4); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(9, 79); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(2, 7); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1); CMP4(1, 1)



static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ0();
      } else {
	B2_COMPXYZ0();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ1();
      } else {
	B2_COMPXYZ1();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ2();
      } else {
	B2_COMPXYZ2();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ3();
      } else {
	B2_COMPXYZ3();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ4();
      } else {
	B2_COMPXYZ4();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ5();
      } else {
	B2_COMPXYZ5();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ6();
      } else {
	B2_COMPXYZ6();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ7();
      } else {
	B2_COMPXYZ7();
      }
      Lptr += B * B * B; // next sibling index

      /* Point to next i */
      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9M)
/**************************************************************************/
/* Based on CPU9I */

static void comp(const int B, real *Lptr, const real Ktmp, real *Mjptr, const int *Mjshift)
{
  //#pragma simd
  //  for (int k = 0; k < B * B * B; k ++) { // SIMD LOOP WAS VECTORIZED.
  //    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
  //  }
#if(0) // unrolling x2
#pragma simd
  for (int k = 0; k < B * B * B; k += 2) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
    Lptr[k + 1] += Ktmp * Mjptr[Mjshift[k + 1]];
  }
#else // unrolling x4
#pragma simd
  for (int k = 0; k < B * B * B; k += 4) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
    Lptr[k + 1] += Ktmp * Mjptr[Mjshift[k + 1]];
    Lptr[k + 2] += Ktmp * Mjptr[Mjshift[k + 2]];
    Lptr[k + 3] += Ktmp * Mjptr[Mjshift[k + 3]];
  }
#endif
}

#define COMP(Kijoff_diff, Mjoff_diff)					\
  {									\
    Kijptr += Kijoff_diff;						\
    Mjptr += Mjoff_diff;						\
    comp(B, Lptr, *Kijptr, Mjptr, Mjshift);				\
  }


/* Created by aux_CPU9F.c */
#define B4_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)

#define B2_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L;
       L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ0();
      } else {
	B2_COMPXYZ0();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ1();
      } else {
	B2_COMPXYZ1();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ2();
      } else {
	B2_COMPXYZ2();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ3();
      } else {
	B2_COMPXYZ3();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ4();
      } else {
	B2_COMPXYZ4();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ5();
      } else {
	B2_COMPXYZ5();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ6();
      } else {
	B2_COMPXYZ6();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ7();
      } else {
	B2_COMPXYZ7();
      }
      Lptr += B * B * B; // next sibling index

      /* Point to next i */
      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9L)
/**************************************************************************/
/* Based on CPU9I */

static void comp(const int B, real *Lptr, const real Ktmp, real *Mjptr, const int *Mjshift)
{
#pragma simd
  for (int k = 0; k < B * B * B; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
  }
}

#define COMP(Kijoff_diff, Mjoff_diff)					\
  {									\
    Kijptr += Kijoff_diff;						\
    Mjptr += Mjoff_diff;						\
    comp(B, Lptr, *Kijptr, Mjptr, Mjshift);				\
  }

/* Created by aux_CPU9F.c */
#define B4_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)

#define B2_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Allocate beforehand (make sense?) */
  real Mj[2 * B + 4][2 * B + 4][2 * B + 4];
  real Kij[316];

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    //    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L;
       L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Load Kij */
      for (int k = 0; k < 316; k ++) {
	Kij[k] = Kptr[k];
      }

      /* Point to next i */
      Kptr += 316;

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      //      Kijptr = Kptr;
      Kijptr = Kij;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ0();
      } else {
	B2_COMPXYZ0();
      }
      Lptr += B * B * B; // next sibling index

      //      Kijptr = Kptr;
      Kijptr = Kij;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ1();
      } else {
	B2_COMPXYZ1();
      }
      Lptr += B * B * B; // next sibling index

      //      Kijptr = Kptr;
      Kijptr = Kij;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ2();
      } else {
	B2_COMPXYZ2();
      }
      Lptr += B * B * B; // next sibling index

      //      Kijptr = Kptr;
      Kijptr = Kij;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ3();
      } else {
	B2_COMPXYZ3();
      }
      Lptr += B * B * B; // next sibling index

      //      Kijptr = Kptr;
      Kijptr = Kij;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ4();
      } else {
	B2_COMPXYZ4();
      }
      Lptr += B * B * B; // next sibling index

      //      Kijptr = Kptr;
      Kijptr = Kij;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ5();
      } else {
	B2_COMPXYZ5();
      }
      Lptr += B * B * B; // next sibling index

      //      Kijptr = Kptr;
      Kijptr = Kij;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ6();
      } else {
	B2_COMPXYZ6();
      }
      Lptr += B * B * B; // next sibling index

      //      Kijptr = Kptr;
      Kijptr = Kij;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ7();
      } else {
	B2_COMPXYZ7();
      }
      Lptr += B * B * B; // next sibling index

      //      /* Point to next i */
      //      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9K)
/**************************************************************************/
/* Based on CPU9J */

static void comp(const int B, real *Lptr[8], real Ktmp[8], real *Mjptr[8], int *Mjshift)
{
#ifdef __ICC
#pragma simd
#endif
  for (int k = 0; k < B * B * B; k ++) { // SIMD LOOP WAS VECTORIZED.
    int itmp = Mjshift[k];
    (Lptr[0])[k] += Ktmp[0] * (Mjptr[0])[itmp];
    (Lptr[1])[k] += Ktmp[1] * (Mjptr[1])[itmp];
    (Lptr[2])[k] += Ktmp[2] * (Mjptr[2])[itmp];
    (Lptr[3])[k] += Ktmp[3] * (Mjptr[3])[itmp];
    (Lptr[4])[k] += Ktmp[4] * (Mjptr[4])[itmp];
    (Lptr[5])[k] += Ktmp[5] * (Mjptr[5])[itmp];
    (Lptr[6])[k] += Ktmp[6] * (Mjptr[6])[itmp];
    (Lptr[7])[k] += Ktmp[7] * (Mjptr[7])[itmp];
  }
}

#define COMP(Kijoff_diff0, Kijoff_diff1, Kijoff_diff2, Kijoff_diff3, Kijoff_diff4, Kijoff_diff5, Kijoff_diff6, Kijoff_diff7, Mjoff_diff0, Mjoff_diff1, Mjoff_diff2, Mjoff_diff3, Mjoff_diff4, Mjoff_diff5, Mjoff_diff6, Mjoff_diff7) \
  {									\
    Kijptr[0] += Kijoff_diff0;						\
    Kijptr[1] += Kijoff_diff1;						\
    Kijptr[2] += Kijoff_diff2;						\
    Kijptr[3] += Kijoff_diff3;						\
    Kijptr[4] += Kijoff_diff4;						\
    Kijptr[5] += Kijoff_diff5;						\
    Kijptr[6] += Kijoff_diff6;						\
    Kijptr[7] += Kijoff_diff7;						\
    real Ktmp[8];							\
    Ktmp[0] = *(Kijptr[0]);						\
    Ktmp[1] = *(Kijptr[1]);						\
    Ktmp[2] = *(Kijptr[2]);						\
    Ktmp[3] = *(Kijptr[3]);						\
    Ktmp[4] = *(Kijptr[4]);						\
    Ktmp[5] = *(Kijptr[5]);						\
    Ktmp[6] = *(Kijptr[6]);						\
    Ktmp[7] = *(Kijptr[7]);						\
    Mjptr[0] += Mjoff_diff0;						\
    Mjptr[1] += Mjoff_diff1;						\
    Mjptr[2] += Mjoff_diff2;						\
    Mjptr[3] += Mjoff_diff3;						\
    Mjptr[4] += Mjoff_diff4;						\
    Mjptr[5] += Mjoff_diff5;						\
    Mjptr[6] += Mjoff_diff6;						\
    Mjptr[7] += Mjoff_diff7;						\
    comp(B, Lptr, Ktmp, Mjptr, Mjshift);				\
  }


/* Created by aux_CPU9J.c */
#define B4_COMPXYZ() COMP(57,8,50,1,56,7,49,0,0,0,0,0,0,0,0,0); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,9,9,9,9,9,9,9,79,79,79,79,79,79,79,79); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,1,1,2,1,1,1,7,1,1,1,7,1,1,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1); COMP(1,2,2,2,1,2,2,2,1,7,7,7,1,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,1,9,1,9,1,9,1,79,1,79,1,79,1,79,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,9,1,9,2,9,1,9,7,79,1,79,7,79,1,79); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(1,2,2,1,1,2,2,1,1,7,7,1,1,7,7,1); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(9,1,9,2,9,1,9,2,79,1,79,7,79,1,79,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,9,1,9,2,9,1,9,7,79,1,79,7,79,1,79); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(1,2,2,1,1,2,2,1,1,7,7,1,1,7,7,1); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(9,1,9,2,9,1,9,2,79,1,79,7,79,1,79,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,9,1,9,1,9,1,9,1,79,1,79,1,79,1,79); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,1); COMP(2,2,2,1,2,2,2,1,7,7,7,1,7,7,7,1); COMP(1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,1); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,1,1,2,1,1,1,2,1,1,1,7,1,1,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,9,9,9,9,9,9,9,79,79,79,79,79,79,79,79); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
#define B2_COMPXYZ() COMP(57,8,50,1,56,7,49,0,0,0,0,0,0,0,0,0); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,9,9,9,9,9,9,9,19,19,19,19,19,19,19,19); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,1,1,2,1,1,1,3,1,1,1,3,1,1,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1); COMP(1,2,2,2,1,2,2,2,1,3,3,3,1,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,1,9,1,9,1,9,1,19,1,19,1,19,1,19,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,9,1,9,2,9,1,9,3,19,1,19,3,19,1,19); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(1,2,2,1,1,2,2,1,1,3,3,1,1,3,3,1); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(9,1,9,2,9,1,9,2,19,1,19,3,19,1,19,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,9,1,9,2,9,1,9,3,19,1,19,3,19,1,19); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(1,2,2,1,1,2,2,1,1,3,3,1,1,3,3,1); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(9,1,9,2,9,1,9,2,19,1,19,3,19,1,19,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,9,1,9,1,9,1,9,1,19,1,19,1,19,1,19); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,1); COMP(2,2,2,1,2,2,2,1,3,3,3,1,3,3,3,1); COMP(1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,1); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,1,1,2,1,1,1,2,1,1,1,3,1,1,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,9,9,9,9,9,9,9,19,19,19,19,19,19,19,19); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)

static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]; independent of sibling index
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Allocate memories beforehand (make sense?) */
  real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // Not cached, perhaps
  real Kij[316]; // Maybe cached
  real *Lptr[8], *Kijptr[8], *Mjptr[8];

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    //    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx, sib=0..7, i=iz=iy=ix=0 */
    //    real *Lptr[8];
    for (int s = 0; s < 8; s ++) {
      Lptr[s] = L + ((((bx * cutoff + 0) * 8 + s) * B + 0) * B + 0) * B + 0;
    }

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Load Kij */
      for (int k = 0; k < 316; k ++) {
	Kij[k] = Kptr[k];
      }

      /* Point to next i */
      Kptr += 316;

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      //      real *Kijptr[8], *Mjptr[8];
      for (int s = 0; s < 8; s ++) {
	//	Kijptr[s] = Kptr; // initial value is independent of s
	Kijptr[s] = Kij; // initial value is independent of s
	Mjptr[s] = (real *)Mj; // initial value is independent of s
      }

      //      /* Point to next i */
      //      Kptr += 316;

      if (B == 4) {
#pragma inline
      	B4_COMPXYZ();
      } else {
#pragma inline
      	B2_COMPXYZ();
      }

      for (int s = 0; s < 8; s ++) {
	Lptr[s] += 8 * B * B * B; // point to next i
      }

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9J)
/**************************************************************************/
/* Based on CPU9I */

//static void comp(const int B, real *Lptr, const real Ktmp, real *Mjptr, const int *Mjshift)
//{
//#pragma simd
//  for (int k = 0; k < B * B * B; k ++) { // SIMD LOOP WAS VECTORIZED.
//    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
//  }
//}
//
//#define COMP(Kijoff_diff, Mjoff_diff)					\
//  {									\
//    Kijptr += Kijoff_diff;						\
//    Mjptr += Mjoff_diff;						\
//    comp(B, Lptr, *Kijptr, Mjptr, Mjshift);				\
//  }


static void comp(const int B, real *Lptr[8], real Ktmp[8], real *Mjptr[8], int *Mjshift)
{
#ifdef __ICC
#pragma simd
#endif
  for (int k = 0; k < B * B * B; k ++) { // SIMD LOOP WAS VECTORIZED.
    int itmp = Mjshift[k];
    (Lptr[0])[k] += Ktmp[0] * (Mjptr[0])[itmp];
    (Lptr[1])[k] += Ktmp[1] * (Mjptr[1])[itmp];
    (Lptr[2])[k] += Ktmp[2] * (Mjptr[2])[itmp];
    (Lptr[3])[k] += Ktmp[3] * (Mjptr[3])[itmp];
    (Lptr[4])[k] += Ktmp[4] * (Mjptr[4])[itmp];
    (Lptr[5])[k] += Ktmp[5] * (Mjptr[5])[itmp];
    (Lptr[6])[k] += Ktmp[6] * (Mjptr[6])[itmp];
    (Lptr[7])[k] += Ktmp[7] * (Mjptr[7])[itmp];
  }
}

#define COMP(Kijoff_diff0, Kijoff_diff1, Kijoff_diff2, Kijoff_diff3, Kijoff_diff4, Kijoff_diff5, Kijoff_diff6, Kijoff_diff7, Mjoff_diff0, Mjoff_diff1, Mjoff_diff2, Mjoff_diff3, Mjoff_diff4, Mjoff_diff5, Mjoff_diff6, Mjoff_diff7) \
  {									\
    Kijptr[0] += Kijoff_diff0;						\
    Kijptr[1] += Kijoff_diff1;						\
    Kijptr[2] += Kijoff_diff2;						\
    Kijptr[3] += Kijoff_diff3;						\
    Kijptr[4] += Kijoff_diff4;						\
    Kijptr[5] += Kijoff_diff5;						\
    Kijptr[6] += Kijoff_diff6;						\
    Kijptr[7] += Kijoff_diff7;						\
    real Ktmp[8];							\
    Ktmp[0] = *(Kijptr[0]);						\
    Ktmp[1] = *(Kijptr[1]);						\
    Ktmp[2] = *(Kijptr[2]);						\
    Ktmp[3] = *(Kijptr[3]);						\
    Ktmp[4] = *(Kijptr[4]);						\
    Ktmp[5] = *(Kijptr[5]);						\
    Ktmp[6] = *(Kijptr[6]);						\
    Ktmp[7] = *(Kijptr[7]);						\
    Mjptr[0] += Mjoff_diff0;						\
    Mjptr[1] += Mjoff_diff1;						\
    Mjptr[2] += Mjoff_diff2;						\
    Mjptr[3] += Mjoff_diff3;						\
    Mjptr[4] += Mjoff_diff4;						\
    Mjptr[5] += Mjoff_diff5;						\
    Mjptr[6] += Mjoff_diff6;						\
    Mjptr[7] += Mjoff_diff7;						\
    comp(B, Lptr, Ktmp, Mjptr, Mjshift);				\
  }


/* Created by aux_CPU9J.c */
#define B4_COMPXYZ() COMP(57,8,50,1,56,7,49,0,0,0,0,0,0,0,0,0); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,9,9,9,9,9,9,9,79,79,79,79,79,79,79,79); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,1,1,2,1,1,1,7,1,1,1,7,1,1,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1); COMP(1,2,2,2,1,2,2,2,1,7,7,7,1,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,1,9,1,9,1,9,1,79,1,79,1,79,1,79,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,9,1,9,2,9,1,9,7,79,1,79,7,79,1,79); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(1,2,2,1,1,2,2,1,1,7,7,1,1,7,7,1); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(9,1,9,2,9,1,9,2,79,1,79,7,79,1,79,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,9,1,9,2,9,1,9,7,79,1,79,7,79,1,79); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(1,2,2,1,1,2,2,1,1,7,7,1,1,7,7,1); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(9,1,9,2,9,1,9,2,79,1,79,7,79,1,79,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,9,1,9,1,9,1,9,1,79,1,79,1,79,1,79); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,7,1,7,1,7,1,7,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,1); COMP(2,2,2,1,2,2,2,1,7,7,7,1,7,7,7,1); COMP(1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,1); COMP(1,2,1,2,1,2,1,2,1,7,1,7,1,7,1,7); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,1,1,2,1,1,1,2,1,1,1,7,1,1,1,7); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,9,9,9,9,9,9,9,79,79,79,79,79,79,79,79); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
#define B2_COMPXYZ() COMP(57,8,50,1,56,7,49,0,0,0,0,0,0,0,0,0); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,9,9,9,9,9,9,9,19,19,19,19,19,19,19,19); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,1,1,2,1,1,1,3,1,1,1,3,1,1,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1); COMP(1,2,2,2,1,2,2,2,1,3,3,3,1,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,1,9,1,9,1,9,1,19,1,19,1,19,1,19,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,9,1,9,2,9,1,9,3,19,1,19,3,19,1,19); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(1,2,2,1,1,2,2,1,1,3,3,1,1,3,3,1); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(9,1,9,2,9,1,9,2,19,1,19,3,19,1,19,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,9,1,9,2,9,1,9,3,19,1,19,3,19,1,19); COMP(1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,4,1,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(1,2,2,1,1,2,2,1,1,3,3,1,1,3,3,1); COMP(1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(9,1,9,2,9,1,9,2,19,1,19,3,19,1,19,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,9,1,9,1,9,1,9,1,19,1,19,1,19,1,19); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,1); COMP(2,2,2,1,2,2,2,1,3,3,3,1,3,3,3,1); COMP(1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,1); COMP(1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3); COMP(1,1,1,1,1,1,1,1,1,4,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,4); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(1,1,1,2,1,1,1,2,1,1,1,3,1,1,1,3); COMP(1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(9,9,9,9,9,9,9,9,19,19,19,19,19,19,19,19); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); COMP(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)

static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]; independent of sibling index
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    //    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    //    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Set a pointer to L; L[chunk][i][sib][iz][iy][ix], where chunk=bx, sib=0..7, i=iz=iy=ix=0 */
    real *Lptr[8];
    for (int s = 0; s < 8; s ++) {
      Lptr[s] = L + ((((bx * cutoff + 0) * 8 + s) * B + 0) * B + 0) * B + 0;
    }

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      //      real *Kijptr, *Mjptr;
      //
      //      Kijptr = Kptr;
      //      Mjptr = (real *)Mj;

      real *Kijptr[8], *Mjptr[8];
      for (int s = 0; s < 8; s ++) {
	Kijptr[s] = Kptr; // initial value is independent of s
	Mjptr[s] = (real *)Mj; // initial value is independent of s
      }

      /* Point to next i */
      Kptr += 316;

      //      if (B == 4) {
      //	B4_COMPXYZ0();
      //      } else {
      //	B2_COMPXYZ0();
      //      }
      //      Lptr += B * B * B; // next sibling index
      //
      //      Kijptr = Kptr;
      //      Mjptr = (real *)Mj;
      //      if (B == 4) {
      //	B4_COMPXYZ1();
      //      } else {
      //	B2_COMPXYZ1();
      //      }
      //      Lptr += B * B * B; // next sibling index
      //
      //      Kijptr = Kptr;
      //      Mjptr = (real *)Mj;
      //      if (B == 4) {
      //	B4_COMPXYZ2();
      //      } else {
      //	B2_COMPXYZ2();
      //      }
      //      Lptr += B * B * B; // next sibling index
      //
      //      Kijptr = Kptr;
      //      Mjptr = (real *)Mj;
      //      if (B == 4) {
      //	B4_COMPXYZ3();
      //      } else {
      //	B2_COMPXYZ3();
      //      }
      //      Lptr += B * B * B; // next sibling index
      //
      //      Kijptr = Kptr;
      //      Mjptr = (real *)Mj;
      //      if (B == 4) {
      //	B4_COMPXYZ4();
      //      } else {
      //	B2_COMPXYZ4();
      //      }
      //      Lptr += B * B * B; // next sibling index
      //
      //      Kijptr = Kptr;
      //      Mjptr = (real *)Mj;
      //      if (B == 4) {
      //	B4_COMPXYZ5();
      //      } else {
      //	B2_COMPXYZ5();
      //      }
      //      Lptr += B * B * B; // next sibling index
      //
      //      Kijptr = Kptr;
      //      Mjptr = (real *)Mj;
      //      if (B == 4) {
      //	B4_COMPXYZ6();
      //      } else {
      //	B2_COMPXYZ6();
      //      }
      //      Lptr += B * B * B; // next sibling index
      //
      //      Kijptr = Kptr;
      //      Mjptr = (real *)Mj;
      //      if (B == 4) {
      //	B4_COMPXYZ7();
      //      } else {
      //	B2_COMPXYZ7();
      //      }
      //      Lptr += B * B * B; // next sibling index
      //
      //      /* Point to next i */
      //      Kptr += 316;

      if (B == 4) {
#pragma inline
      	B4_COMPXYZ();
      } else {
#pragma inline
      	B2_COMPXYZ();
      }

      for (int s = 0; s < 8; s ++) {
	Lptr[s] += 8 * B * B * B; // point to next i
      }

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9I)
/**************************************************************************/
/* Based on CPU9H */

//#define COMP(Kijoff_diff, Mjoff_diff)			\
//  Mjptr += Mjoff_diff;				\
//  Kijptr += Kijoff_diff;				\
//  Lij += (*Kijptr) * (*Mjptr);

#if(0)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  {							\
    Kijptr += Kijoff_diff;				\
    Mjptr += Mjoff_diff;				\
    const real Ktmp = *Kijptr;				\
    for (int k = 0; k < B * B * B; k ++) {		\
      Lptr[k] += Ktmp * Mjptr[Mjshift[k]];		\
    }							\
  }
#endif

static void comp(const int B, real *Lptr, const real Ktmp, real *Mjptr, const int *Mjshift)
{
#pragma simd
  for (int k = 0; k < B * B * B; k ++) { // SIMD LOOP WAS VECTORIZED.
    Lptr[k] += Ktmp * Mjptr[Mjshift[k]];
  }
}

#define COMP(Kijoff_diff, Mjoff_diff)					\
  {									\
    Kijptr += Kijoff_diff;						\
    Mjptr += Mjoff_diff;						\
    comp(B, Lptr, *Kijptr, Mjptr, Mjshift);				\
  }


/* Created by aux_CPU9F.c */
#define B4_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)

#define B2_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Shift for Mj */
  int Mjshift[B * B * B]; // Mjshift[# of targets with the same sibling index in a chunk]
  for (int iz = 0; iz < B; iz ++) {
    for (int iy = 0; iy < B; iy ++) {
      for (int ix = 0; ix < B; ix ++) {
	Mjshift[(iz * B + iy) * B + ix] = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);
      }
    }
  }

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    //    /* Set a pointer to L;
    //       L[chunk][i][iz][iy][ix][sib], where chunk=bx and i=iz=iy=ix=sib=0 */
    //    real *Lptr = L + ((((bx * cutoff + 0) * B + 0) * B + 0) * B + 0) * 8 + 0;

    /* Set a pointer to L;
       L[chunk][i][sib][iz][iy][ix], where chunk=bx and i=sib=iz=iy=ix=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * 8 + 0) * B + 0) * B + 0) * B + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	 S) and accumulate Lij(F) to Li(F) (reduction for j) */
      
      real *Kijptr, *Mjptr;

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ0();
      } else {
	B2_COMPXYZ0();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ1();
      } else {
	B2_COMPXYZ1();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ2();
      } else {
	B2_COMPXYZ2();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ3();
      } else {
	B2_COMPXYZ3();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ4();
      } else {
	B2_COMPXYZ4();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ5();
      } else {
	B2_COMPXYZ5();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ6();
      } else {
	B2_COMPXYZ6();
      }
      Lptr += B * B * B; // next sibling index

      Kijptr = Kptr;
      Mjptr = (real *)Mj;
      if (B == 4) {
	B4_COMPXYZ7();
      } else {
	B2_COMPXYZ7();
      }
      Lptr += B * B * B; // next sibling index

      /* Point to next i */
      Kptr += 316;

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9H)
/**************************************************************************/
/* Based on CPU9F */

#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Kijptr += Kijoff_diff;				\
  Lij += (*Kijptr) * (*Mjptr);

/* Created by aux_CPU9F.c */
#define B4_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)

#define B2_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0; K will not be
     loaded on memory explicitly like in GPU */
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L;
       L[chunk][i][iz][iy][ix][sib], where chunk=bx and i=iz=iy=ix=sib=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * B + 0) * B + 0) * B + 0) * 8 + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      //      /* Load Kij */
      //      real Kij[316]; // cached? --> Maybe
      //      for (int k = 0; k < 316; k ++) { // LOOP WAS VECTORIZED.
      //	Kij[k] = Kptr[k];
      //      }
      //
      //      /* Point to next i */
      //      Kptr += 316;

      /* Loop over target cells with the same sibling-index */
      for (int iz = 0; iz < B; iz ++) {
	for (int iy = 0; iy < B; iy ++) {
	  for (int ix = 0; ix < B; ix ++) {
	    
	    /* Offset */
	    const int Mjshift = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);

	    /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	       S) and accumulate Lij(F) to Li(F) (reduction for j) */
	    real Lij, *Kijptr, *Mjptr;
	    
	    /* Loop over sibling-indices of target cells */
	    Lij = ZERO;
	    //	    Kijptr = Kij;
	    Kijptr = Kptr;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ0();
	    } else {
	      B2_COMPXYZ0();
	    }
	    *Lptr += Lij;
	    Lptr ++;
	    
	    Lij = ZERO;
	    //	    Kijptr = Kij;
	    Kijptr = Kptr;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ1();
	    } else {
	      B2_COMPXYZ1();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    //	    Kijptr = Kij;
	    Kijptr = Kptr;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ2();
	    } else {
	      B2_COMPXYZ2();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    //	    Kijptr = Kij;
	    Kijptr = Kptr;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ3();
	    } else {
	      B2_COMPXYZ3();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    //	    Kijptr = Kij;
	    Kijptr = Kptr;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ4();
	    } else {
	      B2_COMPXYZ4();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    //	    Kijptr = Kij;
	    Kijptr = Kptr;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ5();
	    } else {
	      B2_COMPXYZ5();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    //	    Kijptr = Kij;
	    Kijptr = Kptr;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ6();
	    } else {
	      B2_COMPXYZ6();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    //	    Kijptr = Kij;
	    Kijptr = Kptr;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ7();
	    } else {
	      B2_COMPXYZ7();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	  } // ix
	} // iy
      } // iz

      /* Point to next i */
      Kptr += 316;

    } // i
  } // j
}

/**************************************************************************/
#elif defined(CPU9G)
/**************************************************************************/
/* Based on CPU9F */

#define NROWS 16

#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  for (int l = 0; l < NROWS; l ++) {			\
    Kijptr[l] += Kijoff_diff;				\
    Lij[l] += (*(Kijptr[l])) * (*Mjptr);		\
  }

/* Created by aux_CPU9F.c */
#define B4_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)

#define B2_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0*/
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    //    /* Set a pointer to L;
    //       L[chunk][i][iz][iy][ix][sib], where chunk=bx and i=iz=iy=ix=sib=0 */
    //    real *Lptr = L + ((((bx * cutoff + 0) * B + 0) * B + 0) * B + 0) * 8 + 0;

    /* Loop over rows i */
    //    for (int i = 0; i < cutoff; i ++) {
    for (int i = 0; i < cutoff; i += NROWS) { // unrolled 8x; assume cutoff%8 is 0

      /* Load Kij */
      //      real Kij[316]; // cached? --> Maybe
      //      for (int k = 0; k < 316; k ++) { // LOOP WAS VECTORIZED.
      //	Kij[k] = Kptr[k];
      //      }
      //
      //      /* Point to next i */
      //      Kptr += 316;

      real Kij[NROWS][316];
      for (int l = 0; l < NROWS; l ++) { 
	for (int k = 0; k < 316; k ++) { // LOOP WAS VECTORIZED.
	  Kij[l][k] = Kptr[k];
	}
	Kptr += 316; // point to next i
      }

      /* Set a pointer to L; L[chunk][i,..,i+NROWS-1][iz][iy][ix][sib],
	 where chunk=bx and iz=iy=ix=sib=0 */
      real *Lptr[NROWS];
      for (int l = 0; l < NROWS; l ++) {
	Lptr[l] = L + ((((bx * cutoff + i + l) * B + 0) * B + 0) * B + 0) * 8 + 0;
      }

      /* Loop over target cells with the same sibling-index */
      for (int iz = 0; iz < B; iz ++) {
	for (int iy = 0; iy < B; iy ++) {
	  for (int ix = 0; ix < B; ix ++) {
	    
	    /* Offset */
	    const int Mjshift = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);

	    /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	       S) and accumulate Lij(F) to Li(F) (reduction for j) */
	    //	    real Lij, *Kijptr, *Mjptr;
	    real Lij[NROWS], *Kijptr[NROWS], *Mjptr;
	    
	    /* Loop over sibling-indices of target cells */
	    for (int l = 0; l < NROWS; l ++) {
	      Lij[l] = ZERO;
	      Kijptr[l] = &(Kij[l][0]);
	    }
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ0();
	    } else {
	      B2_COMPXYZ0();
	    }
	    for (int l = 0; l < NROWS; l ++) {
	      *(Lptr[l]) += Lij[l];
	      (Lptr[l]) ++;
	    }

	    for (int l = 0; l < NROWS; l ++) {
	      Lij[l] = ZERO;
	      Kijptr[l] = &(Kij[l][0]);
	    }
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ1();
	    } else {
	      B2_COMPXYZ1();
	    }
	    for (int l = 0; l < NROWS; l ++) {
	      *(Lptr[l]) += Lij[l];
	      (Lptr[l]) ++;
	    }
	    
	    for (int l = 0; l < NROWS; l ++) {
	      Lij[l] = ZERO;
	      Kijptr[l] = &(Kij[l][0]);
	    }
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ2();
	    } else {
	      B2_COMPXYZ2();
	    }
	    for (int l = 0; l < NROWS; l ++) {
	      *(Lptr[l]) += Lij[l];
	      (Lptr[l]) ++;
	    }

	    for (int l = 0; l < NROWS; l ++) {
	      Lij[l] = ZERO;
	      Kijptr[l] = &(Kij[l][0]);
	    }
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ3();
	    } else {
	      B2_COMPXYZ3();
	    }
	    for (int l = 0; l < NROWS; l ++) {
	      *(Lptr[l]) += Lij[l];
	      (Lptr[l]) ++;
	    }

	    for (int l = 0; l < NROWS; l ++) {
	      Lij[l] = ZERO;
	      Kijptr[l] = &(Kij[l][0]);
	    }
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ4();
	    } else {
	      B2_COMPXYZ4();
	    }
	    for (int l = 0; l < NROWS; l ++) {
	      *(Lptr[l]) += Lij[l];
	      (Lptr[l]) ++;
	    }

	    for (int l = 0; l < NROWS; l ++) {
	      Lij[l] = ZERO;
	      Kijptr[l] = &(Kij[l][0]);
	    }
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ5();
	    } else {
	      B2_COMPXYZ5();
	    }
	    for (int l = 0; l < NROWS; l ++) {
	      *(Lptr[l]) += Lij[l];
	      (Lptr[l]) ++;
	    }

	    for (int l = 0; l < NROWS; l ++) {
	      Lij[l] = ZERO;
	      Kijptr[l] = &(Kij[l][0]);
	    }
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ6();
	    } else {
	      B2_COMPXYZ6();
	    }
	    for (int l = 0; l < NROWS; l ++) {
	      *(Lptr[l]) += Lij[l];
	      (Lptr[l]) ++;
	    }

	    for (int l = 0; l < NROWS; l ++) {
	      Lij[l] = ZERO;
	      Kijptr[l] = &(Kij[l][0]);
	    }
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ7();
	    } else {
	      B2_COMPXYZ7();
	    }
	    for (int l = 0; l < NROWS; l ++) {
	      *(Lptr[l]) += Lij[l];
	      (Lptr[l]) ++;
	    }

	  } // ix
	} // iy
      } // iz

    } // i
  } // j
}

/**************************************************************************/
#elif defined(CPU9F)
/**************************************************************************/
/* Based on CPU9D */

#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Kijptr += Kijoff_diff;				\
  Lij += (*Kijptr) * (*Mjptr);

/* Created by aux_CPU9F.c */
#define B4_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 4); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B4_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 4); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 79); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 7); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)

#define B2_COMPXYZ0() COMP(57, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ1() COMP(8, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ2() COMP(50, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ3() COMP(1, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 4); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ4() COMP(56, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ5() COMP(7, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ6() COMP(49, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)
#define B2_COMPXYZ7() COMP(0, 0); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 4); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(9, 19); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(2, 3); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1); COMP(1, 1)


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) along each edge of
     chunk for this level */
  const int ncpe = POW2(level) + 4; // =2*ncpec

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0*/
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][2*B*cz+iz][2*B*cy+iy][2*B*cx+ix], where j=ix=iy=iz=0 */
  real *Mptr = M + Mstart + ((0 * ncpe + (2 * B * cz + 0)) * ncpe + (2 * B * cy + 0)) * ncpe + (2 * B * cx + 0);

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[2 * B + 4][2 * B + 4][2 * B + 4]; // cached? --> NO
    
    for (int iz = 0; iz < 2 * B + 4; iz ++) {
      for (int iy = 0; iy < 2 * B + 4; iy ++) {
	for (int ix = 0; ix < 2 * B + 4; ix ++) {
	  Mj[iz][iy][ix] = Mptr[(iz * ncpe + iy) * ncpe + ix];
	}
      }
    }
    
    /* Point to next j */
    Mptr += ncpe * ncpe * ncpe;

    /* Set a pointer to L;
       L[chunk][i][iz][iy][ix][sib], where chunk=bx and i=iz=iy=ix=sib=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * B + 0) * B + 0) * B + 0) * 8 + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Load Kij */
      real Kij[316]; // cached? --> Maybe
      for (int k = 0; k < 316; k ++) { // LOOP WAS VECTORIZED.
	Kij[k] = Kptr[k];
      }
     
      /* Point to next i */
      Kptr += 316;

      /* Loop over target cells with the same sibling-index */
      for (int iz = 0; iz < B; iz ++) {
	for (int iy = 0; iy < B; iy ++) {
	  for (int ix = 0; ix < B; ix ++) {
	    
	    /* Offset */
	    const int Mjshift = ((2 * iz) * (2 * B + 4) + (2 * iy)) * (2 * B + 4) + (2 * ix);

	    /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	       S) and accumulate Lij(F) to Li(F) (reduction for j) */
	    real Lij, *Kijptr, *Mjptr;
	    
	    /* Loop over sibling-indices of target cells */
	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ0();
	    } else {
	      B2_COMPXYZ0();
	    }
	    *Lptr += Lij;
	    Lptr ++;
	    
	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ1();
	    } else {
	      B2_COMPXYZ1();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ2();
	    } else {
	      B2_COMPXYZ2();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ3();
	    } else {
	      B2_COMPXYZ3();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ4();
	    } else {
	      B2_COMPXYZ4();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ5();
	    } else {
	      B2_COMPXYZ5();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ6();
	    } else {
	      B2_COMPXYZ6();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ7();
	    } else {
	      B2_COMPXYZ7();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	  } // ix
	} // iy
      } // iz

    } // i
  } // j
}

/**************************************************************************/
#elif defined(CPU9E)
/**************************************************************************/
/* Based on CPU9D */

#define CMP4(i, M0, M1, M2, M3, M4, M5, M6, M7, K0, K1, K2, K3, K4, K5, K6, K7) \
  Mjptr[0] += M0; Mjptr[1] += M1; Mjptr[2] += M2; Mjptr[3] += M3; Mjptr[4] += M4; Mjptr[5] += M5; Mjptr[6] += M6; Mjptr[7] += M7; \
  Kijptr[0] += K0; Kijptr[1] += K1; Kijptr[2] += K2; Kijptr[3] += K3; Kijptr[4] += K4; Kijptr[5] += K5; Kijptr[6] += K6; Kijptr[7] += K7; \
  for (int s = 0; s < 8; s ++) { \
    Lij[s] += (*Kijptr[s]) * (*Mjptr[s]); \
  }

#define CMP2(i, M0, M1, M2, M3, M4, M5, M6, M7, K0, K1, K2, K3, K4, K5, K6, K7) \
  Mjptr[0] += M0; Mjptr[1] += M1; Mjptr[2] += M2; Mjptr[3] += M3; Mjptr[4] += M4; Mjptr[5] += M5; Mjptr[6] += M6; Mjptr[7] += M7; \
  Kijptr[0] += K0; Kijptr[1] += K1; Kijptr[2] += K2; Kijptr[3] += K3; Kijptr[4] += K4; Kijptr[5] += K5; Kijptr[6] += K6; Kijptr[7] += K7; \
  for (int s = 0; s < 8; s ++) { \
    Lij[s] += (*Kijptr[s]) * (*Mjptr[s]); \
  }




static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) with the same
     sibling-index along each direction for this level */
  const int ncpec = POW2(level - 1) + 2;

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0*/
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][s][B*cz+iz][B*cy+iy][B*cx+ix], where j=s=ix=iy=iz=0 */
  real *Mptr = M + Mstart + (((0 * 8 + 0) * ncpec + (B * cz + 0)) * ncpec + (B * cy + 0)) * ncpec + (B * cx + 0);

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[8][B + 2][B + 2][B + 2]; // cached? --> NO
    
    for (int s = 0; s < 8; s ++) { // sibling-index for source cells
      for (int iz = 0; iz < B + 2; iz ++) {
	for (int iy = 0; iy < B + 2; iy ++) {
	  for (int ix = 0; ix < B + 2; ix ++) {
	    Mj[s][iz][iy][ix] = Mptr[((s * ncpec + iz) * ncpec + iy) * ncpec + ix];
	  }
	}
      }
    }
    
    /* Point to next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

    /* Set a pointer to L;
       L[chunk][i][iz][iy][ix][sib], where chunk=bx and i=iz=iy=ix=sib=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * B + 0) * B + 0) * B + 0) * 8 + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Load Kij */
      real Kij[316]; // cached?
      for (int k = 0; k < 316; k ++) { // LOOP WAS VECTORIZED.
	Kij[k] = Kptr[k];
      }
     
      /* Point to next i */
      Kptr += 316;

      /* Loop over target cells with the same sibling-index */
      for (int iz = 0; iz < B; iz ++) {
	for (int iy = 0; iy < B; iy ++) {
	  for (int ix = 0; ix < B; ix ++) {
	    
	    /* Offset */
	    const int Mjshift = (iz * (B + 2) + iy) * (B + 2) + ix;

	    /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	       S) and accumulate Lij(F) to Li(F) (reduction for j) */

	    real Lij[8], *Kijptr[8], *Mjptr[8]; // should be placed outside of j?

	    for (int s = 0; s < 8; s ++) {
	      Lij[s] = ZERO;
	      Kijptr[s] = Kij;
	      Mjptr[s] = (real *)Mj + Mjshift;
	    }	    

	    if (B == 2)	{
	      CMP2(0,0,0,0,0,0,0,0,0,57,8,50,1,56,7,49,0); CMP2(1,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(2,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(3,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(4,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(5,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(6,-130,-130,-130,-130,-130,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(7,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(8,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(9,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(10,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(11,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(12,-382,-382,-382,-382,-382,-382,-382,-382,2,2,2,2,2,2,2,2); CMP2(13,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(14,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(15,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(16,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(17,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(18,-130,-130,-130,-130,-130,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(19,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(20,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(21,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(22,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(23,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(24,-382,-382,-382,-382,-382,-382,-382,-382,2,2,2,2,2,2,2,2); CMP2(25,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(26,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(27,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(28,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(29,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(30,-130,-130,-130,-130,-130,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(31,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(32,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(33,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(34,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(35,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(36,-330,-330,-330,-330,-330,-330,-330,-330,9,9,9,9,9,9,9,9); CMP2(37,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(38,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(39,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(40,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(41,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(42,-130,-130,-130,-130,-130,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(43,2,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(44,256,-255,-255,-255,2,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(45,-382,256,256,256,-382,256,256,256,2,1,1,1,2,1,1,1); CMP2(46,2,-255,-255,-255,256,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(47,256,256,256,256,2,256,256,256,1,1,1,1,1,1,1,1); CMP2(48,-130,-382,-382,-382,-130,-382,-382,-382,2,2,2,2,2,2,2,2); CMP2(49,2,256,2,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(50,256,-255,256,-255,2,-255,2,-255,1,1,1,1,1,1,1,1); CMP2(51,-382,256,-130,256,-382,256,-130,256,2,1,2,1,2,1,2,1); CMP2(52,256,-255,2,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(53,-255,256,256,256,-255,256,2,256,1,1,1,1,1,1,1,1); CMP2(54,256,-130,-382,-130,256,-130,-382,-130,1,2,2,2,1,2,2,2); CMP2(55,-255,256,2,256,-255,256,256,256,1,1,1,1,1,1,1,1); CMP2(56,256,-255,256,-255,256,-255,2,-255,1,1,1,1,1,1,1,1); CMP2(57,-130,256,-130,256,-130,256,-130,256,2,1,2,1,2,1,2,1); CMP2(58,256,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(59,-255,256,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(60,256,-382,256,-382,256,-382,256,-382,1,2,1,2,1,2,1,2); CMP2(61,-255,256,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(62,256,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(63,-442,256,-442,256,-442,256,-442,256,9,1,9,1,9,1,9,1); CMP2(64,256,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(65,-255,256,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(66,256,-130,256,-130,256,-130,256,-130,1,2,1,2,1,2,1,2); CMP2(67,-255,256,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(68,256,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(69,-130,256,-130,256,-130,256,-130,256,2,1,2,1,2,1,2,1); CMP2(70,2,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(71,256,256,-255,256,2,256,-255,256,1,1,1,1,1,1,1,1); CMP2(72,-382,-442,256,-442,-382,-442,256,-442,2,9,1,9,2,9,1,9); CMP2(73,2,256,-255,256,256,256,-255,256,1,1,1,1,1,1,1,1); CMP2(74,256,-255,256,-255,2,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(75,-130,256,-382,256,-130,256,-382,256,2,1,2,1,2,1,2,1); CMP2(76,2,-255,2,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(77,256,256,256,256,2,256,2,256,1,1,1,1,1,1,1,1); CMP2(78,-382,-130,-130,-130,-382,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(79,256,2,2,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(80,-255,256,256,-255,-255,2,2,-255,1,1,1,1,1,1,1,1); CMP2(81,256,-382,-382,256,256,-382,-382,256,1,2,2,1,1,2,2,1); CMP2(82,-255,2,2,-255,-255,256,256,-255,1,1,1,1,1,1,1,1); CMP2(83,256,256,256,256,256,2,2,256,1,1,1,1,1,1,1,1); CMP2(84,-130,-130,-130,-382,-130,-130,-130,-382,2,2,2,2,2,2,2,2); CMP2(85,256,2,256,2,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(86,-255,256,-255,256,-255,2,-255,2,1,1,1,1,1,1,1,1); CMP2(87,256,-382,256,-130,256,-382,256,-130,1,2,1,2,1,2,1,2); CMP2(88,-255,256,-255,2,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(89,256,-255,256,256,256,-255,256,2,1,1,1,1,1,1,1,1); CMP2(90,-330,256,-330,-382,-330,256,-330,-382,9,1,9,2,9,1,9,2); CMP2(91,256,-255,256,2,256,-255,256,256,1,1,1,1,1,1,1,1); CMP2(92,-255,256,-255,256,-255,256,-255,2,1,1,1,1,1,1,1,1); CMP2(93,256,-130,256,-130,256,-130,256,-130,1,2,1,2,1,2,1,2); CMP2(94,-255,256,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(95,256,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(96,-130,256,-130,256,-130,256,-130,256,2,1,2,1,2,1,2,1); CMP2(97,2,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(98,256,256,-255,256,2,256,-255,256,1,1,1,1,1,1,1,1); CMP2(99,-382,-330,256,-330,-382,-330,256,-330,2,9,1,9,2,9,1,9); CMP2(100,2,256,-255,256,256,256,-255,256,1,1,1,1,1,1,1,1); CMP2(101,256,-255,256,-255,2,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(102,-130,256,-382,256,-130,256,-382,256,2,1,2,1,2,1,2,1); CMP2(103,2,-255,2,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(104,256,256,256,256,2,256,2,256,1,1,1,1,1,1,1,1); CMP2(105,-382,-130,-130,-130,-382,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(106,256,2,2,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(107,-255,256,256,-255,-255,2,2,-255,1,1,1,1,1,1,1,1); CMP2(108,256,-382,-382,256,256,-382,-382,256,1,2,2,1,1,2,2,1); CMP2(109,-255,2,2,-255,-255,256,256,-255,1,1,1,1,1,1,1,1); CMP2(110,256,256,256,256,256,2,2,256,1,1,1,1,1,1,1,1); CMP2(111,-130,-130,-130,-382,-130,-130,-130,-382,2,2,2,2,2,2,2,2); CMP2(112,256,2,256,2,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(113,-255,256,-255,256,-255,2,-255,2,1,1,1,1,1,1,1,1); CMP2(114,256,-382,256,-130,256,-382,256,-130,1,2,1,2,1,2,1,2); CMP2(115,-255,256,-255,2,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(116,256,-255,256,256,256,-255,256,2,1,1,1,1,1,1,1,1); CMP2(117,-442,256,-442,-382,-442,256,-442,-382,9,1,9,2,9,1,9,2); CMP2(118,256,-255,256,2,256,-255,256,256,1,1,1,1,1,1,1,1); CMP2(119,-255,256,-255,256,-255,256,-255,2,1,1,1,1,1,1,1,1); CMP2(120,256,-130,256,-130,256,-130,256,-130,1,2,1,2,1,2,1,2); CMP2(121,-255,256,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(122,256,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(123,-130,256,-130,256,-130,256,-130,256,2,1,2,1,2,1,2,1); CMP2(124,256,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(125,-255,256,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(126,256,-442,256,-442,256,-442,256,-442,1,9,1,9,1,9,1,9); CMP2(127,-255,256,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(128,256,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(129,-382,256,-382,256,-382,256,-382,256,2,1,2,1,2,1,2,1); CMP2(130,256,-255,256,-255,256,-255,256,-255,1,1,1,1,1,1,1,1); CMP2(131,-255,256,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(132,256,-130,256,-130,256,-130,256,-130,1,2,1,2,1,2,1,2); CMP2(133,-255,2,-255,256,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(134,256,256,256,-255,256,2,256,-255,1,1,1,1,1,1,1,1); CMP2(135,-130,-382,-130,256,-130,-382,-130,256,2,2,2,1,2,2,2,1); CMP2(136,256,2,256,-255,256,256,256,-255,1,1,1,1,1,1,1,1); CMP2(137,-255,256,-255,256,-255,2,-255,256,1,1,1,1,1,1,1,1); CMP2(138,256,-130,256,-382,256,-130,256,-382,1,2,1,2,1,2,1,2); CMP2(139,-255,2,-255,2,-255,256,-255,256,1,1,1,1,1,1,1,1); CMP2(140,256,256,256,256,256,2,256,2,1,1,1,1,1,1,1,1); CMP2(141,-382,-382,-382,-130,-382,-382,-382,-130,2,2,2,2,2,2,2,2); CMP2(142,256,256,256,2,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(143,-255,-255,-255,256,-255,-255,-255,2,1,1,1,1,1,1,1,1); CMP2(144,256,256,256,-382,256,256,256,-382,1,1,1,2,1,1,1,2); CMP2(145,-255,-255,-255,2,-255,-255,-255,256,1,1,1,1,1,1,1,1); CMP2(146,256,256,256,256,256,256,256,2,1,1,1,1,1,1,1,1); CMP2(147,-130,-130,-130,-130,-130,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(148,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(149,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(150,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(151,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(152,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(153,-330,-330,-330,-330,-330,-330,-330,-330,9,9,9,9,9,9,9,9); CMP2(154,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(155,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(156,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(157,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(158,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(159,-130,-130,-130,-130,-130,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(160,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(161,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(162,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(163,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(164,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(165,-382,-382,-382,-382,-382,-382,-382,-382,2,2,2,2,2,2,2,2); CMP2(166,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(167,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(168,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(169,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(170,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(171,-130,-130,-130,-130,-130,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(172,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(173,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(174,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(175,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(176,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(177,-382,-382,-382,-382,-382,-382,-382,-382,2,2,2,2,2,2,2,2); CMP2(178,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(179,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(180,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(181,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(182,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(183,-130,-130,-130,-130,-130,-130,-130,-130,2,2,2,2,2,2,2,2); CMP2(184,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(185,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(186,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1); CMP2(187,-255,-255,-255,-255,-255,-255,-255,-255,1,1,1,1,1,1,1,1); CMP2(188,256,256,256,256,256,256,256,256,1,1,1,1,1,1,1,1);
	    } else {
	      CMP4(0,0,0,0,0,0,0,0,0,57,8,50,1,56,7,49,0); CMP4(1,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(2,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(3,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(4,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(5,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(6,-434,-434,-434,-434,-434,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(7,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(8,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(9,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(10,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(11,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(12,-1292,-1292,-1292,-1292,-1292,-1292,-1292,-1292,2,2,2,2,2,2,2,2); CMP4(13,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(14,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(15,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(16,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(17,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(18,-434,-434,-434,-434,-434,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(19,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(20,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(21,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(22,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(23,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(24,-1292,-1292,-1292,-1292,-1292,-1292,-1292,-1292,2,2,2,2,2,2,2,2); CMP4(25,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(26,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(27,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(28,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(29,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(30,-434,-434,-434,-434,-434,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(31,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(32,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(33,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(34,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(35,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(36,-1094,-1094,-1094,-1094,-1094,-1094,-1094,-1094,9,9,9,9,9,9,9,9); CMP4(37,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(38,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(39,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(40,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(41,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(42,-434,-434,-434,-434,-434,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(43,2,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(44,864,-863,-863,-863,2,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(45,-1292,864,864,864,-1292,864,864,864,2,1,1,1,2,1,1,1); CMP4(46,2,-863,-863,-863,864,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(47,864,864,864,864,2,864,864,864,1,1,1,1,1,1,1,1); CMP4(48,-434,-1292,-1292,-1292,-434,-1292,-1292,-1292,2,2,2,2,2,2,2,2); CMP4(49,2,864,2,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(50,864,-863,864,-863,2,-863,2,-863,1,1,1,1,1,1,1,1); CMP4(51,-1292,864,-434,864,-1292,864,-434,864,2,1,2,1,2,1,2,1); CMP4(52,864,-863,2,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(53,-863,864,864,864,-863,864,2,864,1,1,1,1,1,1,1,1); CMP4(54,864,-434,-1292,-434,864,-434,-1292,-434,1,2,2,2,1,2,2,2); CMP4(55,-863,864,2,864,-863,864,864,864,1,1,1,1,1,1,1,1); CMP4(56,864,-863,864,-863,864,-863,2,-863,1,1,1,1,1,1,1,1); CMP4(57,-434,864,-434,864,-434,864,-434,864,2,1,2,1,2,1,2,1); CMP4(58,864,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(59,-863,864,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(60,864,-1292,864,-1292,864,-1292,864,-1292,1,2,1,2,1,2,1,2); CMP4(61,-863,864,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(62,864,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(63,-1490,864,-1490,864,-1490,864,-1490,864,9,1,9,1,9,1,9,1); CMP4(64,864,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(65,-863,864,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(66,864,-434,864,-434,864,-434,864,-434,1,2,1,2,1,2,1,2); CMP4(67,-863,864,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(68,864,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(69,-434,864,-434,864,-434,864,-434,864,2,1,2,1,2,1,2,1); CMP4(70,2,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(71,864,864,-863,864,2,864,-863,864,1,1,1,1,1,1,1,1); CMP4(72,-1292,-1490,864,-1490,-1292,-1490,864,-1490,2,9,1,9,2,9,1,9); CMP4(73,2,864,-863,864,864,864,-863,864,1,1,1,1,1,1,1,1); CMP4(74,864,-863,864,-863,2,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(75,-434,864,-1292,864,-434,864,-1292,864,2,1,2,1,2,1,2,1); CMP4(76,2,-863,2,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(77,864,864,864,864,2,864,2,864,1,1,1,1,1,1,1,1); CMP4(78,-1292,-434,-434,-434,-1292,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(79,864,2,2,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(80,-863,864,864,-863,-863,2,2,-863,1,1,1,1,1,1,1,1); CMP4(81,864,-1292,-1292,864,864,-1292,-1292,864,1,2,2,1,1,2,2,1); CMP4(82,-863,2,2,-863,-863,864,864,-863,1,1,1,1,1,1,1,1); CMP4(83,864,864,864,864,864,2,2,864,1,1,1,1,1,1,1,1); CMP4(84,-434,-434,-434,-1292,-434,-434,-434,-1292,2,2,2,2,2,2,2,2); CMP4(85,864,2,864,2,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(86,-863,864,-863,864,-863,2,-863,2,1,1,1,1,1,1,1,1); CMP4(87,864,-1292,864,-434,864,-1292,864,-434,1,2,1,2,1,2,1,2); CMP4(88,-863,864,-863,2,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(89,864,-863,864,864,864,-863,864,2,1,1,1,1,1,1,1,1); CMP4(90,-1094,864,-1094,-1292,-1094,864,-1094,-1292,9,1,9,2,9,1,9,2); CMP4(91,864,-863,864,2,864,-863,864,864,1,1,1,1,1,1,1,1); CMP4(92,-863,864,-863,864,-863,864,-863,2,1,1,1,1,1,1,1,1); CMP4(93,864,-434,864,-434,864,-434,864,-434,1,2,1,2,1,2,1,2); CMP4(94,-863,864,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(95,864,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(96,-434,864,-434,864,-434,864,-434,864,2,1,2,1,2,1,2,1); CMP4(97,2,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(98,864,864,-863,864,2,864,-863,864,1,1,1,1,1,1,1,1); CMP4(99,-1292,-1094,864,-1094,-1292,-1094,864,-1094,2,9,1,9,2,9,1,9); CMP4(100,2,864,-863,864,864,864,-863,864,1,1,1,1,1,1,1,1); CMP4(101,864,-863,864,-863,2,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(102,-434,864,-1292,864,-434,864,-1292,864,2,1,2,1,2,1,2,1); CMP4(103,2,-863,2,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(104,864,864,864,864,2,864,2,864,1,1,1,1,1,1,1,1); CMP4(105,-1292,-434,-434,-434,-1292,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(106,864,2,2,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(107,-863,864,864,-863,-863,2,2,-863,1,1,1,1,1,1,1,1); CMP4(108,864,-1292,-1292,864,864,-1292,-1292,864,1,2,2,1,1,2,2,1); CMP4(109,-863,2,2,-863,-863,864,864,-863,1,1,1,1,1,1,1,1); CMP4(110,864,864,864,864,864,2,2,864,1,1,1,1,1,1,1,1); CMP4(111,-434,-434,-434,-1292,-434,-434,-434,-1292,2,2,2,2,2,2,2,2); CMP4(112,864,2,864,2,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(113,-863,864,-863,864,-863,2,-863,2,1,1,1,1,1,1,1,1); CMP4(114,864,-1292,864,-434,864,-1292,864,-434,1,2,1,2,1,2,1,2); CMP4(115,-863,864,-863,2,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(116,864,-863,864,864,864,-863,864,2,1,1,1,1,1,1,1,1); CMP4(117,-1490,864,-1490,-1292,-1490,864,-1490,-1292,9,1,9,2,9,1,9,2); CMP4(118,864,-863,864,2,864,-863,864,864,1,1,1,1,1,1,1,1); CMP4(119,-863,864,-863,864,-863,864,-863,2,1,1,1,1,1,1,1,1); CMP4(120,864,-434,864,-434,864,-434,864,-434,1,2,1,2,1,2,1,2); CMP4(121,-863,864,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(122,864,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(123,-434,864,-434,864,-434,864,-434,864,2,1,2,1,2,1,2,1); CMP4(124,864,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(125,-863,864,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(126,864,-1490,864,-1490,864,-1490,864,-1490,1,9,1,9,1,9,1,9); CMP4(127,-863,864,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(128,864,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(129,-1292,864,-1292,864,-1292,864,-1292,864,2,1,2,1,2,1,2,1); CMP4(130,864,-863,864,-863,864,-863,864,-863,1,1,1,1,1,1,1,1); CMP4(131,-863,864,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(132,864,-434,864,-434,864,-434,864,-434,1,2,1,2,1,2,1,2); CMP4(133,-863,2,-863,864,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(134,864,864,864,-863,864,2,864,-863,1,1,1,1,1,1,1,1); CMP4(135,-434,-1292,-434,864,-434,-1292,-434,864,2,2,2,1,2,2,2,1); CMP4(136,864,2,864,-863,864,864,864,-863,1,1,1,1,1,1,1,1); CMP4(137,-863,864,-863,864,-863,2,-863,864,1,1,1,1,1,1,1,1); CMP4(138,864,-434,864,-1292,864,-434,864,-1292,1,2,1,2,1,2,1,2); CMP4(139,-863,2,-863,2,-863,864,-863,864,1,1,1,1,1,1,1,1); CMP4(140,864,864,864,864,864,2,864,2,1,1,1,1,1,1,1,1); CMP4(141,-1292,-1292,-1292,-434,-1292,-1292,-1292,-434,2,2,2,2,2,2,2,2); CMP4(142,864,864,864,2,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(143,-863,-863,-863,864,-863,-863,-863,2,1,1,1,1,1,1,1,1); CMP4(144,864,864,864,-1292,864,864,864,-1292,1,1,1,2,1,1,1,2); CMP4(145,-863,-863,-863,2,-863,-863,-863,864,1,1,1,1,1,1,1,1); CMP4(146,864,864,864,864,864,864,864,2,1,1,1,1,1,1,1,1); CMP4(147,-434,-434,-434,-434,-434,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(148,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(149,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(150,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(151,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(152,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(153,-1094,-1094,-1094,-1094,-1094,-1094,-1094,-1094,9,9,9,9,9,9,9,9); CMP4(154,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(155,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(156,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(157,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(158,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(159,-434,-434,-434,-434,-434,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(160,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(161,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(162,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(163,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(164,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(165,-1292,-1292,-1292,-1292,-1292,-1292,-1292,-1292,2,2,2,2,2,2,2,2); CMP4(166,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(167,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(168,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(169,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(170,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(171,-434,-434,-434,-434,-434,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(172,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(173,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(174,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(175,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(176,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(177,-1292,-1292,-1292,-1292,-1292,-1292,-1292,-1292,2,2,2,2,2,2,2,2); CMP4(178,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(179,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(180,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(181,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(182,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(183,-434,-434,-434,-434,-434,-434,-434,-434,2,2,2,2,2,2,2,2); CMP4(184,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(185,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(186,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1); CMP4(187,-863,-863,-863,-863,-863,-863,-863,-863,1,1,1,1,1,1,1,1); CMP4(188,864,864,864,864,864,864,864,864,1,1,1,1,1,1,1,1);
	    }

	    for (int s = 0; s < 8; s ++) {
	      *Lptr += Lij[s];
	      Lptr ++;
	    }

	  } // ix
	} // iy
      } // iz

    } // i
  } // j
}

/**************************************************************************/
#elif defined(CPU9D)
/**************************************************************************/
/* Based on CPU9A */

#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Kijptr += Kijoff_diff;				\
  Lij += (*Kijptr) * (*Mjptr);

/* Created by aux_CPU9A.c */
#define B4_COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)

#define B2_COMPXYZ0() COMP(57, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ1() COMP(8, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ2() COMP(50, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ3() COMP(1, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ4() COMP(56, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ5() COMP(7, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ6() COMP(49, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ7() COMP(0, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)


static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) with the same
     sibling-index along each direction for this level */
  const int ncpec = POW2(level - 1) + 2;

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0*/
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][s][B*cz+iz][B*cy+iy][B*cx+ix], where j=s=ix=iy=iz=0 */
  real *Mptr = M + Mstart + (((0 * 8 + 0) * ncpec + (B * cz + 0)) * ncpec + (B * cy + 0)) * ncpec + (B * cx + 0);

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[8][B + 2][B + 2][B + 2]; // cached? --> NO
    
    for (int s = 0; s < 8; s ++) { // sibling-index for source cells
      for (int iz = 0; iz < B + 2; iz ++) {
	for (int iy = 0; iy < B + 2; iy ++) {
	  for (int ix = 0; ix < B + 2; ix ++) {
	    //	    Mj[s][iz][iy][ix] = Mptr[(((j * 8 + s) * ncpec + iz) * ncpec + iy) * ncpec + ix];
	    Mj[s][iz][iy][ix] = Mptr[((s * ncpec + iz) * ncpec + iy) * ncpec + ix];
	  }
	}
      }
    }
    
    /* Point to next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

    /* Set a pointer to L;
       L[chunk][i][iz][iy][ix][sib], where chunk=bx and i=iz=iy=ix=sib=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * B + 0) * B + 0) * B + 0) * 8 + 0;

    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Load Kij */
      real Kij[316]; // cached?
      for (int k = 0; k < 316; k ++) { // LOOP WAS VECTORIZED.
	Kij[k] = Kptr[k];
      }
     
      /* Point to next i */
      Kptr += 316;

      /* Loop over target cells with the same sibling-index */
      for (int iz = 0; iz < B; iz ++) {
	for (int iy = 0; iy < B; iy ++) {
	  for (int ix = 0; ix < B; ix ++) {
	    
	    /* Offset */
	    const int Mjshift = (iz * (B + 2) + iy) * (B + 2) + ix;

	    /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	       S) and accumulate Lij(F) to Li(F) (reduction for j) */
	    real Lij, *Kijptr, *Mjptr;
	    
	    /* Loop over sibling-indices of target cells */
	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ0();
	    } else {
	      B2_COMPXYZ0();
	    }
	    *Lptr += Lij;
	    Lptr ++;
	    
	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ1();
	    } else {
	      B2_COMPXYZ1();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ2();
	    } else {
	      B2_COMPXYZ2();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ3();
	    } else {
	      B2_COMPXYZ3();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ4();
	    } else {
	      B2_COMPXYZ4();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ5();
	    } else {
	      B2_COMPXYZ5();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ6();
	    } else {
	      B2_COMPXYZ6();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjshift;
	    if (B == 4)	{
	      B4_COMPXYZ7();
	    } else {
	      B2_COMPXYZ7();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	  } // ix
	} // iy
      } // iz

    } // i
  } // j
}

/**************************************************************************/
#elif defined(CPU9C)
/**************************************************************************/
/* Based on CPU9B */

#define LD(i, Kijoff, Mjoff)				\
  Kijtmp[i] = Kij[Kijoff];				\
  Mjtmp[i] = *(Mjptr + Mjoff);

#define B4_LOADXYZ0() LD(0, 57, 0); LD(1, 58, 864); LD(2, 59, 1); LD(3, 60, 865); LD(4, 61, 2); LD(5, 62, 866); LD(6, 64, 432); LD(7, 65, 1296); LD(8, 66, 433); LD(9, 67, 1297); LD(10, 68, 434); LD(11, 69, 1298); LD(12, 71, 6); LD(13, 72, 870); LD(14, 73, 7); LD(15, 74, 871); LD(16, 75, 8); LD(17, 76, 872); LD(18, 78, 438); LD(19, 79, 1302); LD(20, 80, 439); LD(21, 81, 1303); LD(22, 82, 440); LD(23, 83, 1304); LD(24, 85, 12); LD(25, 86, 876); LD(26, 87, 13); LD(27, 88, 877); LD(28, 89, 14); LD(29, 90, 878); LD(30, 92, 444); LD(31, 93, 1308); LD(32, 94, 445); LD(33, 95, 1309); LD(34, 96, 446); LD(35, 97, 1310); LD(36, 106, 216); LD(37, 107, 1080); LD(38, 108, 217); LD(39, 109, 1081); LD(40, 110, 218); LD(41, 111, 1082); LD(42, 113, 648); LD(43, 114, 650); LD(44, 115, 1514); LD(45, 117, 222); LD(46, 118, 224); LD(47, 119, 1088); LD(48, 121, 654); LD(49, 122, 656); LD(50, 123, 1520); LD(51, 125, 228); LD(52, 126, 1092); LD(53, 127, 229); LD(54, 128, 1093); LD(55, 129, 230); LD(56, 130, 1094); LD(57, 132, 660); LD(58, 133, 1524); LD(59, 134, 661); LD(60, 135, 1525); LD(61, 136, 662); LD(62, 137, 1526); LD(63, 146, 36); LD(64, 147, 900); LD(65, 148, 37); LD(66, 149, 901); LD(67, 150, 38); LD(68, 151, 902); LD(69, 153, 468); LD(70, 154, 470); LD(71, 155, 1334); LD(72, 157, 42); LD(73, 158, 44); LD(74, 159, 908); LD(75, 161, 474); LD(76, 162, 476); LD(77, 163, 1340); LD(78, 165, 48); LD(79, 166, 912); LD(80, 167, 49); LD(81, 168, 913); LD(82, 169, 50); LD(83, 170, 914); LD(84, 172, 480); LD(85, 173, 1344); LD(86, 174, 481); LD(87, 175, 1345); LD(88, 176, 482); LD(89, 177, 1346); LD(90, 186, 252); LD(91, 187, 1116); LD(92, 188, 253); LD(93, 189, 1117); LD(94, 190, 254); LD(95, 191, 1118); LD(96, 193, 684); LD(97, 194, 686); LD(98, 195, 1550); LD(99, 197, 258); LD(100, 198, 260); LD(101, 199, 1124); LD(102, 201, 690); LD(103, 202, 692); LD(104, 203, 1556); LD(105, 205, 264); LD(106, 206, 1128); LD(107, 207, 265); LD(108, 208, 1129); LD(109, 209, 266); LD(110, 210, 1130); LD(111, 212, 696); LD(112, 213, 1560); LD(113, 214, 697); LD(114, 215, 1561); LD(115, 216, 698); LD(116, 217, 1562); LD(117, 226, 72); LD(118, 227, 936); LD(119, 228, 73); LD(120, 229, 937); LD(121, 230, 74); LD(122, 231, 938); LD(123, 233, 504); LD(124, 234, 1368); LD(125, 235, 505); LD(126, 236, 1369); LD(127, 237, 506); LD(128, 238, 1370); LD(129, 240, 78); LD(130, 241, 942); LD(131, 242, 79); LD(132, 243, 943); LD(133, 244, 80); LD(134, 245, 944); LD(135, 247, 510); LD(136, 248, 1374); LD(137, 249, 511); LD(138, 250, 1375); LD(139, 251, 512); LD(140, 252, 1376); LD(141, 254, 84); LD(142, 255, 948); LD(143, 256, 85); LD(144, 257, 949); LD(145, 258, 86); LD(146, 259, 950); LD(147, 261, 516); LD(148, 262, 1380); LD(149, 263, 517); LD(150, 264, 1381); LD(151, 265, 518); LD(152, 266, 1382); LD(153, 275, 288); LD(154, 276, 1152); LD(155, 277, 289); LD(156, 278, 1153); LD(157, 279, 290); LD(158, 280, 1154); LD(159, 282, 720); LD(160, 283, 1584); LD(161, 284, 721); LD(162, 285, 1585); LD(163, 286, 722); LD(164, 287, 1586); LD(165, 289, 294); LD(166, 290, 1158); LD(167, 291, 295); LD(168, 292, 1159); LD(169, 293, 296); LD(170, 294, 1160); LD(171, 296, 726); LD(172, 297, 1590); LD(173, 298, 727); LD(174, 299, 1591); LD(175, 300, 728); LD(176, 301, 1592); LD(177, 303, 300); LD(178, 304, 1164); LD(179, 305, 301); LD(180, 306, 1165); LD(181, 307, 302); LD(182, 308, 1166); LD(183, 310, 732); LD(184, 311, 1596); LD(185, 312, 733); LD(186, 313, 1597); LD(187, 314, 734); LD(188, 315, 1598)
#define B4_LOADXYZ1() LD(0, 8, 0); LD(1, 9, 864); LD(2, 10, 1); LD(3, 11, 865); LD(4, 12, 2); LD(5, 13, 866); LD(6, 15, 432); LD(7, 16, 1296); LD(8, 17, 433); LD(9, 18, 1297); LD(10, 19, 434); LD(11, 20, 1298); LD(12, 22, 6); LD(13, 23, 870); LD(14, 24, 7); LD(15, 25, 871); LD(16, 26, 8); LD(17, 27, 872); LD(18, 29, 438); LD(19, 30, 1302); LD(20, 31, 439); LD(21, 32, 1303); LD(22, 33, 440); LD(23, 34, 1304); LD(24, 36, 12); LD(25, 37, 876); LD(26, 38, 13); LD(27, 39, 877); LD(28, 40, 14); LD(29, 41, 878); LD(30, 43, 444); LD(31, 44, 1308); LD(32, 45, 445); LD(33, 46, 1309); LD(34, 47, 446); LD(35, 48, 1310); LD(36, 57, 216); LD(37, 58, 1080); LD(38, 59, 217); LD(39, 60, 1081); LD(40, 61, 218); LD(41, 62, 1082); LD(42, 64, 648); LD(43, 65, 1512); LD(44, 66, 649); LD(45, 67, 1513); LD(46, 68, 650); LD(47, 69, 1514); LD(48, 71, 222); LD(49, 72, 1086); LD(50, 73, 223); LD(51, 74, 1087); LD(52, 75, 224); LD(53, 76, 1088); LD(54, 78, 654); LD(55, 79, 1518); LD(56, 80, 655); LD(57, 81, 1519); LD(58, 82, 656); LD(59, 83, 1520); LD(60, 85, 228); LD(61, 86, 1092); LD(62, 87, 229); LD(63, 88, 1093); LD(64, 89, 230); LD(65, 90, 1094); LD(66, 92, 660); LD(67, 93, 1524); LD(68, 94, 661); LD(69, 95, 1525); LD(70, 96, 662); LD(71, 97, 1526); LD(72, 106, 36); LD(73, 107, 900); LD(74, 108, 37); LD(75, 109, 901); LD(76, 110, 38); LD(77, 111, 902); LD(78, 113, 468); LD(79, 114, 470); LD(80, 115, 1334); LD(81, 117, 42); LD(82, 118, 44); LD(83, 119, 908); LD(84, 121, 474); LD(85, 122, 476); LD(86, 123, 1340); LD(87, 125, 48); LD(88, 126, 912); LD(89, 127, 49); LD(90, 128, 913); LD(91, 129, 50); LD(92, 130, 914); LD(93, 132, 480); LD(94, 133, 1344); LD(95, 134, 481); LD(96, 135, 1345); LD(97, 136, 482); LD(98, 137, 1346); LD(99, 146, 252); LD(100, 147, 1116); LD(101, 148, 253); LD(102, 149, 1117); LD(103, 150, 254); LD(104, 151, 1118); LD(105, 153, 684); LD(106, 154, 686); LD(107, 155, 1550); LD(108, 157, 258); LD(109, 158, 260); LD(110, 159, 1124); LD(111, 161, 690); LD(112, 162, 692); LD(113, 163, 1556); LD(114, 165, 264); LD(115, 166, 1128); LD(116, 167, 265); LD(117, 168, 1129); LD(118, 169, 266); LD(119, 170, 1130); LD(120, 172, 696); LD(121, 173, 1560); LD(122, 174, 697); LD(123, 175, 1561); LD(124, 176, 698); LD(125, 177, 1562); LD(126, 186, 72); LD(127, 187, 936); LD(128, 188, 73); LD(129, 189, 937); LD(130, 190, 74); LD(131, 191, 938); LD(132, 193, 504); LD(133, 194, 506); LD(134, 195, 1370); LD(135, 197, 78); LD(136, 198, 80); LD(137, 199, 944); LD(138, 201, 510); LD(139, 202, 512); LD(140, 203, 1376); LD(141, 205, 84); LD(142, 206, 948); LD(143, 207, 85); LD(144, 208, 949); LD(145, 209, 86); LD(146, 210, 950); LD(147, 212, 516); LD(148, 213, 1380); LD(149, 214, 517); LD(150, 215, 1381); LD(151, 216, 518); LD(152, 217, 1382); LD(153, 226, 288); LD(154, 227, 1152); LD(155, 228, 289); LD(156, 229, 1153); LD(157, 230, 290); LD(158, 231, 1154); LD(159, 233, 720); LD(160, 234, 1584); LD(161, 235, 721); LD(162, 236, 1585); LD(163, 237, 722); LD(164, 238, 1586); LD(165, 240, 294); LD(166, 241, 1158); LD(167, 242, 295); LD(168, 243, 1159); LD(169, 244, 296); LD(170, 245, 1160); LD(171, 247, 726); LD(172, 248, 1590); LD(173, 249, 727); LD(174, 250, 1591); LD(175, 251, 728); LD(176, 252, 1592); LD(177, 254, 300); LD(178, 255, 1164); LD(179, 256, 301); LD(180, 257, 1165); LD(181, 258, 302); LD(182, 259, 1166); LD(183, 261, 732); LD(184, 262, 1596); LD(185, 263, 733); LD(186, 264, 1597); LD(187, 265, 734); LD(188, 266, 1598)
#define B4_LOADXYZ2() LD(0, 50, 0); LD(1, 51, 864); LD(2, 52, 1); LD(3, 53, 865); LD(4, 54, 2); LD(5, 55, 866); LD(6, 57, 432); LD(7, 58, 1296); LD(8, 59, 433); LD(9, 60, 1297); LD(10, 61, 434); LD(11, 62, 1298); LD(12, 64, 6); LD(13, 65, 870); LD(14, 66, 7); LD(15, 67, 871); LD(16, 68, 8); LD(17, 69, 872); LD(18, 71, 438); LD(19, 72, 1302); LD(20, 73, 439); LD(21, 74, 1303); LD(22, 75, 440); LD(23, 76, 1304); LD(24, 78, 12); LD(25, 79, 876); LD(26, 80, 13); LD(27, 81, 877); LD(28, 82, 14); LD(29, 83, 878); LD(30, 85, 444); LD(31, 86, 1308); LD(32, 87, 445); LD(33, 88, 1309); LD(34, 89, 446); LD(35, 90, 1310); LD(36, 99, 216); LD(37, 100, 1080); LD(38, 101, 217); LD(39, 102, 1081); LD(40, 103, 218); LD(41, 104, 1082); LD(42, 106, 648); LD(43, 107, 1512); LD(44, 108, 649); LD(45, 109, 1513); LD(46, 110, 650); LD(47, 111, 1514); LD(48, 113, 222); LD(49, 114, 224); LD(50, 115, 1088); LD(51, 117, 654); LD(52, 118, 656); LD(53, 119, 1520); LD(54, 121, 228); LD(55, 122, 230); LD(56, 123, 1094); LD(57, 125, 660); LD(58, 126, 1524); LD(59, 127, 661); LD(60, 128, 1525); LD(61, 129, 662); LD(62, 130, 1526); LD(63, 139, 36); LD(64, 140, 900); LD(65, 141, 37); LD(66, 142, 901); LD(67, 143, 38); LD(68, 144, 902); LD(69, 146, 468); LD(70, 147, 1332); LD(71, 148, 469); LD(72, 149, 1333); LD(73, 150, 470); LD(74, 151, 1334); LD(75, 153, 42); LD(76, 154, 44); LD(77, 155, 908); LD(78, 157, 474); LD(79, 158, 476); LD(80, 159, 1340); LD(81, 161, 48); LD(82, 162, 50); LD(83, 163, 914); LD(84, 165, 480); LD(85, 166, 1344); LD(86, 167, 481); LD(87, 168, 1345); LD(88, 169, 482); LD(89, 170, 1346); LD(90, 179, 252); LD(91, 180, 1116); LD(92, 181, 253); LD(93, 182, 1117); LD(94, 183, 254); LD(95, 184, 1118); LD(96, 186, 684); LD(97, 187, 1548); LD(98, 188, 685); LD(99, 189, 1549); LD(100, 190, 686); LD(101, 191, 1550); LD(102, 193, 258); LD(103, 194, 260); LD(104, 195, 1124); LD(105, 197, 690); LD(106, 198, 692); LD(107, 199, 1556); LD(108, 201, 264); LD(109, 202, 266); LD(110, 203, 1130); LD(111, 205, 696); LD(112, 206, 1560); LD(113, 207, 697); LD(114, 208, 1561); LD(115, 209, 698); LD(116, 210, 1562); LD(117, 219, 72); LD(118, 220, 936); LD(119, 221, 73); LD(120, 222, 937); LD(121, 223, 74); LD(122, 224, 938); LD(123, 226, 504); LD(124, 227, 1368); LD(125, 228, 505); LD(126, 229, 1369); LD(127, 230, 506); LD(128, 231, 1370); LD(129, 233, 78); LD(130, 234, 942); LD(131, 235, 79); LD(132, 236, 943); LD(133, 237, 80); LD(134, 238, 944); LD(135, 240, 510); LD(136, 241, 1374); LD(137, 242, 511); LD(138, 243, 1375); LD(139, 244, 512); LD(140, 245, 1376); LD(141, 247, 84); LD(142, 248, 948); LD(143, 249, 85); LD(144, 250, 949); LD(145, 251, 86); LD(146, 252, 950); LD(147, 254, 516); LD(148, 255, 1380); LD(149, 256, 517); LD(150, 257, 1381); LD(151, 258, 518); LD(152, 259, 1382); LD(153, 268, 288); LD(154, 269, 1152); LD(155, 270, 289); LD(156, 271, 1153); LD(157, 272, 290); LD(158, 273, 1154); LD(159, 275, 720); LD(160, 276, 1584); LD(161, 277, 721); LD(162, 278, 1585); LD(163, 279, 722); LD(164, 280, 1586); LD(165, 282, 294); LD(166, 283, 1158); LD(167, 284, 295); LD(168, 285, 1159); LD(169, 286, 296); LD(170, 287, 1160); LD(171, 289, 726); LD(172, 290, 1590); LD(173, 291, 727); LD(174, 292, 1591); LD(175, 293, 728); LD(176, 294, 1592); LD(177, 296, 300); LD(178, 297, 1164); LD(179, 298, 301); LD(180, 299, 1165); LD(181, 300, 302); LD(182, 301, 1166); LD(183, 303, 732); LD(184, 304, 1596); LD(185, 305, 733); LD(186, 306, 1597); LD(187, 307, 734); LD(188, 308, 1598)
#define B4_LOADXYZ3() LD(0, 1, 0); LD(1, 2, 864); LD(2, 3, 1); LD(3, 4, 865); LD(4, 5, 2); LD(5, 6, 866); LD(6, 8, 432); LD(7, 9, 1296); LD(8, 10, 433); LD(9, 11, 1297); LD(10, 12, 434); LD(11, 13, 1298); LD(12, 15, 6); LD(13, 16, 870); LD(14, 17, 7); LD(15, 18, 871); LD(16, 19, 8); LD(17, 20, 872); LD(18, 22, 438); LD(19, 23, 1302); LD(20, 24, 439); LD(21, 25, 1303); LD(22, 26, 440); LD(23, 27, 1304); LD(24, 29, 12); LD(25, 30, 876); LD(26, 31, 13); LD(27, 32, 877); LD(28, 33, 14); LD(29, 34, 878); LD(30, 36, 444); LD(31, 37, 1308); LD(32, 38, 445); LD(33, 39, 1309); LD(34, 40, 446); LD(35, 41, 1310); LD(36, 50, 216); LD(37, 51, 1080); LD(38, 52, 217); LD(39, 53, 1081); LD(40, 54, 218); LD(41, 55, 1082); LD(42, 57, 648); LD(43, 58, 1512); LD(44, 59, 649); LD(45, 60, 1513); LD(46, 61, 650); LD(47, 62, 1514); LD(48, 64, 222); LD(49, 65, 1086); LD(50, 66, 223); LD(51, 67, 1087); LD(52, 68, 224); LD(53, 69, 1088); LD(54, 71, 654); LD(55, 72, 1518); LD(56, 73, 655); LD(57, 74, 1519); LD(58, 75, 656); LD(59, 76, 1520); LD(60, 78, 228); LD(61, 79, 1092); LD(62, 80, 229); LD(63, 81, 1093); LD(64, 82, 230); LD(65, 83, 1094); LD(66, 85, 660); LD(67, 86, 1524); LD(68, 87, 661); LD(69, 88, 1525); LD(70, 89, 662); LD(71, 90, 1526); LD(72, 99, 36); LD(73, 100, 900); LD(74, 101, 37); LD(75, 102, 901); LD(76, 103, 38); LD(77, 104, 902); LD(78, 106, 468); LD(79, 107, 1332); LD(80, 108, 469); LD(81, 109, 1333); LD(82, 110, 470); LD(83, 111, 1334); LD(84, 113, 42); LD(85, 114, 44); LD(86, 115, 908); LD(87, 117, 474); LD(88, 118, 476); LD(89, 119, 1340); LD(90, 121, 48); LD(91, 122, 50); LD(92, 123, 914); LD(93, 125, 480); LD(94, 126, 1344); LD(95, 127, 481); LD(96, 128, 1345); LD(97, 129, 482); LD(98, 130, 1346); LD(99, 139, 252); LD(100, 140, 1116); LD(101, 141, 253); LD(102, 142, 1117); LD(103, 143, 254); LD(104, 144, 1118); LD(105, 146, 684); LD(106, 147, 1548); LD(107, 148, 685); LD(108, 149, 1549); LD(109, 150, 686); LD(110, 151, 1550); LD(111, 153, 258); LD(112, 154, 260); LD(113, 155, 1124); LD(114, 157, 690); LD(115, 158, 692); LD(116, 159, 1556); LD(117, 161, 264); LD(118, 162, 266); LD(119, 163, 1130); LD(120, 165, 696); LD(121, 166, 1560); LD(122, 167, 697); LD(123, 168, 1561); LD(124, 169, 698); LD(125, 170, 1562); LD(126, 179, 72); LD(127, 180, 936); LD(128, 181, 73); LD(129, 182, 937); LD(130, 183, 74); LD(131, 184, 938); LD(132, 186, 504); LD(133, 187, 1368); LD(134, 188, 505); LD(135, 189, 1369); LD(136, 190, 506); LD(137, 191, 1370); LD(138, 193, 78); LD(139, 194, 80); LD(140, 195, 944); LD(141, 197, 510); LD(142, 198, 512); LD(143, 199, 1376); LD(144, 201, 84); LD(145, 202, 86); LD(146, 203, 950); LD(147, 205, 516); LD(148, 206, 1380); LD(149, 207, 517); LD(150, 208, 1381); LD(151, 209, 518); LD(152, 210, 1382); LD(153, 219, 288); LD(154, 220, 1152); LD(155, 221, 289); LD(156, 222, 1153); LD(157, 223, 290); LD(158, 224, 1154); LD(159, 226, 720); LD(160, 227, 1584); LD(161, 228, 721); LD(162, 229, 1585); LD(163, 230, 722); LD(164, 231, 1586); LD(165, 233, 294); LD(166, 234, 1158); LD(167, 235, 295); LD(168, 236, 1159); LD(169, 237, 296); LD(170, 238, 1160); LD(171, 240, 726); LD(172, 241, 1590); LD(173, 242, 727); LD(174, 243, 1591); LD(175, 244, 728); LD(176, 245, 1592); LD(177, 247, 300); LD(178, 248, 1164); LD(179, 249, 301); LD(180, 250, 1165); LD(181, 251, 302); LD(182, 252, 1166); LD(183, 254, 732); LD(184, 255, 1596); LD(185, 256, 733); LD(186, 257, 1597); LD(187, 258, 734); LD(188, 259, 1598)
#define B4_LOADXYZ4() LD(0, 56, 0); LD(1, 57, 864); LD(2, 58, 1); LD(3, 59, 865); LD(4, 60, 2); LD(5, 61, 866); LD(6, 63, 432); LD(7, 64, 1296); LD(8, 65, 433); LD(9, 66, 1297); LD(10, 67, 434); LD(11, 68, 1298); LD(12, 70, 6); LD(13, 71, 870); LD(14, 72, 7); LD(15, 73, 871); LD(16, 74, 8); LD(17, 75, 872); LD(18, 77, 438); LD(19, 78, 1302); LD(20, 79, 439); LD(21, 80, 1303); LD(22, 81, 440); LD(23, 82, 1304); LD(24, 84, 12); LD(25, 85, 876); LD(26, 86, 13); LD(27, 87, 877); LD(28, 88, 14); LD(29, 89, 878); LD(30, 91, 444); LD(31, 92, 1308); LD(32, 93, 445); LD(33, 94, 1309); LD(34, 95, 446); LD(35, 96, 1310); LD(36, 105, 216); LD(37, 106, 1080); LD(38, 107, 217); LD(39, 108, 1081); LD(40, 109, 218); LD(41, 110, 1082); LD(42, 112, 648); LD(43, 113, 1512); LD(44, 114, 1514); LD(45, 116, 222); LD(46, 117, 1086); LD(47, 118, 1088); LD(48, 120, 654); LD(49, 121, 1518); LD(50, 122, 1520); LD(51, 124, 228); LD(52, 125, 1092); LD(53, 126, 229); LD(54, 127, 1093); LD(55, 128, 230); LD(56, 129, 1094); LD(57, 131, 660); LD(58, 132, 1524); LD(59, 133, 661); LD(60, 134, 1525); LD(61, 135, 662); LD(62, 136, 1526); LD(63, 145, 36); LD(64, 146, 900); LD(65, 147, 37); LD(66, 148, 901); LD(67, 149, 38); LD(68, 150, 902); LD(69, 152, 468); LD(70, 153, 1332); LD(71, 154, 1334); LD(72, 156, 42); LD(73, 157, 906); LD(74, 158, 908); LD(75, 160, 474); LD(76, 161, 1338); LD(77, 162, 1340); LD(78, 164, 48); LD(79, 165, 912); LD(80, 166, 49); LD(81, 167, 913); LD(82, 168, 50); LD(83, 169, 914); LD(84, 171, 480); LD(85, 172, 1344); LD(86, 173, 481); LD(87, 174, 1345); LD(88, 175, 482); LD(89, 176, 1346); LD(90, 185, 252); LD(91, 186, 1116); LD(92, 187, 253); LD(93, 188, 1117); LD(94, 189, 254); LD(95, 190, 1118); LD(96, 192, 684); LD(97, 193, 1548); LD(98, 194, 1550); LD(99, 196, 258); LD(100, 197, 1122); LD(101, 198, 1124); LD(102, 200, 690); LD(103, 201, 1554); LD(104, 202, 1556); LD(105, 204, 264); LD(106, 205, 1128); LD(107, 206, 265); LD(108, 207, 1129); LD(109, 208, 266); LD(110, 209, 1130); LD(111, 211, 696); LD(112, 212, 1560); LD(113, 213, 697); LD(114, 214, 1561); LD(115, 215, 698); LD(116, 216, 1562); LD(117, 225, 72); LD(118, 226, 936); LD(119, 227, 73); LD(120, 228, 937); LD(121, 229, 74); LD(122, 230, 938); LD(123, 232, 504); LD(124, 233, 1368); LD(125, 234, 505); LD(126, 235, 1369); LD(127, 236, 506); LD(128, 237, 1370); LD(129, 239, 78); LD(130, 240, 942); LD(131, 241, 79); LD(132, 242, 943); LD(133, 243, 80); LD(134, 244, 944); LD(135, 246, 510); LD(136, 247, 1374); LD(137, 248, 511); LD(138, 249, 1375); LD(139, 250, 512); LD(140, 251, 1376); LD(141, 253, 84); LD(142, 254, 948); LD(143, 255, 85); LD(144, 256, 949); LD(145, 257, 86); LD(146, 258, 950); LD(147, 260, 516); LD(148, 261, 1380); LD(149, 262, 517); LD(150, 263, 1381); LD(151, 264, 518); LD(152, 265, 1382); LD(153, 274, 288); LD(154, 275, 1152); LD(155, 276, 289); LD(156, 277, 1153); LD(157, 278, 290); LD(158, 279, 1154); LD(159, 281, 720); LD(160, 282, 1584); LD(161, 283, 721); LD(162, 284, 1585); LD(163, 285, 722); LD(164, 286, 1586); LD(165, 288, 294); LD(166, 289, 1158); LD(167, 290, 295); LD(168, 291, 1159); LD(169, 292, 296); LD(170, 293, 1160); LD(171, 295, 726); LD(172, 296, 1590); LD(173, 297, 727); LD(174, 298, 1591); LD(175, 299, 728); LD(176, 300, 1592); LD(177, 302, 300); LD(178, 303, 1164); LD(179, 304, 301); LD(180, 305, 1165); LD(181, 306, 302); LD(182, 307, 1166); LD(183, 309, 732); LD(184, 310, 1596); LD(185, 311, 733); LD(186, 312, 1597); LD(187, 313, 734); LD(188, 314, 1598)
#define B4_LOADXYZ5() LD(0, 7, 0); LD(1, 8, 864); LD(2, 9, 1); LD(3, 10, 865); LD(4, 11, 2); LD(5, 12, 866); LD(6, 14, 432); LD(7, 15, 1296); LD(8, 16, 433); LD(9, 17, 1297); LD(10, 18, 434); LD(11, 19, 1298); LD(12, 21, 6); LD(13, 22, 870); LD(14, 23, 7); LD(15, 24, 871); LD(16, 25, 8); LD(17, 26, 872); LD(18, 28, 438); LD(19, 29, 1302); LD(20, 30, 439); LD(21, 31, 1303); LD(22, 32, 440); LD(23, 33, 1304); LD(24, 35, 12); LD(25, 36, 876); LD(26, 37, 13); LD(27, 38, 877); LD(28, 39, 14); LD(29, 40, 878); LD(30, 42, 444); LD(31, 43, 1308); LD(32, 44, 445); LD(33, 45, 1309); LD(34, 46, 446); LD(35, 47, 1310); LD(36, 56, 216); LD(37, 57, 1080); LD(38, 58, 217); LD(39, 59, 1081); LD(40, 60, 218); LD(41, 61, 1082); LD(42, 63, 648); LD(43, 64, 1512); LD(44, 65, 649); LD(45, 66, 1513); LD(46, 67, 650); LD(47, 68, 1514); LD(48, 70, 222); LD(49, 71, 1086); LD(50, 72, 223); LD(51, 73, 1087); LD(52, 74, 224); LD(53, 75, 1088); LD(54, 77, 654); LD(55, 78, 1518); LD(56, 79, 655); LD(57, 80, 1519); LD(58, 81, 656); LD(59, 82, 1520); LD(60, 84, 228); LD(61, 85, 1092); LD(62, 86, 229); LD(63, 87, 1093); LD(64, 88, 230); LD(65, 89, 1094); LD(66, 91, 660); LD(67, 92, 1524); LD(68, 93, 661); LD(69, 94, 1525); LD(70, 95, 662); LD(71, 96, 1526); LD(72, 105, 36); LD(73, 106, 900); LD(74, 107, 37); LD(75, 108, 901); LD(76, 109, 38); LD(77, 110, 902); LD(78, 112, 468); LD(79, 113, 1332); LD(80, 114, 1334); LD(81, 116, 42); LD(82, 117, 906); LD(83, 118, 908); LD(84, 120, 474); LD(85, 121, 1338); LD(86, 122, 1340); LD(87, 124, 48); LD(88, 125, 912); LD(89, 126, 49); LD(90, 127, 913); LD(91, 128, 50); LD(92, 129, 914); LD(93, 131, 480); LD(94, 132, 1344); LD(95, 133, 481); LD(96, 134, 1345); LD(97, 135, 482); LD(98, 136, 1346); LD(99, 145, 252); LD(100, 146, 1116); LD(101, 147, 253); LD(102, 148, 1117); LD(103, 149, 254); LD(104, 150, 1118); LD(105, 152, 684); LD(106, 153, 1548); LD(107, 154, 1550); LD(108, 156, 258); LD(109, 157, 1122); LD(110, 158, 1124); LD(111, 160, 690); LD(112, 161, 1554); LD(113, 162, 1556); LD(114, 164, 264); LD(115, 165, 1128); LD(116, 166, 265); LD(117, 167, 1129); LD(118, 168, 266); LD(119, 169, 1130); LD(120, 171, 696); LD(121, 172, 1560); LD(122, 173, 697); LD(123, 174, 1561); LD(124, 175, 698); LD(125, 176, 1562); LD(126, 185, 72); LD(127, 186, 936); LD(128, 187, 73); LD(129, 188, 937); LD(130, 189, 74); LD(131, 190, 938); LD(132, 192, 504); LD(133, 193, 1368); LD(134, 194, 1370); LD(135, 196, 78); LD(136, 197, 942); LD(137, 198, 944); LD(138, 200, 510); LD(139, 201, 1374); LD(140, 202, 1376); LD(141, 204, 84); LD(142, 205, 948); LD(143, 206, 85); LD(144, 207, 949); LD(145, 208, 86); LD(146, 209, 950); LD(147, 211, 516); LD(148, 212, 1380); LD(149, 213, 517); LD(150, 214, 1381); LD(151, 215, 518); LD(152, 216, 1382); LD(153, 225, 288); LD(154, 226, 1152); LD(155, 227, 289); LD(156, 228, 1153); LD(157, 229, 290); LD(158, 230, 1154); LD(159, 232, 720); LD(160, 233, 1584); LD(161, 234, 721); LD(162, 235, 1585); LD(163, 236, 722); LD(164, 237, 1586); LD(165, 239, 294); LD(166, 240, 1158); LD(167, 241, 295); LD(168, 242, 1159); LD(169, 243, 296); LD(170, 244, 1160); LD(171, 246, 726); LD(172, 247, 1590); LD(173, 248, 727); LD(174, 249, 1591); LD(175, 250, 728); LD(176, 251, 1592); LD(177, 253, 300); LD(178, 254, 1164); LD(179, 255, 301); LD(180, 256, 1165); LD(181, 257, 302); LD(182, 258, 1166); LD(183, 260, 732); LD(184, 261, 1596); LD(185, 262, 733); LD(186, 263, 1597); LD(187, 264, 734); LD(188, 265, 1598)
#define B4_LOADXYZ6() LD(0, 49, 0); LD(1, 50, 864); LD(2, 51, 1); LD(3, 52, 865); LD(4, 53, 2); LD(5, 54, 866); LD(6, 56, 432); LD(7, 57, 1296); LD(8, 58, 433); LD(9, 59, 1297); LD(10, 60, 434); LD(11, 61, 1298); LD(12, 63, 6); LD(13, 64, 870); LD(14, 65, 7); LD(15, 66, 871); LD(16, 67, 8); LD(17, 68, 872); LD(18, 70, 438); LD(19, 71, 1302); LD(20, 72, 439); LD(21, 73, 1303); LD(22, 74, 440); LD(23, 75, 1304); LD(24, 77, 12); LD(25, 78, 876); LD(26, 79, 13); LD(27, 80, 877); LD(28, 81, 14); LD(29, 82, 878); LD(30, 84, 444); LD(31, 85, 1308); LD(32, 86, 445); LD(33, 87, 1309); LD(34, 88, 446); LD(35, 89, 1310); LD(36, 98, 216); LD(37, 99, 1080); LD(38, 100, 217); LD(39, 101, 1081); LD(40, 102, 218); LD(41, 103, 1082); LD(42, 105, 648); LD(43, 106, 1512); LD(44, 107, 649); LD(45, 108, 1513); LD(46, 109, 650); LD(47, 110, 1514); LD(48, 112, 222); LD(49, 113, 1086); LD(50, 114, 1088); LD(51, 116, 654); LD(52, 117, 1518); LD(53, 118, 1520); LD(54, 120, 228); LD(55, 121, 1092); LD(56, 122, 1094); LD(57, 124, 660); LD(58, 125, 1524); LD(59, 126, 661); LD(60, 127, 1525); LD(61, 128, 662); LD(62, 129, 1526); LD(63, 138, 36); LD(64, 139, 900); LD(65, 140, 37); LD(66, 141, 901); LD(67, 142, 38); LD(68, 143, 902); LD(69, 145, 468); LD(70, 146, 1332); LD(71, 147, 469); LD(72, 148, 1333); LD(73, 149, 470); LD(74, 150, 1334); LD(75, 152, 42); LD(76, 153, 906); LD(77, 154, 908); LD(78, 156, 474); LD(79, 157, 1338); LD(80, 158, 1340); LD(81, 160, 48); LD(82, 161, 912); LD(83, 162, 914); LD(84, 164, 480); LD(85, 165, 1344); LD(86, 166, 481); LD(87, 167, 1345); LD(88, 168, 482); LD(89, 169, 1346); LD(90, 178, 252); LD(91, 179, 1116); LD(92, 180, 253); LD(93, 181, 1117); LD(94, 182, 254); LD(95, 183, 1118); LD(96, 185, 684); LD(97, 186, 1548); LD(98, 187, 685); LD(99, 188, 1549); LD(100, 189, 686); LD(101, 190, 1550); LD(102, 192, 258); LD(103, 193, 1122); LD(104, 194, 1124); LD(105, 196, 690); LD(106, 197, 1554); LD(107, 198, 1556); LD(108, 200, 264); LD(109, 201, 1128); LD(110, 202, 1130); LD(111, 204, 696); LD(112, 205, 1560); LD(113, 206, 697); LD(114, 207, 1561); LD(115, 208, 698); LD(116, 209, 1562); LD(117, 218, 72); LD(118, 219, 936); LD(119, 220, 73); LD(120, 221, 937); LD(121, 222, 74); LD(122, 223, 938); LD(123, 225, 504); LD(124, 226, 1368); LD(125, 227, 505); LD(126, 228, 1369); LD(127, 229, 506); LD(128, 230, 1370); LD(129, 232, 78); LD(130, 233, 942); LD(131, 234, 79); LD(132, 235, 943); LD(133, 236, 80); LD(134, 237, 944); LD(135, 239, 510); LD(136, 240, 1374); LD(137, 241, 511); LD(138, 242, 1375); LD(139, 243, 512); LD(140, 244, 1376); LD(141, 246, 84); LD(142, 247, 948); LD(143, 248, 85); LD(144, 249, 949); LD(145, 250, 86); LD(146, 251, 950); LD(147, 253, 516); LD(148, 254, 1380); LD(149, 255, 517); LD(150, 256, 1381); LD(151, 257, 518); LD(152, 258, 1382); LD(153, 267, 288); LD(154, 268, 1152); LD(155, 269, 289); LD(156, 270, 1153); LD(157, 271, 290); LD(158, 272, 1154); LD(159, 274, 720); LD(160, 275, 1584); LD(161, 276, 721); LD(162, 277, 1585); LD(163, 278, 722); LD(164, 279, 1586); LD(165, 281, 294); LD(166, 282, 1158); LD(167, 283, 295); LD(168, 284, 1159); LD(169, 285, 296); LD(170, 286, 1160); LD(171, 288, 726); LD(172, 289, 1590); LD(173, 290, 727); LD(174, 291, 1591); LD(175, 292, 728); LD(176, 293, 1592); LD(177, 295, 300); LD(178, 296, 1164); LD(179, 297, 301); LD(180, 298, 1165); LD(181, 299, 302); LD(182, 300, 1166); LD(183, 302, 732); LD(184, 303, 1596); LD(185, 304, 733); LD(186, 305, 1597); LD(187, 306, 734); LD(188, 307, 1598)
#define B4_LOADXYZ7() LD(0, 0, 0); LD(1, 1, 864); LD(2, 2, 1); LD(3, 3, 865); LD(4, 4, 2); LD(5, 5, 866); LD(6, 7, 432); LD(7, 8, 1296); LD(8, 9, 433); LD(9, 10, 1297); LD(10, 11, 434); LD(11, 12, 1298); LD(12, 14, 6); LD(13, 15, 870); LD(14, 16, 7); LD(15, 17, 871); LD(16, 18, 8); LD(17, 19, 872); LD(18, 21, 438); LD(19, 22, 1302); LD(20, 23, 439); LD(21, 24, 1303); LD(22, 25, 440); LD(23, 26, 1304); LD(24, 28, 12); LD(25, 29, 876); LD(26, 30, 13); LD(27, 31, 877); LD(28, 32, 14); LD(29, 33, 878); LD(30, 35, 444); LD(31, 36, 1308); LD(32, 37, 445); LD(33, 38, 1309); LD(34, 39, 446); LD(35, 40, 1310); LD(36, 49, 216); LD(37, 50, 1080); LD(38, 51, 217); LD(39, 52, 1081); LD(40, 53, 218); LD(41, 54, 1082); LD(42, 56, 648); LD(43, 57, 1512); LD(44, 58, 649); LD(45, 59, 1513); LD(46, 60, 650); LD(47, 61, 1514); LD(48, 63, 222); LD(49, 64, 1086); LD(50, 65, 223); LD(51, 66, 1087); LD(52, 67, 224); LD(53, 68, 1088); LD(54, 70, 654); LD(55, 71, 1518); LD(56, 72, 655); LD(57, 73, 1519); LD(58, 74, 656); LD(59, 75, 1520); LD(60, 77, 228); LD(61, 78, 1092); LD(62, 79, 229); LD(63, 80, 1093); LD(64, 81, 230); LD(65, 82, 1094); LD(66, 84, 660); LD(67, 85, 1524); LD(68, 86, 661); LD(69, 87, 1525); LD(70, 88, 662); LD(71, 89, 1526); LD(72, 98, 36); LD(73, 99, 900); LD(74, 100, 37); LD(75, 101, 901); LD(76, 102, 38); LD(77, 103, 902); LD(78, 105, 468); LD(79, 106, 1332); LD(80, 107, 469); LD(81, 108, 1333); LD(82, 109, 470); LD(83, 110, 1334); LD(84, 112, 42); LD(85, 113, 906); LD(86, 114, 908); LD(87, 116, 474); LD(88, 117, 1338); LD(89, 118, 1340); LD(90, 120, 48); LD(91, 121, 912); LD(92, 122, 914); LD(93, 124, 480); LD(94, 125, 1344); LD(95, 126, 481); LD(96, 127, 1345); LD(97, 128, 482); LD(98, 129, 1346); LD(99, 138, 252); LD(100, 139, 1116); LD(101, 140, 253); LD(102, 141, 1117); LD(103, 142, 254); LD(104, 143, 1118); LD(105, 145, 684); LD(106, 146, 1548); LD(107, 147, 685); LD(108, 148, 1549); LD(109, 149, 686); LD(110, 150, 1550); LD(111, 152, 258); LD(112, 153, 1122); LD(113, 154, 1124); LD(114, 156, 690); LD(115, 157, 1554); LD(116, 158, 1556); LD(117, 160, 264); LD(118, 161, 1128); LD(119, 162, 1130); LD(120, 164, 696); LD(121, 165, 1560); LD(122, 166, 697); LD(123, 167, 1561); LD(124, 168, 698); LD(125, 169, 1562); LD(126, 178, 72); LD(127, 179, 936); LD(128, 180, 73); LD(129, 181, 937); LD(130, 182, 74); LD(131, 183, 938); LD(132, 185, 504); LD(133, 186, 1368); LD(134, 187, 505); LD(135, 188, 1369); LD(136, 189, 506); LD(137, 190, 1370); LD(138, 192, 78); LD(139, 193, 942); LD(140, 194, 944); LD(141, 196, 510); LD(142, 197, 1374); LD(143, 198, 1376); LD(144, 200, 84); LD(145, 201, 948); LD(146, 202, 950); LD(147, 204, 516); LD(148, 205, 1380); LD(149, 206, 517); LD(150, 207, 1381); LD(151, 208, 518); LD(152, 209, 1382); LD(153, 218, 288); LD(154, 219, 1152); LD(155, 220, 289); LD(156, 221, 1153); LD(157, 222, 290); LD(158, 223, 1154); LD(159, 225, 720); LD(160, 226, 1584); LD(161, 227, 721); LD(162, 228, 1585); LD(163, 229, 722); LD(164, 230, 1586); LD(165, 232, 294); LD(166, 233, 1158); LD(167, 234, 295); LD(168, 235, 1159); LD(169, 236, 296); LD(170, 237, 1160); LD(171, 239, 726); LD(172, 240, 1590); LD(173, 241, 727); LD(174, 242, 1591); LD(175, 243, 728); LD(176, 244, 1592); LD(177, 246, 300); LD(178, 247, 1164); LD(179, 248, 301); LD(180, 249, 1165); LD(181, 250, 302); LD(182, 251, 1166); LD(183, 253, 732); LD(184, 254, 1596); LD(185, 255, 733); LD(186, 256, 1597); LD(187, 257, 734); LD(188, 258, 1598)

#define B2_LOADXYZ0() LD(0, 57, 0); LD(1, 58, 256); LD(2, 59, 1); LD(3, 60, 257); LD(4, 61, 2); LD(5, 62, 258); LD(6, 64, 128); LD(7, 65, 384); LD(8, 66, 129); LD(9, 67, 385); LD(10, 68, 130); LD(11, 69, 386); LD(12, 71, 4); LD(13, 72, 260); LD(14, 73, 5); LD(15, 74, 261); LD(16, 75, 6); LD(17, 76, 262); LD(18, 78, 132); LD(19, 79, 388); LD(20, 80, 133); LD(21, 81, 389); LD(22, 82, 134); LD(23, 83, 390); LD(24, 85, 8); LD(25, 86, 264); LD(26, 87, 9); LD(27, 88, 265); LD(28, 89, 10); LD(29, 90, 266); LD(30, 92, 136); LD(31, 93, 392); LD(32, 94, 137); LD(33, 95, 393); LD(34, 96, 138); LD(35, 97, 394); LD(36, 106, 64); LD(37, 107, 320); LD(38, 108, 65); LD(39, 109, 321); LD(40, 110, 66); LD(41, 111, 322); LD(42, 113, 192); LD(43, 114, 194); LD(44, 115, 450); LD(45, 117, 68); LD(46, 118, 70); LD(47, 119, 326); LD(48, 121, 196); LD(49, 122, 198); LD(50, 123, 454); LD(51, 125, 72); LD(52, 126, 328); LD(53, 127, 73); LD(54, 128, 329); LD(55, 129, 74); LD(56, 130, 330); LD(57, 132, 200); LD(58, 133, 456); LD(59, 134, 201); LD(60, 135, 457); LD(61, 136, 202); LD(62, 137, 458); LD(63, 146, 16); LD(64, 147, 272); LD(65, 148, 17); LD(66, 149, 273); LD(67, 150, 18); LD(68, 151, 274); LD(69, 153, 144); LD(70, 154, 146); LD(71, 155, 402); LD(72, 157, 20); LD(73, 158, 22); LD(74, 159, 278); LD(75, 161, 148); LD(76, 162, 150); LD(77, 163, 406); LD(78, 165, 24); LD(79, 166, 280); LD(80, 167, 25); LD(81, 168, 281); LD(82, 169, 26); LD(83, 170, 282); LD(84, 172, 152); LD(85, 173, 408); LD(86, 174, 153); LD(87, 175, 409); LD(88, 176, 154); LD(89, 177, 410); LD(90, 186, 80); LD(91, 187, 336); LD(92, 188, 81); LD(93, 189, 337); LD(94, 190, 82); LD(95, 191, 338); LD(96, 193, 208); LD(97, 194, 210); LD(98, 195, 466); LD(99, 197, 84); LD(100, 198, 86); LD(101, 199, 342); LD(102, 201, 212); LD(103, 202, 214); LD(104, 203, 470); LD(105, 205, 88); LD(106, 206, 344); LD(107, 207, 89); LD(108, 208, 345); LD(109, 209, 90); LD(110, 210, 346); LD(111, 212, 216); LD(112, 213, 472); LD(113, 214, 217); LD(114, 215, 473); LD(115, 216, 218); LD(116, 217, 474); LD(117, 226, 32); LD(118, 227, 288); LD(119, 228, 33); LD(120, 229, 289); LD(121, 230, 34); LD(122, 231, 290); LD(123, 233, 160); LD(124, 234, 416); LD(125, 235, 161); LD(126, 236, 417); LD(127, 237, 162); LD(128, 238, 418); LD(129, 240, 36); LD(130, 241, 292); LD(131, 242, 37); LD(132, 243, 293); LD(133, 244, 38); LD(134, 245, 294); LD(135, 247, 164); LD(136, 248, 420); LD(137, 249, 165); LD(138, 250, 421); LD(139, 251, 166); LD(140, 252, 422); LD(141, 254, 40); LD(142, 255, 296); LD(143, 256, 41); LD(144, 257, 297); LD(145, 258, 42); LD(146, 259, 298); LD(147, 261, 168); LD(148, 262, 424); LD(149, 263, 169); LD(150, 264, 425); LD(151, 265, 170); LD(152, 266, 426); LD(153, 275, 96); LD(154, 276, 352); LD(155, 277, 97); LD(156, 278, 353); LD(157, 279, 98); LD(158, 280, 354); LD(159, 282, 224); LD(160, 283, 480); LD(161, 284, 225); LD(162, 285, 481); LD(163, 286, 226); LD(164, 287, 482); LD(165, 289, 100); LD(166, 290, 356); LD(167, 291, 101); LD(168, 292, 357); LD(169, 293, 102); LD(170, 294, 358); LD(171, 296, 228); LD(172, 297, 484); LD(173, 298, 229); LD(174, 299, 485); LD(175, 300, 230); LD(176, 301, 486); LD(177, 303, 104); LD(178, 304, 360); LD(179, 305, 105); LD(180, 306, 361); LD(181, 307, 106); LD(182, 308, 362); LD(183, 310, 232); LD(184, 311, 488); LD(185, 312, 233); LD(186, 313, 489); LD(187, 314, 234); LD(188, 315, 490)
#define B2_LOADXYZ1() LD(0, 8, 0); LD(1, 9, 256); LD(2, 10, 1); LD(3, 11, 257); LD(4, 12, 2); LD(5, 13, 258); LD(6, 15, 128); LD(7, 16, 384); LD(8, 17, 129); LD(9, 18, 385); LD(10, 19, 130); LD(11, 20, 386); LD(12, 22, 4); LD(13, 23, 260); LD(14, 24, 5); LD(15, 25, 261); LD(16, 26, 6); LD(17, 27, 262); LD(18, 29, 132); LD(19, 30, 388); LD(20, 31, 133); LD(21, 32, 389); LD(22, 33, 134); LD(23, 34, 390); LD(24, 36, 8); LD(25, 37, 264); LD(26, 38, 9); LD(27, 39, 265); LD(28, 40, 10); LD(29, 41, 266); LD(30, 43, 136); LD(31, 44, 392); LD(32, 45, 137); LD(33, 46, 393); LD(34, 47, 138); LD(35, 48, 394); LD(36, 57, 64); LD(37, 58, 320); LD(38, 59, 65); LD(39, 60, 321); LD(40, 61, 66); LD(41, 62, 322); LD(42, 64, 192); LD(43, 65, 448); LD(44, 66, 193); LD(45, 67, 449); LD(46, 68, 194); LD(47, 69, 450); LD(48, 71, 68); LD(49, 72, 324); LD(50, 73, 69); LD(51, 74, 325); LD(52, 75, 70); LD(53, 76, 326); LD(54, 78, 196); LD(55, 79, 452); LD(56, 80, 197); LD(57, 81, 453); LD(58, 82, 198); LD(59, 83, 454); LD(60, 85, 72); LD(61, 86, 328); LD(62, 87, 73); LD(63, 88, 329); LD(64, 89, 74); LD(65, 90, 330); LD(66, 92, 200); LD(67, 93, 456); LD(68, 94, 201); LD(69, 95, 457); LD(70, 96, 202); LD(71, 97, 458); LD(72, 106, 16); LD(73, 107, 272); LD(74, 108, 17); LD(75, 109, 273); LD(76, 110, 18); LD(77, 111, 274); LD(78, 113, 144); LD(79, 114, 146); LD(80, 115, 402); LD(81, 117, 20); LD(82, 118, 22); LD(83, 119, 278); LD(84, 121, 148); LD(85, 122, 150); LD(86, 123, 406); LD(87, 125, 24); LD(88, 126, 280); LD(89, 127, 25); LD(90, 128, 281); LD(91, 129, 26); LD(92, 130, 282); LD(93, 132, 152); LD(94, 133, 408); LD(95, 134, 153); LD(96, 135, 409); LD(97, 136, 154); LD(98, 137, 410); LD(99, 146, 80); LD(100, 147, 336); LD(101, 148, 81); LD(102, 149, 337); LD(103, 150, 82); LD(104, 151, 338); LD(105, 153, 208); LD(106, 154, 210); LD(107, 155, 466); LD(108, 157, 84); LD(109, 158, 86); LD(110, 159, 342); LD(111, 161, 212); LD(112, 162, 214); LD(113, 163, 470); LD(114, 165, 88); LD(115, 166, 344); LD(116, 167, 89); LD(117, 168, 345); LD(118, 169, 90); LD(119, 170, 346); LD(120, 172, 216); LD(121, 173, 472); LD(122, 174, 217); LD(123, 175, 473); LD(124, 176, 218); LD(125, 177, 474); LD(126, 186, 32); LD(127, 187, 288); LD(128, 188, 33); LD(129, 189, 289); LD(130, 190, 34); LD(131, 191, 290); LD(132, 193, 160); LD(133, 194, 162); LD(134, 195, 418); LD(135, 197, 36); LD(136, 198, 38); LD(137, 199, 294); LD(138, 201, 164); LD(139, 202, 166); LD(140, 203, 422); LD(141, 205, 40); LD(142, 206, 296); LD(143, 207, 41); LD(144, 208, 297); LD(145, 209, 42); LD(146, 210, 298); LD(147, 212, 168); LD(148, 213, 424); LD(149, 214, 169); LD(150, 215, 425); LD(151, 216, 170); LD(152, 217, 426); LD(153, 226, 96); LD(154, 227, 352); LD(155, 228, 97); LD(156, 229, 353); LD(157, 230, 98); LD(158, 231, 354); LD(159, 233, 224); LD(160, 234, 480); LD(161, 235, 225); LD(162, 236, 481); LD(163, 237, 226); LD(164, 238, 482); LD(165, 240, 100); LD(166, 241, 356); LD(167, 242, 101); LD(168, 243, 357); LD(169, 244, 102); LD(170, 245, 358); LD(171, 247, 228); LD(172, 248, 484); LD(173, 249, 229); LD(174, 250, 485); LD(175, 251, 230); LD(176, 252, 486); LD(177, 254, 104); LD(178, 255, 360); LD(179, 256, 105); LD(180, 257, 361); LD(181, 258, 106); LD(182, 259, 362); LD(183, 261, 232); LD(184, 262, 488); LD(185, 263, 233); LD(186, 264, 489); LD(187, 265, 234); LD(188, 266, 490)
#define B2_LOADXYZ2() LD(0, 50, 0); LD(1, 51, 256); LD(2, 52, 1); LD(3, 53, 257); LD(4, 54, 2); LD(5, 55, 258); LD(6, 57, 128); LD(7, 58, 384); LD(8, 59, 129); LD(9, 60, 385); LD(10, 61, 130); LD(11, 62, 386); LD(12, 64, 4); LD(13, 65, 260); LD(14, 66, 5); LD(15, 67, 261); LD(16, 68, 6); LD(17, 69, 262); LD(18, 71, 132); LD(19, 72, 388); LD(20, 73, 133); LD(21, 74, 389); LD(22, 75, 134); LD(23, 76, 390); LD(24, 78, 8); LD(25, 79, 264); LD(26, 80, 9); LD(27, 81, 265); LD(28, 82, 10); LD(29, 83, 266); LD(30, 85, 136); LD(31, 86, 392); LD(32, 87, 137); LD(33, 88, 393); LD(34, 89, 138); LD(35, 90, 394); LD(36, 99, 64); LD(37, 100, 320); LD(38, 101, 65); LD(39, 102, 321); LD(40, 103, 66); LD(41, 104, 322); LD(42, 106, 192); LD(43, 107, 448); LD(44, 108, 193); LD(45, 109, 449); LD(46, 110, 194); LD(47, 111, 450); LD(48, 113, 68); LD(49, 114, 70); LD(50, 115, 326); LD(51, 117, 196); LD(52, 118, 198); LD(53, 119, 454); LD(54, 121, 72); LD(55, 122, 74); LD(56, 123, 330); LD(57, 125, 200); LD(58, 126, 456); LD(59, 127, 201); LD(60, 128, 457); LD(61, 129, 202); LD(62, 130, 458); LD(63, 139, 16); LD(64, 140, 272); LD(65, 141, 17); LD(66, 142, 273); LD(67, 143, 18); LD(68, 144, 274); LD(69, 146, 144); LD(70, 147, 400); LD(71, 148, 145); LD(72, 149, 401); LD(73, 150, 146); LD(74, 151, 402); LD(75, 153, 20); LD(76, 154, 22); LD(77, 155, 278); LD(78, 157, 148); LD(79, 158, 150); LD(80, 159, 406); LD(81, 161, 24); LD(82, 162, 26); LD(83, 163, 282); LD(84, 165, 152); LD(85, 166, 408); LD(86, 167, 153); LD(87, 168, 409); LD(88, 169, 154); LD(89, 170, 410); LD(90, 179, 80); LD(91, 180, 336); LD(92, 181, 81); LD(93, 182, 337); LD(94, 183, 82); LD(95, 184, 338); LD(96, 186, 208); LD(97, 187, 464); LD(98, 188, 209); LD(99, 189, 465); LD(100, 190, 210); LD(101, 191, 466); LD(102, 193, 84); LD(103, 194, 86); LD(104, 195, 342); LD(105, 197, 212); LD(106, 198, 214); LD(107, 199, 470); LD(108, 201, 88); LD(109, 202, 90); LD(110, 203, 346); LD(111, 205, 216); LD(112, 206, 472); LD(113, 207, 217); LD(114, 208, 473); LD(115, 209, 218); LD(116, 210, 474); LD(117, 219, 32); LD(118, 220, 288); LD(119, 221, 33); LD(120, 222, 289); LD(121, 223, 34); LD(122, 224, 290); LD(123, 226, 160); LD(124, 227, 416); LD(125, 228, 161); LD(126, 229, 417); LD(127, 230, 162); LD(128, 231, 418); LD(129, 233, 36); LD(130, 234, 292); LD(131, 235, 37); LD(132, 236, 293); LD(133, 237, 38); LD(134, 238, 294); LD(135, 240, 164); LD(136, 241, 420); LD(137, 242, 165); LD(138, 243, 421); LD(139, 244, 166); LD(140, 245, 422); LD(141, 247, 40); LD(142, 248, 296); LD(143, 249, 41); LD(144, 250, 297); LD(145, 251, 42); LD(146, 252, 298); LD(147, 254, 168); LD(148, 255, 424); LD(149, 256, 169); LD(150, 257, 425); LD(151, 258, 170); LD(152, 259, 426); LD(153, 268, 96); LD(154, 269, 352); LD(155, 270, 97); LD(156, 271, 353); LD(157, 272, 98); LD(158, 273, 354); LD(159, 275, 224); LD(160, 276, 480); LD(161, 277, 225); LD(162, 278, 481); LD(163, 279, 226); LD(164, 280, 482); LD(165, 282, 100); LD(166, 283, 356); LD(167, 284, 101); LD(168, 285, 357); LD(169, 286, 102); LD(170, 287, 358); LD(171, 289, 228); LD(172, 290, 484); LD(173, 291, 229); LD(174, 292, 485); LD(175, 293, 230); LD(176, 294, 486); LD(177, 296, 104); LD(178, 297, 360); LD(179, 298, 105); LD(180, 299, 361); LD(181, 300, 106); LD(182, 301, 362); LD(183, 303, 232); LD(184, 304, 488); LD(185, 305, 233); LD(186, 306, 489); LD(187, 307, 234); LD(188, 308, 490)
#define B2_LOADXYZ3() LD(0, 1, 0); LD(1, 2, 256); LD(2, 3, 1); LD(3, 4, 257); LD(4, 5, 2); LD(5, 6, 258); LD(6, 8, 128); LD(7, 9, 384); LD(8, 10, 129); LD(9, 11, 385); LD(10, 12, 130); LD(11, 13, 386); LD(12, 15, 4); LD(13, 16, 260); LD(14, 17, 5); LD(15, 18, 261); LD(16, 19, 6); LD(17, 20, 262); LD(18, 22, 132); LD(19, 23, 388); LD(20, 24, 133); LD(21, 25, 389); LD(22, 26, 134); LD(23, 27, 390); LD(24, 29, 8); LD(25, 30, 264); LD(26, 31, 9); LD(27, 32, 265); LD(28, 33, 10); LD(29, 34, 266); LD(30, 36, 136); LD(31, 37, 392); LD(32, 38, 137); LD(33, 39, 393); LD(34, 40, 138); LD(35, 41, 394); LD(36, 50, 64); LD(37, 51, 320); LD(38, 52, 65); LD(39, 53, 321); LD(40, 54, 66); LD(41, 55, 322); LD(42, 57, 192); LD(43, 58, 448); LD(44, 59, 193); LD(45, 60, 449); LD(46, 61, 194); LD(47, 62, 450); LD(48, 64, 68); LD(49, 65, 324); LD(50, 66, 69); LD(51, 67, 325); LD(52, 68, 70); LD(53, 69, 326); LD(54, 71, 196); LD(55, 72, 452); LD(56, 73, 197); LD(57, 74, 453); LD(58, 75, 198); LD(59, 76, 454); LD(60, 78, 72); LD(61, 79, 328); LD(62, 80, 73); LD(63, 81, 329); LD(64, 82, 74); LD(65, 83, 330); LD(66, 85, 200); LD(67, 86, 456); LD(68, 87, 201); LD(69, 88, 457); LD(70, 89, 202); LD(71, 90, 458); LD(72, 99, 16); LD(73, 100, 272); LD(74, 101, 17); LD(75, 102, 273); LD(76, 103, 18); LD(77, 104, 274); LD(78, 106, 144); LD(79, 107, 400); LD(80, 108, 145); LD(81, 109, 401); LD(82, 110, 146); LD(83, 111, 402); LD(84, 113, 20); LD(85, 114, 22); LD(86, 115, 278); LD(87, 117, 148); LD(88, 118, 150); LD(89, 119, 406); LD(90, 121, 24); LD(91, 122, 26); LD(92, 123, 282); LD(93, 125, 152); LD(94, 126, 408); LD(95, 127, 153); LD(96, 128, 409); LD(97, 129, 154); LD(98, 130, 410); LD(99, 139, 80); LD(100, 140, 336); LD(101, 141, 81); LD(102, 142, 337); LD(103, 143, 82); LD(104, 144, 338); LD(105, 146, 208); LD(106, 147, 464); LD(107, 148, 209); LD(108, 149, 465); LD(109, 150, 210); LD(110, 151, 466); LD(111, 153, 84); LD(112, 154, 86); LD(113, 155, 342); LD(114, 157, 212); LD(115, 158, 214); LD(116, 159, 470); LD(117, 161, 88); LD(118, 162, 90); LD(119, 163, 346); LD(120, 165, 216); LD(121, 166, 472); LD(122, 167, 217); LD(123, 168, 473); LD(124, 169, 218); LD(125, 170, 474); LD(126, 179, 32); LD(127, 180, 288); LD(128, 181, 33); LD(129, 182, 289); LD(130, 183, 34); LD(131, 184, 290); LD(132, 186, 160); LD(133, 187, 416); LD(134, 188, 161); LD(135, 189, 417); LD(136, 190, 162); LD(137, 191, 418); LD(138, 193, 36); LD(139, 194, 38); LD(140, 195, 294); LD(141, 197, 164); LD(142, 198, 166); LD(143, 199, 422); LD(144, 201, 40); LD(145, 202, 42); LD(146, 203, 298); LD(147, 205, 168); LD(148, 206, 424); LD(149, 207, 169); LD(150, 208, 425); LD(151, 209, 170); LD(152, 210, 426); LD(153, 219, 96); LD(154, 220, 352); LD(155, 221, 97); LD(156, 222, 353); LD(157, 223, 98); LD(158, 224, 354); LD(159, 226, 224); LD(160, 227, 480); LD(161, 228, 225); LD(162, 229, 481); LD(163, 230, 226); LD(164, 231, 482); LD(165, 233, 100); LD(166, 234, 356); LD(167, 235, 101); LD(168, 236, 357); LD(169, 237, 102); LD(170, 238, 358); LD(171, 240, 228); LD(172, 241, 484); LD(173, 242, 229); LD(174, 243, 485); LD(175, 244, 230); LD(176, 245, 486); LD(177, 247, 104); LD(178, 248, 360); LD(179, 249, 105); LD(180, 250, 361); LD(181, 251, 106); LD(182, 252, 362); LD(183, 254, 232); LD(184, 255, 488); LD(185, 256, 233); LD(186, 257, 489); LD(187, 258, 234); LD(188, 259, 490)
#define B2_LOADXYZ4() LD(0, 56, 0); LD(1, 57, 256); LD(2, 58, 1); LD(3, 59, 257); LD(4, 60, 2); LD(5, 61, 258); LD(6, 63, 128); LD(7, 64, 384); LD(8, 65, 129); LD(9, 66, 385); LD(10, 67, 130); LD(11, 68, 386); LD(12, 70, 4); LD(13, 71, 260); LD(14, 72, 5); LD(15, 73, 261); LD(16, 74, 6); LD(17, 75, 262); LD(18, 77, 132); LD(19, 78, 388); LD(20, 79, 133); LD(21, 80, 389); LD(22, 81, 134); LD(23, 82, 390); LD(24, 84, 8); LD(25, 85, 264); LD(26, 86, 9); LD(27, 87, 265); LD(28, 88, 10); LD(29, 89, 266); LD(30, 91, 136); LD(31, 92, 392); LD(32, 93, 137); LD(33, 94, 393); LD(34, 95, 138); LD(35, 96, 394); LD(36, 105, 64); LD(37, 106, 320); LD(38, 107, 65); LD(39, 108, 321); LD(40, 109, 66); LD(41, 110, 322); LD(42, 112, 192); LD(43, 113, 448); LD(44, 114, 450); LD(45, 116, 68); LD(46, 117, 324); LD(47, 118, 326); LD(48, 120, 196); LD(49, 121, 452); LD(50, 122, 454); LD(51, 124, 72); LD(52, 125, 328); LD(53, 126, 73); LD(54, 127, 329); LD(55, 128, 74); LD(56, 129, 330); LD(57, 131, 200); LD(58, 132, 456); LD(59, 133, 201); LD(60, 134, 457); LD(61, 135, 202); LD(62, 136, 458); LD(63, 145, 16); LD(64, 146, 272); LD(65, 147, 17); LD(66, 148, 273); LD(67, 149, 18); LD(68, 150, 274); LD(69, 152, 144); LD(70, 153, 400); LD(71, 154, 402); LD(72, 156, 20); LD(73, 157, 276); LD(74, 158, 278); LD(75, 160, 148); LD(76, 161, 404); LD(77, 162, 406); LD(78, 164, 24); LD(79, 165, 280); LD(80, 166, 25); LD(81, 167, 281); LD(82, 168, 26); LD(83, 169, 282); LD(84, 171, 152); LD(85, 172, 408); LD(86, 173, 153); LD(87, 174, 409); LD(88, 175, 154); LD(89, 176, 410); LD(90, 185, 80); LD(91, 186, 336); LD(92, 187, 81); LD(93, 188, 337); LD(94, 189, 82); LD(95, 190, 338); LD(96, 192, 208); LD(97, 193, 464); LD(98, 194, 466); LD(99, 196, 84); LD(100, 197, 340); LD(101, 198, 342); LD(102, 200, 212); LD(103, 201, 468); LD(104, 202, 470); LD(105, 204, 88); LD(106, 205, 344); LD(107, 206, 89); LD(108, 207, 345); LD(109, 208, 90); LD(110, 209, 346); LD(111, 211, 216); LD(112, 212, 472); LD(113, 213, 217); LD(114, 214, 473); LD(115, 215, 218); LD(116, 216, 474); LD(117, 225, 32); LD(118, 226, 288); LD(119, 227, 33); LD(120, 228, 289); LD(121, 229, 34); LD(122, 230, 290); LD(123, 232, 160); LD(124, 233, 416); LD(125, 234, 161); LD(126, 235, 417); LD(127, 236, 162); LD(128, 237, 418); LD(129, 239, 36); LD(130, 240, 292); LD(131, 241, 37); LD(132, 242, 293); LD(133, 243, 38); LD(134, 244, 294); LD(135, 246, 164); LD(136, 247, 420); LD(137, 248, 165); LD(138, 249, 421); LD(139, 250, 166); LD(140, 251, 422); LD(141, 253, 40); LD(142, 254, 296); LD(143, 255, 41); LD(144, 256, 297); LD(145, 257, 42); LD(146, 258, 298); LD(147, 260, 168); LD(148, 261, 424); LD(149, 262, 169); LD(150, 263, 425); LD(151, 264, 170); LD(152, 265, 426); LD(153, 274, 96); LD(154, 275, 352); LD(155, 276, 97); LD(156, 277, 353); LD(157, 278, 98); LD(158, 279, 354); LD(159, 281, 224); LD(160, 282, 480); LD(161, 283, 225); LD(162, 284, 481); LD(163, 285, 226); LD(164, 286, 482); LD(165, 288, 100); LD(166, 289, 356); LD(167, 290, 101); LD(168, 291, 357); LD(169, 292, 102); LD(170, 293, 358); LD(171, 295, 228); LD(172, 296, 484); LD(173, 297, 229); LD(174, 298, 485); LD(175, 299, 230); LD(176, 300, 486); LD(177, 302, 104); LD(178, 303, 360); LD(179, 304, 105); LD(180, 305, 361); LD(181, 306, 106); LD(182, 307, 362); LD(183, 309, 232); LD(184, 310, 488); LD(185, 311, 233); LD(186, 312, 489); LD(187, 313, 234); LD(188, 314, 490)
#define B2_LOADXYZ5() LD(0, 7, 0); LD(1, 8, 256); LD(2, 9, 1); LD(3, 10, 257); LD(4, 11, 2); LD(5, 12, 258); LD(6, 14, 128); LD(7, 15, 384); LD(8, 16, 129); LD(9, 17, 385); LD(10, 18, 130); LD(11, 19, 386); LD(12, 21, 4); LD(13, 22, 260); LD(14, 23, 5); LD(15, 24, 261); LD(16, 25, 6); LD(17, 26, 262); LD(18, 28, 132); LD(19, 29, 388); LD(20, 30, 133); LD(21, 31, 389); LD(22, 32, 134); LD(23, 33, 390); LD(24, 35, 8); LD(25, 36, 264); LD(26, 37, 9); LD(27, 38, 265); LD(28, 39, 10); LD(29, 40, 266); LD(30, 42, 136); LD(31, 43, 392); LD(32, 44, 137); LD(33, 45, 393); LD(34, 46, 138); LD(35, 47, 394); LD(36, 56, 64); LD(37, 57, 320); LD(38, 58, 65); LD(39, 59, 321); LD(40, 60, 66); LD(41, 61, 322); LD(42, 63, 192); LD(43, 64, 448); LD(44, 65, 193); LD(45, 66, 449); LD(46, 67, 194); LD(47, 68, 450); LD(48, 70, 68); LD(49, 71, 324); LD(50, 72, 69); LD(51, 73, 325); LD(52, 74, 70); LD(53, 75, 326); LD(54, 77, 196); LD(55, 78, 452); LD(56, 79, 197); LD(57, 80, 453); LD(58, 81, 198); LD(59, 82, 454); LD(60, 84, 72); LD(61, 85, 328); LD(62, 86, 73); LD(63, 87, 329); LD(64, 88, 74); LD(65, 89, 330); LD(66, 91, 200); LD(67, 92, 456); LD(68, 93, 201); LD(69, 94, 457); LD(70, 95, 202); LD(71, 96, 458); LD(72, 105, 16); LD(73, 106, 272); LD(74, 107, 17); LD(75, 108, 273); LD(76, 109, 18); LD(77, 110, 274); LD(78, 112, 144); LD(79, 113, 400); LD(80, 114, 402); LD(81, 116, 20); LD(82, 117, 276); LD(83, 118, 278); LD(84, 120, 148); LD(85, 121, 404); LD(86, 122, 406); LD(87, 124, 24); LD(88, 125, 280); LD(89, 126, 25); LD(90, 127, 281); LD(91, 128, 26); LD(92, 129, 282); LD(93, 131, 152); LD(94, 132, 408); LD(95, 133, 153); LD(96, 134, 409); LD(97, 135, 154); LD(98, 136, 410); LD(99, 145, 80); LD(100, 146, 336); LD(101, 147, 81); LD(102, 148, 337); LD(103, 149, 82); LD(104, 150, 338); LD(105, 152, 208); LD(106, 153, 464); LD(107, 154, 466); LD(108, 156, 84); LD(109, 157, 340); LD(110, 158, 342); LD(111, 160, 212); LD(112, 161, 468); LD(113, 162, 470); LD(114, 164, 88); LD(115, 165, 344); LD(116, 166, 89); LD(117, 167, 345); LD(118, 168, 90); LD(119, 169, 346); LD(120, 171, 216); LD(121, 172, 472); LD(122, 173, 217); LD(123, 174, 473); LD(124, 175, 218); LD(125, 176, 474); LD(126, 185, 32); LD(127, 186, 288); LD(128, 187, 33); LD(129, 188, 289); LD(130, 189, 34); LD(131, 190, 290); LD(132, 192, 160); LD(133, 193, 416); LD(134, 194, 418); LD(135, 196, 36); LD(136, 197, 292); LD(137, 198, 294); LD(138, 200, 164); LD(139, 201, 420); LD(140, 202, 422); LD(141, 204, 40); LD(142, 205, 296); LD(143, 206, 41); LD(144, 207, 297); LD(145, 208, 42); LD(146, 209, 298); LD(147, 211, 168); LD(148, 212, 424); LD(149, 213, 169); LD(150, 214, 425); LD(151, 215, 170); LD(152, 216, 426); LD(153, 225, 96); LD(154, 226, 352); LD(155, 227, 97); LD(156, 228, 353); LD(157, 229, 98); LD(158, 230, 354); LD(159, 232, 224); LD(160, 233, 480); LD(161, 234, 225); LD(162, 235, 481); LD(163, 236, 226); LD(164, 237, 482); LD(165, 239, 100); LD(166, 240, 356); LD(167, 241, 101); LD(168, 242, 357); LD(169, 243, 102); LD(170, 244, 358); LD(171, 246, 228); LD(172, 247, 484); LD(173, 248, 229); LD(174, 249, 485); LD(175, 250, 230); LD(176, 251, 486); LD(177, 253, 104); LD(178, 254, 360); LD(179, 255, 105); LD(180, 256, 361); LD(181, 257, 106); LD(182, 258, 362); LD(183, 260, 232); LD(184, 261, 488); LD(185, 262, 233); LD(186, 263, 489); LD(187, 264, 234); LD(188, 265, 490)
#define B2_LOADXYZ6() LD(0, 49, 0); LD(1, 50, 256); LD(2, 51, 1); LD(3, 52, 257); LD(4, 53, 2); LD(5, 54, 258); LD(6, 56, 128); LD(7, 57, 384); LD(8, 58, 129); LD(9, 59, 385); LD(10, 60, 130); LD(11, 61, 386); LD(12, 63, 4); LD(13, 64, 260); LD(14, 65, 5); LD(15, 66, 261); LD(16, 67, 6); LD(17, 68, 262); LD(18, 70, 132); LD(19, 71, 388); LD(20, 72, 133); LD(21, 73, 389); LD(22, 74, 134); LD(23, 75, 390); LD(24, 77, 8); LD(25, 78, 264); LD(26, 79, 9); LD(27, 80, 265); LD(28, 81, 10); LD(29, 82, 266); LD(30, 84, 136); LD(31, 85, 392); LD(32, 86, 137); LD(33, 87, 393); LD(34, 88, 138); LD(35, 89, 394); LD(36, 98, 64); LD(37, 99, 320); LD(38, 100, 65); LD(39, 101, 321); LD(40, 102, 66); LD(41, 103, 322); LD(42, 105, 192); LD(43, 106, 448); LD(44, 107, 193); LD(45, 108, 449); LD(46, 109, 194); LD(47, 110, 450); LD(48, 112, 68); LD(49, 113, 324); LD(50, 114, 326); LD(51, 116, 196); LD(52, 117, 452); LD(53, 118, 454); LD(54, 120, 72); LD(55, 121, 328); LD(56, 122, 330); LD(57, 124, 200); LD(58, 125, 456); LD(59, 126, 201); LD(60, 127, 457); LD(61, 128, 202); LD(62, 129, 458); LD(63, 138, 16); LD(64, 139, 272); LD(65, 140, 17); LD(66, 141, 273); LD(67, 142, 18); LD(68, 143, 274); LD(69, 145, 144); LD(70, 146, 400); LD(71, 147, 145); LD(72, 148, 401); LD(73, 149, 146); LD(74, 150, 402); LD(75, 152, 20); LD(76, 153, 276); LD(77, 154, 278); LD(78, 156, 148); LD(79, 157, 404); LD(80, 158, 406); LD(81, 160, 24); LD(82, 161, 280); LD(83, 162, 282); LD(84, 164, 152); LD(85, 165, 408); LD(86, 166, 153); LD(87, 167, 409); LD(88, 168, 154); LD(89, 169, 410); LD(90, 178, 80); LD(91, 179, 336); LD(92, 180, 81); LD(93, 181, 337); LD(94, 182, 82); LD(95, 183, 338); LD(96, 185, 208); LD(97, 186, 464); LD(98, 187, 209); LD(99, 188, 465); LD(100, 189, 210); LD(101, 190, 466); LD(102, 192, 84); LD(103, 193, 340); LD(104, 194, 342); LD(105, 196, 212); LD(106, 197, 468); LD(107, 198, 470); LD(108, 200, 88); LD(109, 201, 344); LD(110, 202, 346); LD(111, 204, 216); LD(112, 205, 472); LD(113, 206, 217); LD(114, 207, 473); LD(115, 208, 218); LD(116, 209, 474); LD(117, 218, 32); LD(118, 219, 288); LD(119, 220, 33); LD(120, 221, 289); LD(121, 222, 34); LD(122, 223, 290); LD(123, 225, 160); LD(124, 226, 416); LD(125, 227, 161); LD(126, 228, 417); LD(127, 229, 162); LD(128, 230, 418); LD(129, 232, 36); LD(130, 233, 292); LD(131, 234, 37); LD(132, 235, 293); LD(133, 236, 38); LD(134, 237, 294); LD(135, 239, 164); LD(136, 240, 420); LD(137, 241, 165); LD(138, 242, 421); LD(139, 243, 166); LD(140, 244, 422); LD(141, 246, 40); LD(142, 247, 296); LD(143, 248, 41); LD(144, 249, 297); LD(145, 250, 42); LD(146, 251, 298); LD(147, 253, 168); LD(148, 254, 424); LD(149, 255, 169); LD(150, 256, 425); LD(151, 257, 170); LD(152, 258, 426); LD(153, 267, 96); LD(154, 268, 352); LD(155, 269, 97); LD(156, 270, 353); LD(157, 271, 98); LD(158, 272, 354); LD(159, 274, 224); LD(160, 275, 480); LD(161, 276, 225); LD(162, 277, 481); LD(163, 278, 226); LD(164, 279, 482); LD(165, 281, 100); LD(166, 282, 356); LD(167, 283, 101); LD(168, 284, 357); LD(169, 285, 102); LD(170, 286, 358); LD(171, 288, 228); LD(172, 289, 484); LD(173, 290, 229); LD(174, 291, 485); LD(175, 292, 230); LD(176, 293, 486); LD(177, 295, 104); LD(178, 296, 360); LD(179, 297, 105); LD(180, 298, 361); LD(181, 299, 106); LD(182, 300, 362); LD(183, 302, 232); LD(184, 303, 488); LD(185, 304, 233); LD(186, 305, 489); LD(187, 306, 234); LD(188, 307, 490)
#define B2_LOADXYZ7() LD(0, 0, 0); LD(1, 1, 256); LD(2, 2, 1); LD(3, 3, 257); LD(4, 4, 2); LD(5, 5, 258); LD(6, 7, 128); LD(7, 8, 384); LD(8, 9, 129); LD(9, 10, 385); LD(10, 11, 130); LD(11, 12, 386); LD(12, 14, 4); LD(13, 15, 260); LD(14, 16, 5); LD(15, 17, 261); LD(16, 18, 6); LD(17, 19, 262); LD(18, 21, 132); LD(19, 22, 388); LD(20, 23, 133); LD(21, 24, 389); LD(22, 25, 134); LD(23, 26, 390); LD(24, 28, 8); LD(25, 29, 264); LD(26, 30, 9); LD(27, 31, 265); LD(28, 32, 10); LD(29, 33, 266); LD(30, 35, 136); LD(31, 36, 392); LD(32, 37, 137); LD(33, 38, 393); LD(34, 39, 138); LD(35, 40, 394); LD(36, 49, 64); LD(37, 50, 320); LD(38, 51, 65); LD(39, 52, 321); LD(40, 53, 66); LD(41, 54, 322); LD(42, 56, 192); LD(43, 57, 448); LD(44, 58, 193); LD(45, 59, 449); LD(46, 60, 194); LD(47, 61, 450); LD(48, 63, 68); LD(49, 64, 324); LD(50, 65, 69); LD(51, 66, 325); LD(52, 67, 70); LD(53, 68, 326); LD(54, 70, 196); LD(55, 71, 452); LD(56, 72, 197); LD(57, 73, 453); LD(58, 74, 198); LD(59, 75, 454); LD(60, 77, 72); LD(61, 78, 328); LD(62, 79, 73); LD(63, 80, 329); LD(64, 81, 74); LD(65, 82, 330); LD(66, 84, 200); LD(67, 85, 456); LD(68, 86, 201); LD(69, 87, 457); LD(70, 88, 202); LD(71, 89, 458); LD(72, 98, 16); LD(73, 99, 272); LD(74, 100, 17); LD(75, 101, 273); LD(76, 102, 18); LD(77, 103, 274); LD(78, 105, 144); LD(79, 106, 400); LD(80, 107, 145); LD(81, 108, 401); LD(82, 109, 146); LD(83, 110, 402); LD(84, 112, 20); LD(85, 113, 276); LD(86, 114, 278); LD(87, 116, 148); LD(88, 117, 404); LD(89, 118, 406); LD(90, 120, 24); LD(91, 121, 280); LD(92, 122, 282); LD(93, 124, 152); LD(94, 125, 408); LD(95, 126, 153); LD(96, 127, 409); LD(97, 128, 154); LD(98, 129, 410); LD(99, 138, 80); LD(100, 139, 336); LD(101, 140, 81); LD(102, 141, 337); LD(103, 142, 82); LD(104, 143, 338); LD(105, 145, 208); LD(106, 146, 464); LD(107, 147, 209); LD(108, 148, 465); LD(109, 149, 210); LD(110, 150, 466); LD(111, 152, 84); LD(112, 153, 340); LD(113, 154, 342); LD(114, 156, 212); LD(115, 157, 468); LD(116, 158, 470); LD(117, 160, 88); LD(118, 161, 344); LD(119, 162, 346); LD(120, 164, 216); LD(121, 165, 472); LD(122, 166, 217); LD(123, 167, 473); LD(124, 168, 218); LD(125, 169, 474); LD(126, 178, 32); LD(127, 179, 288); LD(128, 180, 33); LD(129, 181, 289); LD(130, 182, 34); LD(131, 183, 290); LD(132, 185, 160); LD(133, 186, 416); LD(134, 187, 161); LD(135, 188, 417); LD(136, 189, 162); LD(137, 190, 418); LD(138, 192, 36); LD(139, 193, 292); LD(140, 194, 294); LD(141, 196, 164); LD(142, 197, 420); LD(143, 198, 422); LD(144, 200, 40); LD(145, 201, 296); LD(146, 202, 298); LD(147, 204, 168); LD(148, 205, 424); LD(149, 206, 169); LD(150, 207, 425); LD(151, 208, 170); LD(152, 209, 426); LD(153, 218, 96); LD(154, 219, 352); LD(155, 220, 97); LD(156, 221, 353); LD(157, 222, 98); LD(158, 223, 354); LD(159, 225, 224); LD(160, 226, 480); LD(161, 227, 225); LD(162, 228, 481); LD(163, 229, 226); LD(164, 230, 482); LD(165, 232, 100); LD(166, 233, 356); LD(167, 234, 101); LD(168, 235, 357); LD(169, 236, 102); LD(170, 237, 358); LD(171, 239, 228); LD(172, 240, 484); LD(173, 241, 229); LD(174, 242, 485); LD(175, 243, 230); LD(176, 244, 486); LD(177, 246, 104); LD(178, 247, 360); LD(179, 248, 105); LD(180, 249, 361); LD(181, 250, 106); LD(182, 251, 362); LD(183, 253, 232); LD(184, 254, 488); LD(185, 255, 233); LD(186, 256, 489); LD(187, 257, 234); LD(188, 258, 490)

static void accumulate(real *Kijtmp, real *Mjtmp, real *Lptr)
{
  real Lij = ZERO;
  for (int k = 0; k < 189; k ++) { // LOOP WAS VECTORIZED.
    Lij += Kijtmp[k] * Mjtmp[k];
  }
  *Lptr += Lij;
}

static void comp_chunk_coordinates(const int level, const int B, const int bx, int *cx, int *cy, int *cz)
{
  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);
  
  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  *cx = bx % nch;
  *cy = (bx % (nch * nch)) / nch;
  *cz = bx / (nch * nch);

}

static void m2l_kern_ij_blocking_b4(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) with the same
     sibling-index along each direction for this level */
  const int ncpec = POW2(level - 1) + 2;

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0*/
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][s][B*cz+iz][B*cy+iy][B*cx+ix], where j=s=ix=iy=iz=0 */
  real *Mptr = M + Mstart + (((0 * 8 + 0) * ncpec + (B * cz + 0)) * ncpec + (B * cy + 0)) * ncpec + (B * cx + 0);

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[8][B + 2][B + 2][B + 2]; // cached?
    
    for (int s = 0; s < 8; s ++) { // sibling-index for source cells
      for (int iz = 0; iz < B + 2; iz ++) {
	for (int iy = 0; iy < B + 2; iy ++) {
	  for (int ix = 0; ix < B + 2; ix ++) { // LOOP WAS VECTORIZED.
	    Mj[s][iz][iy][ix] = Mptr[((s * ncpec + iz) * ncpec + iy) * ncpec + ix];
	  }
	}
      }
    }

    /* Point to next j */
    Mptr += 8 * ncpec * ncpec * ncpec;
    
    /* Set a pointer to L;
       L[chunk][i][iz][iy][ix][sib], where chunk=bx and i=iz=iy=ix=sib=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * B + 0) * B + 0) * B + 0) * 8 + 0;
    
    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Load Kij */
      real Kij[316]; // cached?
      for (int k = 0; k < 316; k ++) { // LOOP WAS VECTORIZED.
	Kij[k] = Kptr[k];
      }

      /* Point to next i */
      Kptr += 316;
     
      /* Loop over target cells with the same sibling-index */
      for (int iz = 0; iz < B; iz ++) {
	for (int iy = 0; iy < B; iy ++) {
	  for (int ix = 0; ix < B; ix ++) {
	    
	    /* Offset */
	    const int Mjshift = (iz * (B + 2) + iy) * (B + 2) + ix;

	    /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	       S) and accumulate Lij(F) to Li(F) (reduction for j) */
	    real Kijtmp[189], Mjtmp[189];
	    const real *Mjptr = (real *)Mj + Mjshift;
	    
	    /* Loop over sibling-indices of target cells */

	    B4_LOADXYZ0();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B4_LOADXYZ1();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B4_LOADXYZ2();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B4_LOADXYZ3();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B4_LOADXYZ4();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B4_LOADXYZ5();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B4_LOADXYZ6();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B4_LOADXYZ7();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	  } // ix
	} // iy
      } // iz

    } // i
  } // j
}

static void m2l_kern_ij_blocking_b2(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) with the same
     sibling-index along each direction for this level */
  const int ncpec = POW2(level - 1) + 2;

  /* Compute the coordinates of this chunk */
  int cx, cy, cz;
  comp_chunk_coordinates(level, B, bx, &cx, &cy, &cz);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0*/
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][s][B*cz+iz][B*cy+iy][B*cx+ix], where j=s=ix=iy=iz=0 */
  real *Mptr = M + Mstart + (((0 * 8 + 0) * ncpec + (B * cz + 0)) * ncpec + (B * cy + 0)) * ncpec + (B * cx + 0);

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[8][B + 2][B + 2][B + 2]; // cached?
    
    for (int s = 0; s < 8; s ++) { // sibling-index for source cells
      for (int iz = 0; iz < B + 2; iz ++) {
	for (int iy = 0; iy < B + 2; iy ++) {
	  for (int ix = 0; ix < B + 2; ix ++) { // LOOP WAS VECTORIZED.
	    Mj[s][iz][iy][ix] = Mptr[((s * ncpec + iz) * ncpec + iy) * ncpec + ix];
	  }
	}
      }
    }

    /* Point to next j */
    Mptr += 8 * ncpec * ncpec * ncpec;
    
    /* Set a pointer to L;
       L[chunk][i][iz][iy][ix][sib], where chunk=bx and i=iz=iy=ix=sib=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * B + 0) * B + 0) * B + 0) * 8 + 0;
    
    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Load Kij */
      real Kij[316]; // cached?
      for (int k = 0; k < 316; k ++) { // LOOP WAS VECTORIZED.
	Kij[k] = Kptr[k];
      }

      /* Point to next i */
      Kptr += 316;
     
      /* Loop over target cells with the same sibling-index */
      for (int iz = 0; iz < B; iz ++) {
	for (int iy = 0; iy < B; iy ++) {
	  for (int ix = 0; ix < B; ix ++) {
	    
	    /* Offset */
	    const int Mjshift = (iz * (B + 2) + iy) * (B + 2) + ix;

	    /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	       S) and accumulate Lij(F) to Li(F) (reduction for j) */
	    real Kijtmp[189], Mjtmp[189];
	    const real *Mjptr = (real *)Mj + Mjshift;
	    
	    /* Loop over sibling-indices of target cells */

	    B2_LOADXYZ0();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B2_LOADXYZ1();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B2_LOADXYZ2();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B2_LOADXYZ3();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B2_LOADXYZ4();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B2_LOADXYZ5();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B2_LOADXYZ6();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    B2_LOADXYZ7();
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	  } // ix
	} // iy
      } // iz

    } // i
  } // j
}
/**************************************************************************/
#elif defined(CPU9B)
/**************************************************************************/
/* Based on CPU9A */

#define LD(i, Kijoff, Mjoff)				\
  Kijtmp[i] = Kij[Kijoff];				\
  Mjtmp[i] = *(Mjptr + Mjoff);

#define B4_LOADXYZ0() LD(0, 57, 0); LD(1, 58, 864); LD(2, 59, 1); LD(3, 60, 865); LD(4, 61, 2); LD(5, 62, 866); LD(6, 64, 432); LD(7, 65, 1296); LD(8, 66, 433); LD(9, 67, 1297); LD(10, 68, 434); LD(11, 69, 1298); LD(12, 71, 6); LD(13, 72, 870); LD(14, 73, 7); LD(15, 74, 871); LD(16, 75, 8); LD(17, 76, 872); LD(18, 78, 438); LD(19, 79, 1302); LD(20, 80, 439); LD(21, 81, 1303); LD(22, 82, 440); LD(23, 83, 1304); LD(24, 85, 12); LD(25, 86, 876); LD(26, 87, 13); LD(27, 88, 877); LD(28, 89, 14); LD(29, 90, 878); LD(30, 92, 444); LD(31, 93, 1308); LD(32, 94, 445); LD(33, 95, 1309); LD(34, 96, 446); LD(35, 97, 1310); LD(36, 106, 216); LD(37, 107, 1080); LD(38, 108, 217); LD(39, 109, 1081); LD(40, 110, 218); LD(41, 111, 1082); LD(42, 113, 648); LD(43, 114, 650); LD(44, 115, 1514); LD(45, 117, 222); LD(46, 118, 224); LD(47, 119, 1088); LD(48, 121, 654); LD(49, 122, 656); LD(50, 123, 1520); LD(51, 125, 228); LD(52, 126, 1092); LD(53, 127, 229); LD(54, 128, 1093); LD(55, 129, 230); LD(56, 130, 1094); LD(57, 132, 660); LD(58, 133, 1524); LD(59, 134, 661); LD(60, 135, 1525); LD(61, 136, 662); LD(62, 137, 1526); LD(63, 146, 36); LD(64, 147, 900); LD(65, 148, 37); LD(66, 149, 901); LD(67, 150, 38); LD(68, 151, 902); LD(69, 153, 468); LD(70, 154, 470); LD(71, 155, 1334); LD(72, 157, 42); LD(73, 158, 44); LD(74, 159, 908); LD(75, 161, 474); LD(76, 162, 476); LD(77, 163, 1340); LD(78, 165, 48); LD(79, 166, 912); LD(80, 167, 49); LD(81, 168, 913); LD(82, 169, 50); LD(83, 170, 914); LD(84, 172, 480); LD(85, 173, 1344); LD(86, 174, 481); LD(87, 175, 1345); LD(88, 176, 482); LD(89, 177, 1346); LD(90, 186, 252); LD(91, 187, 1116); LD(92, 188, 253); LD(93, 189, 1117); LD(94, 190, 254); LD(95, 191, 1118); LD(96, 193, 684); LD(97, 194, 686); LD(98, 195, 1550); LD(99, 197, 258); LD(100, 198, 260); LD(101, 199, 1124); LD(102, 201, 690); LD(103, 202, 692); LD(104, 203, 1556); LD(105, 205, 264); LD(106, 206, 1128); LD(107, 207, 265); LD(108, 208, 1129); LD(109, 209, 266); LD(110, 210, 1130); LD(111, 212, 696); LD(112, 213, 1560); LD(113, 214, 697); LD(114, 215, 1561); LD(115, 216, 698); LD(116, 217, 1562); LD(117, 226, 72); LD(118, 227, 936); LD(119, 228, 73); LD(120, 229, 937); LD(121, 230, 74); LD(122, 231, 938); LD(123, 233, 504); LD(124, 234, 1368); LD(125, 235, 505); LD(126, 236, 1369); LD(127, 237, 506); LD(128, 238, 1370); LD(129, 240, 78); LD(130, 241, 942); LD(131, 242, 79); LD(132, 243, 943); LD(133, 244, 80); LD(134, 245, 944); LD(135, 247, 510); LD(136, 248, 1374); LD(137, 249, 511); LD(138, 250, 1375); LD(139, 251, 512); LD(140, 252, 1376); LD(141, 254, 84); LD(142, 255, 948); LD(143, 256, 85); LD(144, 257, 949); LD(145, 258, 86); LD(146, 259, 950); LD(147, 261, 516); LD(148, 262, 1380); LD(149, 263, 517); LD(150, 264, 1381); LD(151, 265, 518); LD(152, 266, 1382); LD(153, 275, 288); LD(154, 276, 1152); LD(155, 277, 289); LD(156, 278, 1153); LD(157, 279, 290); LD(158, 280, 1154); LD(159, 282, 720); LD(160, 283, 1584); LD(161, 284, 721); LD(162, 285, 1585); LD(163, 286, 722); LD(164, 287, 1586); LD(165, 289, 294); LD(166, 290, 1158); LD(167, 291, 295); LD(168, 292, 1159); LD(169, 293, 296); LD(170, 294, 1160); LD(171, 296, 726); LD(172, 297, 1590); LD(173, 298, 727); LD(174, 299, 1591); LD(175, 300, 728); LD(176, 301, 1592); LD(177, 303, 300); LD(178, 304, 1164); LD(179, 305, 301); LD(180, 306, 1165); LD(181, 307, 302); LD(182, 308, 1166); LD(183, 310, 732); LD(184, 311, 1596); LD(185, 312, 733); LD(186, 313, 1597); LD(187, 314, 734); LD(188, 315, 1598)
#define B4_LOADXYZ1() LD(0, 8, 0); LD(1, 9, 864); LD(2, 10, 1); LD(3, 11, 865); LD(4, 12, 2); LD(5, 13, 866); LD(6, 15, 432); LD(7, 16, 1296); LD(8, 17, 433); LD(9, 18, 1297); LD(10, 19, 434); LD(11, 20, 1298); LD(12, 22, 6); LD(13, 23, 870); LD(14, 24, 7); LD(15, 25, 871); LD(16, 26, 8); LD(17, 27, 872); LD(18, 29, 438); LD(19, 30, 1302); LD(20, 31, 439); LD(21, 32, 1303); LD(22, 33, 440); LD(23, 34, 1304); LD(24, 36, 12); LD(25, 37, 876); LD(26, 38, 13); LD(27, 39, 877); LD(28, 40, 14); LD(29, 41, 878); LD(30, 43, 444); LD(31, 44, 1308); LD(32, 45, 445); LD(33, 46, 1309); LD(34, 47, 446); LD(35, 48, 1310); LD(36, 57, 216); LD(37, 58, 1080); LD(38, 59, 217); LD(39, 60, 1081); LD(40, 61, 218); LD(41, 62, 1082); LD(42, 64, 648); LD(43, 65, 1512); LD(44, 66, 649); LD(45, 67, 1513); LD(46, 68, 650); LD(47, 69, 1514); LD(48, 71, 222); LD(49, 72, 1086); LD(50, 73, 223); LD(51, 74, 1087); LD(52, 75, 224); LD(53, 76, 1088); LD(54, 78, 654); LD(55, 79, 1518); LD(56, 80, 655); LD(57, 81, 1519); LD(58, 82, 656); LD(59, 83, 1520); LD(60, 85, 228); LD(61, 86, 1092); LD(62, 87, 229); LD(63, 88, 1093); LD(64, 89, 230); LD(65, 90, 1094); LD(66, 92, 660); LD(67, 93, 1524); LD(68, 94, 661); LD(69, 95, 1525); LD(70, 96, 662); LD(71, 97, 1526); LD(72, 106, 36); LD(73, 107, 900); LD(74, 108, 37); LD(75, 109, 901); LD(76, 110, 38); LD(77, 111, 902); LD(78, 113, 468); LD(79, 114, 470); LD(80, 115, 1334); LD(81, 117, 42); LD(82, 118, 44); LD(83, 119, 908); LD(84, 121, 474); LD(85, 122, 476); LD(86, 123, 1340); LD(87, 125, 48); LD(88, 126, 912); LD(89, 127, 49); LD(90, 128, 913); LD(91, 129, 50); LD(92, 130, 914); LD(93, 132, 480); LD(94, 133, 1344); LD(95, 134, 481); LD(96, 135, 1345); LD(97, 136, 482); LD(98, 137, 1346); LD(99, 146, 252); LD(100, 147, 1116); LD(101, 148, 253); LD(102, 149, 1117); LD(103, 150, 254); LD(104, 151, 1118); LD(105, 153, 684); LD(106, 154, 686); LD(107, 155, 1550); LD(108, 157, 258); LD(109, 158, 260); LD(110, 159, 1124); LD(111, 161, 690); LD(112, 162, 692); LD(113, 163, 1556); LD(114, 165, 264); LD(115, 166, 1128); LD(116, 167, 265); LD(117, 168, 1129); LD(118, 169, 266); LD(119, 170, 1130); LD(120, 172, 696); LD(121, 173, 1560); LD(122, 174, 697); LD(123, 175, 1561); LD(124, 176, 698); LD(125, 177, 1562); LD(126, 186, 72); LD(127, 187, 936); LD(128, 188, 73); LD(129, 189, 937); LD(130, 190, 74); LD(131, 191, 938); LD(132, 193, 504); LD(133, 194, 506); LD(134, 195, 1370); LD(135, 197, 78); LD(136, 198, 80); LD(137, 199, 944); LD(138, 201, 510); LD(139, 202, 512); LD(140, 203, 1376); LD(141, 205, 84); LD(142, 206, 948); LD(143, 207, 85); LD(144, 208, 949); LD(145, 209, 86); LD(146, 210, 950); LD(147, 212, 516); LD(148, 213, 1380); LD(149, 214, 517); LD(150, 215, 1381); LD(151, 216, 518); LD(152, 217, 1382); LD(153, 226, 288); LD(154, 227, 1152); LD(155, 228, 289); LD(156, 229, 1153); LD(157, 230, 290); LD(158, 231, 1154); LD(159, 233, 720); LD(160, 234, 1584); LD(161, 235, 721); LD(162, 236, 1585); LD(163, 237, 722); LD(164, 238, 1586); LD(165, 240, 294); LD(166, 241, 1158); LD(167, 242, 295); LD(168, 243, 1159); LD(169, 244, 296); LD(170, 245, 1160); LD(171, 247, 726); LD(172, 248, 1590); LD(173, 249, 727); LD(174, 250, 1591); LD(175, 251, 728); LD(176, 252, 1592); LD(177, 254, 300); LD(178, 255, 1164); LD(179, 256, 301); LD(180, 257, 1165); LD(181, 258, 302); LD(182, 259, 1166); LD(183, 261, 732); LD(184, 262, 1596); LD(185, 263, 733); LD(186, 264, 1597); LD(187, 265, 734); LD(188, 266, 1598)
#define B4_LOADXYZ2() LD(0, 50, 0); LD(1, 51, 864); LD(2, 52, 1); LD(3, 53, 865); LD(4, 54, 2); LD(5, 55, 866); LD(6, 57, 432); LD(7, 58, 1296); LD(8, 59, 433); LD(9, 60, 1297); LD(10, 61, 434); LD(11, 62, 1298); LD(12, 64, 6); LD(13, 65, 870); LD(14, 66, 7); LD(15, 67, 871); LD(16, 68, 8); LD(17, 69, 872); LD(18, 71, 438); LD(19, 72, 1302); LD(20, 73, 439); LD(21, 74, 1303); LD(22, 75, 440); LD(23, 76, 1304); LD(24, 78, 12); LD(25, 79, 876); LD(26, 80, 13); LD(27, 81, 877); LD(28, 82, 14); LD(29, 83, 878); LD(30, 85, 444); LD(31, 86, 1308); LD(32, 87, 445); LD(33, 88, 1309); LD(34, 89, 446); LD(35, 90, 1310); LD(36, 99, 216); LD(37, 100, 1080); LD(38, 101, 217); LD(39, 102, 1081); LD(40, 103, 218); LD(41, 104, 1082); LD(42, 106, 648); LD(43, 107, 1512); LD(44, 108, 649); LD(45, 109, 1513); LD(46, 110, 650); LD(47, 111, 1514); LD(48, 113, 222); LD(49, 114, 224); LD(50, 115, 1088); LD(51, 117, 654); LD(52, 118, 656); LD(53, 119, 1520); LD(54, 121, 228); LD(55, 122, 230); LD(56, 123, 1094); LD(57, 125, 660); LD(58, 126, 1524); LD(59, 127, 661); LD(60, 128, 1525); LD(61, 129, 662); LD(62, 130, 1526); LD(63, 139, 36); LD(64, 140, 900); LD(65, 141, 37); LD(66, 142, 901); LD(67, 143, 38); LD(68, 144, 902); LD(69, 146, 468); LD(70, 147, 1332); LD(71, 148, 469); LD(72, 149, 1333); LD(73, 150, 470); LD(74, 151, 1334); LD(75, 153, 42); LD(76, 154, 44); LD(77, 155, 908); LD(78, 157, 474); LD(79, 158, 476); LD(80, 159, 1340); LD(81, 161, 48); LD(82, 162, 50); LD(83, 163, 914); LD(84, 165, 480); LD(85, 166, 1344); LD(86, 167, 481); LD(87, 168, 1345); LD(88, 169, 482); LD(89, 170, 1346); LD(90, 179, 252); LD(91, 180, 1116); LD(92, 181, 253); LD(93, 182, 1117); LD(94, 183, 254); LD(95, 184, 1118); LD(96, 186, 684); LD(97, 187, 1548); LD(98, 188, 685); LD(99, 189, 1549); LD(100, 190, 686); LD(101, 191, 1550); LD(102, 193, 258); LD(103, 194, 260); LD(104, 195, 1124); LD(105, 197, 690); LD(106, 198, 692); LD(107, 199, 1556); LD(108, 201, 264); LD(109, 202, 266); LD(110, 203, 1130); LD(111, 205, 696); LD(112, 206, 1560); LD(113, 207, 697); LD(114, 208, 1561); LD(115, 209, 698); LD(116, 210, 1562); LD(117, 219, 72); LD(118, 220, 936); LD(119, 221, 73); LD(120, 222, 937); LD(121, 223, 74); LD(122, 224, 938); LD(123, 226, 504); LD(124, 227, 1368); LD(125, 228, 505); LD(126, 229, 1369); LD(127, 230, 506); LD(128, 231, 1370); LD(129, 233, 78); LD(130, 234, 942); LD(131, 235, 79); LD(132, 236, 943); LD(133, 237, 80); LD(134, 238, 944); LD(135, 240, 510); LD(136, 241, 1374); LD(137, 242, 511); LD(138, 243, 1375); LD(139, 244, 512); LD(140, 245, 1376); LD(141, 247, 84); LD(142, 248, 948); LD(143, 249, 85); LD(144, 250, 949); LD(145, 251, 86); LD(146, 252, 950); LD(147, 254, 516); LD(148, 255, 1380); LD(149, 256, 517); LD(150, 257, 1381); LD(151, 258, 518); LD(152, 259, 1382); LD(153, 268, 288); LD(154, 269, 1152); LD(155, 270, 289); LD(156, 271, 1153); LD(157, 272, 290); LD(158, 273, 1154); LD(159, 275, 720); LD(160, 276, 1584); LD(161, 277, 721); LD(162, 278, 1585); LD(163, 279, 722); LD(164, 280, 1586); LD(165, 282, 294); LD(166, 283, 1158); LD(167, 284, 295); LD(168, 285, 1159); LD(169, 286, 296); LD(170, 287, 1160); LD(171, 289, 726); LD(172, 290, 1590); LD(173, 291, 727); LD(174, 292, 1591); LD(175, 293, 728); LD(176, 294, 1592); LD(177, 296, 300); LD(178, 297, 1164); LD(179, 298, 301); LD(180, 299, 1165); LD(181, 300, 302); LD(182, 301, 1166); LD(183, 303, 732); LD(184, 304, 1596); LD(185, 305, 733); LD(186, 306, 1597); LD(187, 307, 734); LD(188, 308, 1598)
#define B4_LOADXYZ3() LD(0, 1, 0); LD(1, 2, 864); LD(2, 3, 1); LD(3, 4, 865); LD(4, 5, 2); LD(5, 6, 866); LD(6, 8, 432); LD(7, 9, 1296); LD(8, 10, 433); LD(9, 11, 1297); LD(10, 12, 434); LD(11, 13, 1298); LD(12, 15, 6); LD(13, 16, 870); LD(14, 17, 7); LD(15, 18, 871); LD(16, 19, 8); LD(17, 20, 872); LD(18, 22, 438); LD(19, 23, 1302); LD(20, 24, 439); LD(21, 25, 1303); LD(22, 26, 440); LD(23, 27, 1304); LD(24, 29, 12); LD(25, 30, 876); LD(26, 31, 13); LD(27, 32, 877); LD(28, 33, 14); LD(29, 34, 878); LD(30, 36, 444); LD(31, 37, 1308); LD(32, 38, 445); LD(33, 39, 1309); LD(34, 40, 446); LD(35, 41, 1310); LD(36, 50, 216); LD(37, 51, 1080); LD(38, 52, 217); LD(39, 53, 1081); LD(40, 54, 218); LD(41, 55, 1082); LD(42, 57, 648); LD(43, 58, 1512); LD(44, 59, 649); LD(45, 60, 1513); LD(46, 61, 650); LD(47, 62, 1514); LD(48, 64, 222); LD(49, 65, 1086); LD(50, 66, 223); LD(51, 67, 1087); LD(52, 68, 224); LD(53, 69, 1088); LD(54, 71, 654); LD(55, 72, 1518); LD(56, 73, 655); LD(57, 74, 1519); LD(58, 75, 656); LD(59, 76, 1520); LD(60, 78, 228); LD(61, 79, 1092); LD(62, 80, 229); LD(63, 81, 1093); LD(64, 82, 230); LD(65, 83, 1094); LD(66, 85, 660); LD(67, 86, 1524); LD(68, 87, 661); LD(69, 88, 1525); LD(70, 89, 662); LD(71, 90, 1526); LD(72, 99, 36); LD(73, 100, 900); LD(74, 101, 37); LD(75, 102, 901); LD(76, 103, 38); LD(77, 104, 902); LD(78, 106, 468); LD(79, 107, 1332); LD(80, 108, 469); LD(81, 109, 1333); LD(82, 110, 470); LD(83, 111, 1334); LD(84, 113, 42); LD(85, 114, 44); LD(86, 115, 908); LD(87, 117, 474); LD(88, 118, 476); LD(89, 119, 1340); LD(90, 121, 48); LD(91, 122, 50); LD(92, 123, 914); LD(93, 125, 480); LD(94, 126, 1344); LD(95, 127, 481); LD(96, 128, 1345); LD(97, 129, 482); LD(98, 130, 1346); LD(99, 139, 252); LD(100, 140, 1116); LD(101, 141, 253); LD(102, 142, 1117); LD(103, 143, 254); LD(104, 144, 1118); LD(105, 146, 684); LD(106, 147, 1548); LD(107, 148, 685); LD(108, 149, 1549); LD(109, 150, 686); LD(110, 151, 1550); LD(111, 153, 258); LD(112, 154, 260); LD(113, 155, 1124); LD(114, 157, 690); LD(115, 158, 692); LD(116, 159, 1556); LD(117, 161, 264); LD(118, 162, 266); LD(119, 163, 1130); LD(120, 165, 696); LD(121, 166, 1560); LD(122, 167, 697); LD(123, 168, 1561); LD(124, 169, 698); LD(125, 170, 1562); LD(126, 179, 72); LD(127, 180, 936); LD(128, 181, 73); LD(129, 182, 937); LD(130, 183, 74); LD(131, 184, 938); LD(132, 186, 504); LD(133, 187, 1368); LD(134, 188, 505); LD(135, 189, 1369); LD(136, 190, 506); LD(137, 191, 1370); LD(138, 193, 78); LD(139, 194, 80); LD(140, 195, 944); LD(141, 197, 510); LD(142, 198, 512); LD(143, 199, 1376); LD(144, 201, 84); LD(145, 202, 86); LD(146, 203, 950); LD(147, 205, 516); LD(148, 206, 1380); LD(149, 207, 517); LD(150, 208, 1381); LD(151, 209, 518); LD(152, 210, 1382); LD(153, 219, 288); LD(154, 220, 1152); LD(155, 221, 289); LD(156, 222, 1153); LD(157, 223, 290); LD(158, 224, 1154); LD(159, 226, 720); LD(160, 227, 1584); LD(161, 228, 721); LD(162, 229, 1585); LD(163, 230, 722); LD(164, 231, 1586); LD(165, 233, 294); LD(166, 234, 1158); LD(167, 235, 295); LD(168, 236, 1159); LD(169, 237, 296); LD(170, 238, 1160); LD(171, 240, 726); LD(172, 241, 1590); LD(173, 242, 727); LD(174, 243, 1591); LD(175, 244, 728); LD(176, 245, 1592); LD(177, 247, 300); LD(178, 248, 1164); LD(179, 249, 301); LD(180, 250, 1165); LD(181, 251, 302); LD(182, 252, 1166); LD(183, 254, 732); LD(184, 255, 1596); LD(185, 256, 733); LD(186, 257, 1597); LD(187, 258, 734); LD(188, 259, 1598)
#define B4_LOADXYZ4() LD(0, 56, 0); LD(1, 57, 864); LD(2, 58, 1); LD(3, 59, 865); LD(4, 60, 2); LD(5, 61, 866); LD(6, 63, 432); LD(7, 64, 1296); LD(8, 65, 433); LD(9, 66, 1297); LD(10, 67, 434); LD(11, 68, 1298); LD(12, 70, 6); LD(13, 71, 870); LD(14, 72, 7); LD(15, 73, 871); LD(16, 74, 8); LD(17, 75, 872); LD(18, 77, 438); LD(19, 78, 1302); LD(20, 79, 439); LD(21, 80, 1303); LD(22, 81, 440); LD(23, 82, 1304); LD(24, 84, 12); LD(25, 85, 876); LD(26, 86, 13); LD(27, 87, 877); LD(28, 88, 14); LD(29, 89, 878); LD(30, 91, 444); LD(31, 92, 1308); LD(32, 93, 445); LD(33, 94, 1309); LD(34, 95, 446); LD(35, 96, 1310); LD(36, 105, 216); LD(37, 106, 1080); LD(38, 107, 217); LD(39, 108, 1081); LD(40, 109, 218); LD(41, 110, 1082); LD(42, 112, 648); LD(43, 113, 1512); LD(44, 114, 1514); LD(45, 116, 222); LD(46, 117, 1086); LD(47, 118, 1088); LD(48, 120, 654); LD(49, 121, 1518); LD(50, 122, 1520); LD(51, 124, 228); LD(52, 125, 1092); LD(53, 126, 229); LD(54, 127, 1093); LD(55, 128, 230); LD(56, 129, 1094); LD(57, 131, 660); LD(58, 132, 1524); LD(59, 133, 661); LD(60, 134, 1525); LD(61, 135, 662); LD(62, 136, 1526); LD(63, 145, 36); LD(64, 146, 900); LD(65, 147, 37); LD(66, 148, 901); LD(67, 149, 38); LD(68, 150, 902); LD(69, 152, 468); LD(70, 153, 1332); LD(71, 154, 1334); LD(72, 156, 42); LD(73, 157, 906); LD(74, 158, 908); LD(75, 160, 474); LD(76, 161, 1338); LD(77, 162, 1340); LD(78, 164, 48); LD(79, 165, 912); LD(80, 166, 49); LD(81, 167, 913); LD(82, 168, 50); LD(83, 169, 914); LD(84, 171, 480); LD(85, 172, 1344); LD(86, 173, 481); LD(87, 174, 1345); LD(88, 175, 482); LD(89, 176, 1346); LD(90, 185, 252); LD(91, 186, 1116); LD(92, 187, 253); LD(93, 188, 1117); LD(94, 189, 254); LD(95, 190, 1118); LD(96, 192, 684); LD(97, 193, 1548); LD(98, 194, 1550); LD(99, 196, 258); LD(100, 197, 1122); LD(101, 198, 1124); LD(102, 200, 690); LD(103, 201, 1554); LD(104, 202, 1556); LD(105, 204, 264); LD(106, 205, 1128); LD(107, 206, 265); LD(108, 207, 1129); LD(109, 208, 266); LD(110, 209, 1130); LD(111, 211, 696); LD(112, 212, 1560); LD(113, 213, 697); LD(114, 214, 1561); LD(115, 215, 698); LD(116, 216, 1562); LD(117, 225, 72); LD(118, 226, 936); LD(119, 227, 73); LD(120, 228, 937); LD(121, 229, 74); LD(122, 230, 938); LD(123, 232, 504); LD(124, 233, 1368); LD(125, 234, 505); LD(126, 235, 1369); LD(127, 236, 506); LD(128, 237, 1370); LD(129, 239, 78); LD(130, 240, 942); LD(131, 241, 79); LD(132, 242, 943); LD(133, 243, 80); LD(134, 244, 944); LD(135, 246, 510); LD(136, 247, 1374); LD(137, 248, 511); LD(138, 249, 1375); LD(139, 250, 512); LD(140, 251, 1376); LD(141, 253, 84); LD(142, 254, 948); LD(143, 255, 85); LD(144, 256, 949); LD(145, 257, 86); LD(146, 258, 950); LD(147, 260, 516); LD(148, 261, 1380); LD(149, 262, 517); LD(150, 263, 1381); LD(151, 264, 518); LD(152, 265, 1382); LD(153, 274, 288); LD(154, 275, 1152); LD(155, 276, 289); LD(156, 277, 1153); LD(157, 278, 290); LD(158, 279, 1154); LD(159, 281, 720); LD(160, 282, 1584); LD(161, 283, 721); LD(162, 284, 1585); LD(163, 285, 722); LD(164, 286, 1586); LD(165, 288, 294); LD(166, 289, 1158); LD(167, 290, 295); LD(168, 291, 1159); LD(169, 292, 296); LD(170, 293, 1160); LD(171, 295, 726); LD(172, 296, 1590); LD(173, 297, 727); LD(174, 298, 1591); LD(175, 299, 728); LD(176, 300, 1592); LD(177, 302, 300); LD(178, 303, 1164); LD(179, 304, 301); LD(180, 305, 1165); LD(181, 306, 302); LD(182, 307, 1166); LD(183, 309, 732); LD(184, 310, 1596); LD(185, 311, 733); LD(186, 312, 1597); LD(187, 313, 734); LD(188, 314, 1598)
#define B4_LOADXYZ5() LD(0, 7, 0); LD(1, 8, 864); LD(2, 9, 1); LD(3, 10, 865); LD(4, 11, 2); LD(5, 12, 866); LD(6, 14, 432); LD(7, 15, 1296); LD(8, 16, 433); LD(9, 17, 1297); LD(10, 18, 434); LD(11, 19, 1298); LD(12, 21, 6); LD(13, 22, 870); LD(14, 23, 7); LD(15, 24, 871); LD(16, 25, 8); LD(17, 26, 872); LD(18, 28, 438); LD(19, 29, 1302); LD(20, 30, 439); LD(21, 31, 1303); LD(22, 32, 440); LD(23, 33, 1304); LD(24, 35, 12); LD(25, 36, 876); LD(26, 37, 13); LD(27, 38, 877); LD(28, 39, 14); LD(29, 40, 878); LD(30, 42, 444); LD(31, 43, 1308); LD(32, 44, 445); LD(33, 45, 1309); LD(34, 46, 446); LD(35, 47, 1310); LD(36, 56, 216); LD(37, 57, 1080); LD(38, 58, 217); LD(39, 59, 1081); LD(40, 60, 218); LD(41, 61, 1082); LD(42, 63, 648); LD(43, 64, 1512); LD(44, 65, 649); LD(45, 66, 1513); LD(46, 67, 650); LD(47, 68, 1514); LD(48, 70, 222); LD(49, 71, 1086); LD(50, 72, 223); LD(51, 73, 1087); LD(52, 74, 224); LD(53, 75, 1088); LD(54, 77, 654); LD(55, 78, 1518); LD(56, 79, 655); LD(57, 80, 1519); LD(58, 81, 656); LD(59, 82, 1520); LD(60, 84, 228); LD(61, 85, 1092); LD(62, 86, 229); LD(63, 87, 1093); LD(64, 88, 230); LD(65, 89, 1094); LD(66, 91, 660); LD(67, 92, 1524); LD(68, 93, 661); LD(69, 94, 1525); LD(70, 95, 662); LD(71, 96, 1526); LD(72, 105, 36); LD(73, 106, 900); LD(74, 107, 37); LD(75, 108, 901); LD(76, 109, 38); LD(77, 110, 902); LD(78, 112, 468); LD(79, 113, 1332); LD(80, 114, 1334); LD(81, 116, 42); LD(82, 117, 906); LD(83, 118, 908); LD(84, 120, 474); LD(85, 121, 1338); LD(86, 122, 1340); LD(87, 124, 48); LD(88, 125, 912); LD(89, 126, 49); LD(90, 127, 913); LD(91, 128, 50); LD(92, 129, 914); LD(93, 131, 480); LD(94, 132, 1344); LD(95, 133, 481); LD(96, 134, 1345); LD(97, 135, 482); LD(98, 136, 1346); LD(99, 145, 252); LD(100, 146, 1116); LD(101, 147, 253); LD(102, 148, 1117); LD(103, 149, 254); LD(104, 150, 1118); LD(105, 152, 684); LD(106, 153, 1548); LD(107, 154, 1550); LD(108, 156, 258); LD(109, 157, 1122); LD(110, 158, 1124); LD(111, 160, 690); LD(112, 161, 1554); LD(113, 162, 1556); LD(114, 164, 264); LD(115, 165, 1128); LD(116, 166, 265); LD(117, 167, 1129); LD(118, 168, 266); LD(119, 169, 1130); LD(120, 171, 696); LD(121, 172, 1560); LD(122, 173, 697); LD(123, 174, 1561); LD(124, 175, 698); LD(125, 176, 1562); LD(126, 185, 72); LD(127, 186, 936); LD(128, 187, 73); LD(129, 188, 937); LD(130, 189, 74); LD(131, 190, 938); LD(132, 192, 504); LD(133, 193, 1368); LD(134, 194, 1370); LD(135, 196, 78); LD(136, 197, 942); LD(137, 198, 944); LD(138, 200, 510); LD(139, 201, 1374); LD(140, 202, 1376); LD(141, 204, 84); LD(142, 205, 948); LD(143, 206, 85); LD(144, 207, 949); LD(145, 208, 86); LD(146, 209, 950); LD(147, 211, 516); LD(148, 212, 1380); LD(149, 213, 517); LD(150, 214, 1381); LD(151, 215, 518); LD(152, 216, 1382); LD(153, 225, 288); LD(154, 226, 1152); LD(155, 227, 289); LD(156, 228, 1153); LD(157, 229, 290); LD(158, 230, 1154); LD(159, 232, 720); LD(160, 233, 1584); LD(161, 234, 721); LD(162, 235, 1585); LD(163, 236, 722); LD(164, 237, 1586); LD(165, 239, 294); LD(166, 240, 1158); LD(167, 241, 295); LD(168, 242, 1159); LD(169, 243, 296); LD(170, 244, 1160); LD(171, 246, 726); LD(172, 247, 1590); LD(173, 248, 727); LD(174, 249, 1591); LD(175, 250, 728); LD(176, 251, 1592); LD(177, 253, 300); LD(178, 254, 1164); LD(179, 255, 301); LD(180, 256, 1165); LD(181, 257, 302); LD(182, 258, 1166); LD(183, 260, 732); LD(184, 261, 1596); LD(185, 262, 733); LD(186, 263, 1597); LD(187, 264, 734); LD(188, 265, 1598)
#define B4_LOADXYZ6() LD(0, 49, 0); LD(1, 50, 864); LD(2, 51, 1); LD(3, 52, 865); LD(4, 53, 2); LD(5, 54, 866); LD(6, 56, 432); LD(7, 57, 1296); LD(8, 58, 433); LD(9, 59, 1297); LD(10, 60, 434); LD(11, 61, 1298); LD(12, 63, 6); LD(13, 64, 870); LD(14, 65, 7); LD(15, 66, 871); LD(16, 67, 8); LD(17, 68, 872); LD(18, 70, 438); LD(19, 71, 1302); LD(20, 72, 439); LD(21, 73, 1303); LD(22, 74, 440); LD(23, 75, 1304); LD(24, 77, 12); LD(25, 78, 876); LD(26, 79, 13); LD(27, 80, 877); LD(28, 81, 14); LD(29, 82, 878); LD(30, 84, 444); LD(31, 85, 1308); LD(32, 86, 445); LD(33, 87, 1309); LD(34, 88, 446); LD(35, 89, 1310); LD(36, 98, 216); LD(37, 99, 1080); LD(38, 100, 217); LD(39, 101, 1081); LD(40, 102, 218); LD(41, 103, 1082); LD(42, 105, 648); LD(43, 106, 1512); LD(44, 107, 649); LD(45, 108, 1513); LD(46, 109, 650); LD(47, 110, 1514); LD(48, 112, 222); LD(49, 113, 1086); LD(50, 114, 1088); LD(51, 116, 654); LD(52, 117, 1518); LD(53, 118, 1520); LD(54, 120, 228); LD(55, 121, 1092); LD(56, 122, 1094); LD(57, 124, 660); LD(58, 125, 1524); LD(59, 126, 661); LD(60, 127, 1525); LD(61, 128, 662); LD(62, 129, 1526); LD(63, 138, 36); LD(64, 139, 900); LD(65, 140, 37); LD(66, 141, 901); LD(67, 142, 38); LD(68, 143, 902); LD(69, 145, 468); LD(70, 146, 1332); LD(71, 147, 469); LD(72, 148, 1333); LD(73, 149, 470); LD(74, 150, 1334); LD(75, 152, 42); LD(76, 153, 906); LD(77, 154, 908); LD(78, 156, 474); LD(79, 157, 1338); LD(80, 158, 1340); LD(81, 160, 48); LD(82, 161, 912); LD(83, 162, 914); LD(84, 164, 480); LD(85, 165, 1344); LD(86, 166, 481); LD(87, 167, 1345); LD(88, 168, 482); LD(89, 169, 1346); LD(90, 178, 252); LD(91, 179, 1116); LD(92, 180, 253); LD(93, 181, 1117); LD(94, 182, 254); LD(95, 183, 1118); LD(96, 185, 684); LD(97, 186, 1548); LD(98, 187, 685); LD(99, 188, 1549); LD(100, 189, 686); LD(101, 190, 1550); LD(102, 192, 258); LD(103, 193, 1122); LD(104, 194, 1124); LD(105, 196, 690); LD(106, 197, 1554); LD(107, 198, 1556); LD(108, 200, 264); LD(109, 201, 1128); LD(110, 202, 1130); LD(111, 204, 696); LD(112, 205, 1560); LD(113, 206, 697); LD(114, 207, 1561); LD(115, 208, 698); LD(116, 209, 1562); LD(117, 218, 72); LD(118, 219, 936); LD(119, 220, 73); LD(120, 221, 937); LD(121, 222, 74); LD(122, 223, 938); LD(123, 225, 504); LD(124, 226, 1368); LD(125, 227, 505); LD(126, 228, 1369); LD(127, 229, 506); LD(128, 230, 1370); LD(129, 232, 78); LD(130, 233, 942); LD(131, 234, 79); LD(132, 235, 943); LD(133, 236, 80); LD(134, 237, 944); LD(135, 239, 510); LD(136, 240, 1374); LD(137, 241, 511); LD(138, 242, 1375); LD(139, 243, 512); LD(140, 244, 1376); LD(141, 246, 84); LD(142, 247, 948); LD(143, 248, 85); LD(144, 249, 949); LD(145, 250, 86); LD(146, 251, 950); LD(147, 253, 516); LD(148, 254, 1380); LD(149, 255, 517); LD(150, 256, 1381); LD(151, 257, 518); LD(152, 258, 1382); LD(153, 267, 288); LD(154, 268, 1152); LD(155, 269, 289); LD(156, 270, 1153); LD(157, 271, 290); LD(158, 272, 1154); LD(159, 274, 720); LD(160, 275, 1584); LD(161, 276, 721); LD(162, 277, 1585); LD(163, 278, 722); LD(164, 279, 1586); LD(165, 281, 294); LD(166, 282, 1158); LD(167, 283, 295); LD(168, 284, 1159); LD(169, 285, 296); LD(170, 286, 1160); LD(171, 288, 726); LD(172, 289, 1590); LD(173, 290, 727); LD(174, 291, 1591); LD(175, 292, 728); LD(176, 293, 1592); LD(177, 295, 300); LD(178, 296, 1164); LD(179, 297, 301); LD(180, 298, 1165); LD(181, 299, 302); LD(182, 300, 1166); LD(183, 302, 732); LD(184, 303, 1596); LD(185, 304, 733); LD(186, 305, 1597); LD(187, 306, 734); LD(188, 307, 1598)
#define B4_LOADXYZ7() LD(0, 0, 0); LD(1, 1, 864); LD(2, 2, 1); LD(3, 3, 865); LD(4, 4, 2); LD(5, 5, 866); LD(6, 7, 432); LD(7, 8, 1296); LD(8, 9, 433); LD(9, 10, 1297); LD(10, 11, 434); LD(11, 12, 1298); LD(12, 14, 6); LD(13, 15, 870); LD(14, 16, 7); LD(15, 17, 871); LD(16, 18, 8); LD(17, 19, 872); LD(18, 21, 438); LD(19, 22, 1302); LD(20, 23, 439); LD(21, 24, 1303); LD(22, 25, 440); LD(23, 26, 1304); LD(24, 28, 12); LD(25, 29, 876); LD(26, 30, 13); LD(27, 31, 877); LD(28, 32, 14); LD(29, 33, 878); LD(30, 35, 444); LD(31, 36, 1308); LD(32, 37, 445); LD(33, 38, 1309); LD(34, 39, 446); LD(35, 40, 1310); LD(36, 49, 216); LD(37, 50, 1080); LD(38, 51, 217); LD(39, 52, 1081); LD(40, 53, 218); LD(41, 54, 1082); LD(42, 56, 648); LD(43, 57, 1512); LD(44, 58, 649); LD(45, 59, 1513); LD(46, 60, 650); LD(47, 61, 1514); LD(48, 63, 222); LD(49, 64, 1086); LD(50, 65, 223); LD(51, 66, 1087); LD(52, 67, 224); LD(53, 68, 1088); LD(54, 70, 654); LD(55, 71, 1518); LD(56, 72, 655); LD(57, 73, 1519); LD(58, 74, 656); LD(59, 75, 1520); LD(60, 77, 228); LD(61, 78, 1092); LD(62, 79, 229); LD(63, 80, 1093); LD(64, 81, 230); LD(65, 82, 1094); LD(66, 84, 660); LD(67, 85, 1524); LD(68, 86, 661); LD(69, 87, 1525); LD(70, 88, 662); LD(71, 89, 1526); LD(72, 98, 36); LD(73, 99, 900); LD(74, 100, 37); LD(75, 101, 901); LD(76, 102, 38); LD(77, 103, 902); LD(78, 105, 468); LD(79, 106, 1332); LD(80, 107, 469); LD(81, 108, 1333); LD(82, 109, 470); LD(83, 110, 1334); LD(84, 112, 42); LD(85, 113, 906); LD(86, 114, 908); LD(87, 116, 474); LD(88, 117, 1338); LD(89, 118, 1340); LD(90, 120, 48); LD(91, 121, 912); LD(92, 122, 914); LD(93, 124, 480); LD(94, 125, 1344); LD(95, 126, 481); LD(96, 127, 1345); LD(97, 128, 482); LD(98, 129, 1346); LD(99, 138, 252); LD(100, 139, 1116); LD(101, 140, 253); LD(102, 141, 1117); LD(103, 142, 254); LD(104, 143, 1118); LD(105, 145, 684); LD(106, 146, 1548); LD(107, 147, 685); LD(108, 148, 1549); LD(109, 149, 686); LD(110, 150, 1550); LD(111, 152, 258); LD(112, 153, 1122); LD(113, 154, 1124); LD(114, 156, 690); LD(115, 157, 1554); LD(116, 158, 1556); LD(117, 160, 264); LD(118, 161, 1128); LD(119, 162, 1130); LD(120, 164, 696); LD(121, 165, 1560); LD(122, 166, 697); LD(123, 167, 1561); LD(124, 168, 698); LD(125, 169, 1562); LD(126, 178, 72); LD(127, 179, 936); LD(128, 180, 73); LD(129, 181, 937); LD(130, 182, 74); LD(131, 183, 938); LD(132, 185, 504); LD(133, 186, 1368); LD(134, 187, 505); LD(135, 188, 1369); LD(136, 189, 506); LD(137, 190, 1370); LD(138, 192, 78); LD(139, 193, 942); LD(140, 194, 944); LD(141, 196, 510); LD(142, 197, 1374); LD(143, 198, 1376); LD(144, 200, 84); LD(145, 201, 948); LD(146, 202, 950); LD(147, 204, 516); LD(148, 205, 1380); LD(149, 206, 517); LD(150, 207, 1381); LD(151, 208, 518); LD(152, 209, 1382); LD(153, 218, 288); LD(154, 219, 1152); LD(155, 220, 289); LD(156, 221, 1153); LD(157, 222, 290); LD(158, 223, 1154); LD(159, 225, 720); LD(160, 226, 1584); LD(161, 227, 721); LD(162, 228, 1585); LD(163, 229, 722); LD(164, 230, 1586); LD(165, 232, 294); LD(166, 233, 1158); LD(167, 234, 295); LD(168, 235, 1159); LD(169, 236, 296); LD(170, 237, 1160); LD(171, 239, 726); LD(172, 240, 1590); LD(173, 241, 727); LD(174, 242, 1591); LD(175, 243, 728); LD(176, 244, 1592); LD(177, 246, 300); LD(178, 247, 1164); LD(179, 248, 301); LD(180, 249, 1165); LD(181, 250, 302); LD(182, 251, 1166); LD(183, 253, 732); LD(184, 254, 1596); LD(185, 255, 733); LD(186, 256, 1597); LD(187, 257, 734); LD(188, 258, 1598)

#define B2_LOADXYZ0() LD(0, 57, 0); LD(1, 58, 256); LD(2, 59, 1); LD(3, 60, 257); LD(4, 61, 2); LD(5, 62, 258); LD(6, 64, 128); LD(7, 65, 384); LD(8, 66, 129); LD(9, 67, 385); LD(10, 68, 130); LD(11, 69, 386); LD(12, 71, 4); LD(13, 72, 260); LD(14, 73, 5); LD(15, 74, 261); LD(16, 75, 6); LD(17, 76, 262); LD(18, 78, 132); LD(19, 79, 388); LD(20, 80, 133); LD(21, 81, 389); LD(22, 82, 134); LD(23, 83, 390); LD(24, 85, 8); LD(25, 86, 264); LD(26, 87, 9); LD(27, 88, 265); LD(28, 89, 10); LD(29, 90, 266); LD(30, 92, 136); LD(31, 93, 392); LD(32, 94, 137); LD(33, 95, 393); LD(34, 96, 138); LD(35, 97, 394); LD(36, 106, 64); LD(37, 107, 320); LD(38, 108, 65); LD(39, 109, 321); LD(40, 110, 66); LD(41, 111, 322); LD(42, 113, 192); LD(43, 114, 194); LD(44, 115, 450); LD(45, 117, 68); LD(46, 118, 70); LD(47, 119, 326); LD(48, 121, 196); LD(49, 122, 198); LD(50, 123, 454); LD(51, 125, 72); LD(52, 126, 328); LD(53, 127, 73); LD(54, 128, 329); LD(55, 129, 74); LD(56, 130, 330); LD(57, 132, 200); LD(58, 133, 456); LD(59, 134, 201); LD(60, 135, 457); LD(61, 136, 202); LD(62, 137, 458); LD(63, 146, 16); LD(64, 147, 272); LD(65, 148, 17); LD(66, 149, 273); LD(67, 150, 18); LD(68, 151, 274); LD(69, 153, 144); LD(70, 154, 146); LD(71, 155, 402); LD(72, 157, 20); LD(73, 158, 22); LD(74, 159, 278); LD(75, 161, 148); LD(76, 162, 150); LD(77, 163, 406); LD(78, 165, 24); LD(79, 166, 280); LD(80, 167, 25); LD(81, 168, 281); LD(82, 169, 26); LD(83, 170, 282); LD(84, 172, 152); LD(85, 173, 408); LD(86, 174, 153); LD(87, 175, 409); LD(88, 176, 154); LD(89, 177, 410); LD(90, 186, 80); LD(91, 187, 336); LD(92, 188, 81); LD(93, 189, 337); LD(94, 190, 82); LD(95, 191, 338); LD(96, 193, 208); LD(97, 194, 210); LD(98, 195, 466); LD(99, 197, 84); LD(100, 198, 86); LD(101, 199, 342); LD(102, 201, 212); LD(103, 202, 214); LD(104, 203, 470); LD(105, 205, 88); LD(106, 206, 344); LD(107, 207, 89); LD(108, 208, 345); LD(109, 209, 90); LD(110, 210, 346); LD(111, 212, 216); LD(112, 213, 472); LD(113, 214, 217); LD(114, 215, 473); LD(115, 216, 218); LD(116, 217, 474); LD(117, 226, 32); LD(118, 227, 288); LD(119, 228, 33); LD(120, 229, 289); LD(121, 230, 34); LD(122, 231, 290); LD(123, 233, 160); LD(124, 234, 416); LD(125, 235, 161); LD(126, 236, 417); LD(127, 237, 162); LD(128, 238, 418); LD(129, 240, 36); LD(130, 241, 292); LD(131, 242, 37); LD(132, 243, 293); LD(133, 244, 38); LD(134, 245, 294); LD(135, 247, 164); LD(136, 248, 420); LD(137, 249, 165); LD(138, 250, 421); LD(139, 251, 166); LD(140, 252, 422); LD(141, 254, 40); LD(142, 255, 296); LD(143, 256, 41); LD(144, 257, 297); LD(145, 258, 42); LD(146, 259, 298); LD(147, 261, 168); LD(148, 262, 424); LD(149, 263, 169); LD(150, 264, 425); LD(151, 265, 170); LD(152, 266, 426); LD(153, 275, 96); LD(154, 276, 352); LD(155, 277, 97); LD(156, 278, 353); LD(157, 279, 98); LD(158, 280, 354); LD(159, 282, 224); LD(160, 283, 480); LD(161, 284, 225); LD(162, 285, 481); LD(163, 286, 226); LD(164, 287, 482); LD(165, 289, 100); LD(166, 290, 356); LD(167, 291, 101); LD(168, 292, 357); LD(169, 293, 102); LD(170, 294, 358); LD(171, 296, 228); LD(172, 297, 484); LD(173, 298, 229); LD(174, 299, 485); LD(175, 300, 230); LD(176, 301, 486); LD(177, 303, 104); LD(178, 304, 360); LD(179, 305, 105); LD(180, 306, 361); LD(181, 307, 106); LD(182, 308, 362); LD(183, 310, 232); LD(184, 311, 488); LD(185, 312, 233); LD(186, 313, 489); LD(187, 314, 234); LD(188, 315, 490)
#define B2_LOADXYZ1() LD(0, 8, 0); LD(1, 9, 256); LD(2, 10, 1); LD(3, 11, 257); LD(4, 12, 2); LD(5, 13, 258); LD(6, 15, 128); LD(7, 16, 384); LD(8, 17, 129); LD(9, 18, 385); LD(10, 19, 130); LD(11, 20, 386); LD(12, 22, 4); LD(13, 23, 260); LD(14, 24, 5); LD(15, 25, 261); LD(16, 26, 6); LD(17, 27, 262); LD(18, 29, 132); LD(19, 30, 388); LD(20, 31, 133); LD(21, 32, 389); LD(22, 33, 134); LD(23, 34, 390); LD(24, 36, 8); LD(25, 37, 264); LD(26, 38, 9); LD(27, 39, 265); LD(28, 40, 10); LD(29, 41, 266); LD(30, 43, 136); LD(31, 44, 392); LD(32, 45, 137); LD(33, 46, 393); LD(34, 47, 138); LD(35, 48, 394); LD(36, 57, 64); LD(37, 58, 320); LD(38, 59, 65); LD(39, 60, 321); LD(40, 61, 66); LD(41, 62, 322); LD(42, 64, 192); LD(43, 65, 448); LD(44, 66, 193); LD(45, 67, 449); LD(46, 68, 194); LD(47, 69, 450); LD(48, 71, 68); LD(49, 72, 324); LD(50, 73, 69); LD(51, 74, 325); LD(52, 75, 70); LD(53, 76, 326); LD(54, 78, 196); LD(55, 79, 452); LD(56, 80, 197); LD(57, 81, 453); LD(58, 82, 198); LD(59, 83, 454); LD(60, 85, 72); LD(61, 86, 328); LD(62, 87, 73); LD(63, 88, 329); LD(64, 89, 74); LD(65, 90, 330); LD(66, 92, 200); LD(67, 93, 456); LD(68, 94, 201); LD(69, 95, 457); LD(70, 96, 202); LD(71, 97, 458); LD(72, 106, 16); LD(73, 107, 272); LD(74, 108, 17); LD(75, 109, 273); LD(76, 110, 18); LD(77, 111, 274); LD(78, 113, 144); LD(79, 114, 146); LD(80, 115, 402); LD(81, 117, 20); LD(82, 118, 22); LD(83, 119, 278); LD(84, 121, 148); LD(85, 122, 150); LD(86, 123, 406); LD(87, 125, 24); LD(88, 126, 280); LD(89, 127, 25); LD(90, 128, 281); LD(91, 129, 26); LD(92, 130, 282); LD(93, 132, 152); LD(94, 133, 408); LD(95, 134, 153); LD(96, 135, 409); LD(97, 136, 154); LD(98, 137, 410); LD(99, 146, 80); LD(100, 147, 336); LD(101, 148, 81); LD(102, 149, 337); LD(103, 150, 82); LD(104, 151, 338); LD(105, 153, 208); LD(106, 154, 210); LD(107, 155, 466); LD(108, 157, 84); LD(109, 158, 86); LD(110, 159, 342); LD(111, 161, 212); LD(112, 162, 214); LD(113, 163, 470); LD(114, 165, 88); LD(115, 166, 344); LD(116, 167, 89); LD(117, 168, 345); LD(118, 169, 90); LD(119, 170, 346); LD(120, 172, 216); LD(121, 173, 472); LD(122, 174, 217); LD(123, 175, 473); LD(124, 176, 218); LD(125, 177, 474); LD(126, 186, 32); LD(127, 187, 288); LD(128, 188, 33); LD(129, 189, 289); LD(130, 190, 34); LD(131, 191, 290); LD(132, 193, 160); LD(133, 194, 162); LD(134, 195, 418); LD(135, 197, 36); LD(136, 198, 38); LD(137, 199, 294); LD(138, 201, 164); LD(139, 202, 166); LD(140, 203, 422); LD(141, 205, 40); LD(142, 206, 296); LD(143, 207, 41); LD(144, 208, 297); LD(145, 209, 42); LD(146, 210, 298); LD(147, 212, 168); LD(148, 213, 424); LD(149, 214, 169); LD(150, 215, 425); LD(151, 216, 170); LD(152, 217, 426); LD(153, 226, 96); LD(154, 227, 352); LD(155, 228, 97); LD(156, 229, 353); LD(157, 230, 98); LD(158, 231, 354); LD(159, 233, 224); LD(160, 234, 480); LD(161, 235, 225); LD(162, 236, 481); LD(163, 237, 226); LD(164, 238, 482); LD(165, 240, 100); LD(166, 241, 356); LD(167, 242, 101); LD(168, 243, 357); LD(169, 244, 102); LD(170, 245, 358); LD(171, 247, 228); LD(172, 248, 484); LD(173, 249, 229); LD(174, 250, 485); LD(175, 251, 230); LD(176, 252, 486); LD(177, 254, 104); LD(178, 255, 360); LD(179, 256, 105); LD(180, 257, 361); LD(181, 258, 106); LD(182, 259, 362); LD(183, 261, 232); LD(184, 262, 488); LD(185, 263, 233); LD(186, 264, 489); LD(187, 265, 234); LD(188, 266, 490)
#define B2_LOADXYZ2() LD(0, 50, 0); LD(1, 51, 256); LD(2, 52, 1); LD(3, 53, 257); LD(4, 54, 2); LD(5, 55, 258); LD(6, 57, 128); LD(7, 58, 384); LD(8, 59, 129); LD(9, 60, 385); LD(10, 61, 130); LD(11, 62, 386); LD(12, 64, 4); LD(13, 65, 260); LD(14, 66, 5); LD(15, 67, 261); LD(16, 68, 6); LD(17, 69, 262); LD(18, 71, 132); LD(19, 72, 388); LD(20, 73, 133); LD(21, 74, 389); LD(22, 75, 134); LD(23, 76, 390); LD(24, 78, 8); LD(25, 79, 264); LD(26, 80, 9); LD(27, 81, 265); LD(28, 82, 10); LD(29, 83, 266); LD(30, 85, 136); LD(31, 86, 392); LD(32, 87, 137); LD(33, 88, 393); LD(34, 89, 138); LD(35, 90, 394); LD(36, 99, 64); LD(37, 100, 320); LD(38, 101, 65); LD(39, 102, 321); LD(40, 103, 66); LD(41, 104, 322); LD(42, 106, 192); LD(43, 107, 448); LD(44, 108, 193); LD(45, 109, 449); LD(46, 110, 194); LD(47, 111, 450); LD(48, 113, 68); LD(49, 114, 70); LD(50, 115, 326); LD(51, 117, 196); LD(52, 118, 198); LD(53, 119, 454); LD(54, 121, 72); LD(55, 122, 74); LD(56, 123, 330); LD(57, 125, 200); LD(58, 126, 456); LD(59, 127, 201); LD(60, 128, 457); LD(61, 129, 202); LD(62, 130, 458); LD(63, 139, 16); LD(64, 140, 272); LD(65, 141, 17); LD(66, 142, 273); LD(67, 143, 18); LD(68, 144, 274); LD(69, 146, 144); LD(70, 147, 400); LD(71, 148, 145); LD(72, 149, 401); LD(73, 150, 146); LD(74, 151, 402); LD(75, 153, 20); LD(76, 154, 22); LD(77, 155, 278); LD(78, 157, 148); LD(79, 158, 150); LD(80, 159, 406); LD(81, 161, 24); LD(82, 162, 26); LD(83, 163, 282); LD(84, 165, 152); LD(85, 166, 408); LD(86, 167, 153); LD(87, 168, 409); LD(88, 169, 154); LD(89, 170, 410); LD(90, 179, 80); LD(91, 180, 336); LD(92, 181, 81); LD(93, 182, 337); LD(94, 183, 82); LD(95, 184, 338); LD(96, 186, 208); LD(97, 187, 464); LD(98, 188, 209); LD(99, 189, 465); LD(100, 190, 210); LD(101, 191, 466); LD(102, 193, 84); LD(103, 194, 86); LD(104, 195, 342); LD(105, 197, 212); LD(106, 198, 214); LD(107, 199, 470); LD(108, 201, 88); LD(109, 202, 90); LD(110, 203, 346); LD(111, 205, 216); LD(112, 206, 472); LD(113, 207, 217); LD(114, 208, 473); LD(115, 209, 218); LD(116, 210, 474); LD(117, 219, 32); LD(118, 220, 288); LD(119, 221, 33); LD(120, 222, 289); LD(121, 223, 34); LD(122, 224, 290); LD(123, 226, 160); LD(124, 227, 416); LD(125, 228, 161); LD(126, 229, 417); LD(127, 230, 162); LD(128, 231, 418); LD(129, 233, 36); LD(130, 234, 292); LD(131, 235, 37); LD(132, 236, 293); LD(133, 237, 38); LD(134, 238, 294); LD(135, 240, 164); LD(136, 241, 420); LD(137, 242, 165); LD(138, 243, 421); LD(139, 244, 166); LD(140, 245, 422); LD(141, 247, 40); LD(142, 248, 296); LD(143, 249, 41); LD(144, 250, 297); LD(145, 251, 42); LD(146, 252, 298); LD(147, 254, 168); LD(148, 255, 424); LD(149, 256, 169); LD(150, 257, 425); LD(151, 258, 170); LD(152, 259, 426); LD(153, 268, 96); LD(154, 269, 352); LD(155, 270, 97); LD(156, 271, 353); LD(157, 272, 98); LD(158, 273, 354); LD(159, 275, 224); LD(160, 276, 480); LD(161, 277, 225); LD(162, 278, 481); LD(163, 279, 226); LD(164, 280, 482); LD(165, 282, 100); LD(166, 283, 356); LD(167, 284, 101); LD(168, 285, 357); LD(169, 286, 102); LD(170, 287, 358); LD(171, 289, 228); LD(172, 290, 484); LD(173, 291, 229); LD(174, 292, 485); LD(175, 293, 230); LD(176, 294, 486); LD(177, 296, 104); LD(178, 297, 360); LD(179, 298, 105); LD(180, 299, 361); LD(181, 300, 106); LD(182, 301, 362); LD(183, 303, 232); LD(184, 304, 488); LD(185, 305, 233); LD(186, 306, 489); LD(187, 307, 234); LD(188, 308, 490)
#define B2_LOADXYZ3() LD(0, 1, 0); LD(1, 2, 256); LD(2, 3, 1); LD(3, 4, 257); LD(4, 5, 2); LD(5, 6, 258); LD(6, 8, 128); LD(7, 9, 384); LD(8, 10, 129); LD(9, 11, 385); LD(10, 12, 130); LD(11, 13, 386); LD(12, 15, 4); LD(13, 16, 260); LD(14, 17, 5); LD(15, 18, 261); LD(16, 19, 6); LD(17, 20, 262); LD(18, 22, 132); LD(19, 23, 388); LD(20, 24, 133); LD(21, 25, 389); LD(22, 26, 134); LD(23, 27, 390); LD(24, 29, 8); LD(25, 30, 264); LD(26, 31, 9); LD(27, 32, 265); LD(28, 33, 10); LD(29, 34, 266); LD(30, 36, 136); LD(31, 37, 392); LD(32, 38, 137); LD(33, 39, 393); LD(34, 40, 138); LD(35, 41, 394); LD(36, 50, 64); LD(37, 51, 320); LD(38, 52, 65); LD(39, 53, 321); LD(40, 54, 66); LD(41, 55, 322); LD(42, 57, 192); LD(43, 58, 448); LD(44, 59, 193); LD(45, 60, 449); LD(46, 61, 194); LD(47, 62, 450); LD(48, 64, 68); LD(49, 65, 324); LD(50, 66, 69); LD(51, 67, 325); LD(52, 68, 70); LD(53, 69, 326); LD(54, 71, 196); LD(55, 72, 452); LD(56, 73, 197); LD(57, 74, 453); LD(58, 75, 198); LD(59, 76, 454); LD(60, 78, 72); LD(61, 79, 328); LD(62, 80, 73); LD(63, 81, 329); LD(64, 82, 74); LD(65, 83, 330); LD(66, 85, 200); LD(67, 86, 456); LD(68, 87, 201); LD(69, 88, 457); LD(70, 89, 202); LD(71, 90, 458); LD(72, 99, 16); LD(73, 100, 272); LD(74, 101, 17); LD(75, 102, 273); LD(76, 103, 18); LD(77, 104, 274); LD(78, 106, 144); LD(79, 107, 400); LD(80, 108, 145); LD(81, 109, 401); LD(82, 110, 146); LD(83, 111, 402); LD(84, 113, 20); LD(85, 114, 22); LD(86, 115, 278); LD(87, 117, 148); LD(88, 118, 150); LD(89, 119, 406); LD(90, 121, 24); LD(91, 122, 26); LD(92, 123, 282); LD(93, 125, 152); LD(94, 126, 408); LD(95, 127, 153); LD(96, 128, 409); LD(97, 129, 154); LD(98, 130, 410); LD(99, 139, 80); LD(100, 140, 336); LD(101, 141, 81); LD(102, 142, 337); LD(103, 143, 82); LD(104, 144, 338); LD(105, 146, 208); LD(106, 147, 464); LD(107, 148, 209); LD(108, 149, 465); LD(109, 150, 210); LD(110, 151, 466); LD(111, 153, 84); LD(112, 154, 86); LD(113, 155, 342); LD(114, 157, 212); LD(115, 158, 214); LD(116, 159, 470); LD(117, 161, 88); LD(118, 162, 90); LD(119, 163, 346); LD(120, 165, 216); LD(121, 166, 472); LD(122, 167, 217); LD(123, 168, 473); LD(124, 169, 218); LD(125, 170, 474); LD(126, 179, 32); LD(127, 180, 288); LD(128, 181, 33); LD(129, 182, 289); LD(130, 183, 34); LD(131, 184, 290); LD(132, 186, 160); LD(133, 187, 416); LD(134, 188, 161); LD(135, 189, 417); LD(136, 190, 162); LD(137, 191, 418); LD(138, 193, 36); LD(139, 194, 38); LD(140, 195, 294); LD(141, 197, 164); LD(142, 198, 166); LD(143, 199, 422); LD(144, 201, 40); LD(145, 202, 42); LD(146, 203, 298); LD(147, 205, 168); LD(148, 206, 424); LD(149, 207, 169); LD(150, 208, 425); LD(151, 209, 170); LD(152, 210, 426); LD(153, 219, 96); LD(154, 220, 352); LD(155, 221, 97); LD(156, 222, 353); LD(157, 223, 98); LD(158, 224, 354); LD(159, 226, 224); LD(160, 227, 480); LD(161, 228, 225); LD(162, 229, 481); LD(163, 230, 226); LD(164, 231, 482); LD(165, 233, 100); LD(166, 234, 356); LD(167, 235, 101); LD(168, 236, 357); LD(169, 237, 102); LD(170, 238, 358); LD(171, 240, 228); LD(172, 241, 484); LD(173, 242, 229); LD(174, 243, 485); LD(175, 244, 230); LD(176, 245, 486); LD(177, 247, 104); LD(178, 248, 360); LD(179, 249, 105); LD(180, 250, 361); LD(181, 251, 106); LD(182, 252, 362); LD(183, 254, 232); LD(184, 255, 488); LD(185, 256, 233); LD(186, 257, 489); LD(187, 258, 234); LD(188, 259, 490)
#define B2_LOADXYZ4() LD(0, 56, 0); LD(1, 57, 256); LD(2, 58, 1); LD(3, 59, 257); LD(4, 60, 2); LD(5, 61, 258); LD(6, 63, 128); LD(7, 64, 384); LD(8, 65, 129); LD(9, 66, 385); LD(10, 67, 130); LD(11, 68, 386); LD(12, 70, 4); LD(13, 71, 260); LD(14, 72, 5); LD(15, 73, 261); LD(16, 74, 6); LD(17, 75, 262); LD(18, 77, 132); LD(19, 78, 388); LD(20, 79, 133); LD(21, 80, 389); LD(22, 81, 134); LD(23, 82, 390); LD(24, 84, 8); LD(25, 85, 264); LD(26, 86, 9); LD(27, 87, 265); LD(28, 88, 10); LD(29, 89, 266); LD(30, 91, 136); LD(31, 92, 392); LD(32, 93, 137); LD(33, 94, 393); LD(34, 95, 138); LD(35, 96, 394); LD(36, 105, 64); LD(37, 106, 320); LD(38, 107, 65); LD(39, 108, 321); LD(40, 109, 66); LD(41, 110, 322); LD(42, 112, 192); LD(43, 113, 448); LD(44, 114, 450); LD(45, 116, 68); LD(46, 117, 324); LD(47, 118, 326); LD(48, 120, 196); LD(49, 121, 452); LD(50, 122, 454); LD(51, 124, 72); LD(52, 125, 328); LD(53, 126, 73); LD(54, 127, 329); LD(55, 128, 74); LD(56, 129, 330); LD(57, 131, 200); LD(58, 132, 456); LD(59, 133, 201); LD(60, 134, 457); LD(61, 135, 202); LD(62, 136, 458); LD(63, 145, 16); LD(64, 146, 272); LD(65, 147, 17); LD(66, 148, 273); LD(67, 149, 18); LD(68, 150, 274); LD(69, 152, 144); LD(70, 153, 400); LD(71, 154, 402); LD(72, 156, 20); LD(73, 157, 276); LD(74, 158, 278); LD(75, 160, 148); LD(76, 161, 404); LD(77, 162, 406); LD(78, 164, 24); LD(79, 165, 280); LD(80, 166, 25); LD(81, 167, 281); LD(82, 168, 26); LD(83, 169, 282); LD(84, 171, 152); LD(85, 172, 408); LD(86, 173, 153); LD(87, 174, 409); LD(88, 175, 154); LD(89, 176, 410); LD(90, 185, 80); LD(91, 186, 336); LD(92, 187, 81); LD(93, 188, 337); LD(94, 189, 82); LD(95, 190, 338); LD(96, 192, 208); LD(97, 193, 464); LD(98, 194, 466); LD(99, 196, 84); LD(100, 197, 340); LD(101, 198, 342); LD(102, 200, 212); LD(103, 201, 468); LD(104, 202, 470); LD(105, 204, 88); LD(106, 205, 344); LD(107, 206, 89); LD(108, 207, 345); LD(109, 208, 90); LD(110, 209, 346); LD(111, 211, 216); LD(112, 212, 472); LD(113, 213, 217); LD(114, 214, 473); LD(115, 215, 218); LD(116, 216, 474); LD(117, 225, 32); LD(118, 226, 288); LD(119, 227, 33); LD(120, 228, 289); LD(121, 229, 34); LD(122, 230, 290); LD(123, 232, 160); LD(124, 233, 416); LD(125, 234, 161); LD(126, 235, 417); LD(127, 236, 162); LD(128, 237, 418); LD(129, 239, 36); LD(130, 240, 292); LD(131, 241, 37); LD(132, 242, 293); LD(133, 243, 38); LD(134, 244, 294); LD(135, 246, 164); LD(136, 247, 420); LD(137, 248, 165); LD(138, 249, 421); LD(139, 250, 166); LD(140, 251, 422); LD(141, 253, 40); LD(142, 254, 296); LD(143, 255, 41); LD(144, 256, 297); LD(145, 257, 42); LD(146, 258, 298); LD(147, 260, 168); LD(148, 261, 424); LD(149, 262, 169); LD(150, 263, 425); LD(151, 264, 170); LD(152, 265, 426); LD(153, 274, 96); LD(154, 275, 352); LD(155, 276, 97); LD(156, 277, 353); LD(157, 278, 98); LD(158, 279, 354); LD(159, 281, 224); LD(160, 282, 480); LD(161, 283, 225); LD(162, 284, 481); LD(163, 285, 226); LD(164, 286, 482); LD(165, 288, 100); LD(166, 289, 356); LD(167, 290, 101); LD(168, 291, 357); LD(169, 292, 102); LD(170, 293, 358); LD(171, 295, 228); LD(172, 296, 484); LD(173, 297, 229); LD(174, 298, 485); LD(175, 299, 230); LD(176, 300, 486); LD(177, 302, 104); LD(178, 303, 360); LD(179, 304, 105); LD(180, 305, 361); LD(181, 306, 106); LD(182, 307, 362); LD(183, 309, 232); LD(184, 310, 488); LD(185, 311, 233); LD(186, 312, 489); LD(187, 313, 234); LD(188, 314, 490)
#define B2_LOADXYZ5() LD(0, 7, 0); LD(1, 8, 256); LD(2, 9, 1); LD(3, 10, 257); LD(4, 11, 2); LD(5, 12, 258); LD(6, 14, 128); LD(7, 15, 384); LD(8, 16, 129); LD(9, 17, 385); LD(10, 18, 130); LD(11, 19, 386); LD(12, 21, 4); LD(13, 22, 260); LD(14, 23, 5); LD(15, 24, 261); LD(16, 25, 6); LD(17, 26, 262); LD(18, 28, 132); LD(19, 29, 388); LD(20, 30, 133); LD(21, 31, 389); LD(22, 32, 134); LD(23, 33, 390); LD(24, 35, 8); LD(25, 36, 264); LD(26, 37, 9); LD(27, 38, 265); LD(28, 39, 10); LD(29, 40, 266); LD(30, 42, 136); LD(31, 43, 392); LD(32, 44, 137); LD(33, 45, 393); LD(34, 46, 138); LD(35, 47, 394); LD(36, 56, 64); LD(37, 57, 320); LD(38, 58, 65); LD(39, 59, 321); LD(40, 60, 66); LD(41, 61, 322); LD(42, 63, 192); LD(43, 64, 448); LD(44, 65, 193); LD(45, 66, 449); LD(46, 67, 194); LD(47, 68, 450); LD(48, 70, 68); LD(49, 71, 324); LD(50, 72, 69); LD(51, 73, 325); LD(52, 74, 70); LD(53, 75, 326); LD(54, 77, 196); LD(55, 78, 452); LD(56, 79, 197); LD(57, 80, 453); LD(58, 81, 198); LD(59, 82, 454); LD(60, 84, 72); LD(61, 85, 328); LD(62, 86, 73); LD(63, 87, 329); LD(64, 88, 74); LD(65, 89, 330); LD(66, 91, 200); LD(67, 92, 456); LD(68, 93, 201); LD(69, 94, 457); LD(70, 95, 202); LD(71, 96, 458); LD(72, 105, 16); LD(73, 106, 272); LD(74, 107, 17); LD(75, 108, 273); LD(76, 109, 18); LD(77, 110, 274); LD(78, 112, 144); LD(79, 113, 400); LD(80, 114, 402); LD(81, 116, 20); LD(82, 117, 276); LD(83, 118, 278); LD(84, 120, 148); LD(85, 121, 404); LD(86, 122, 406); LD(87, 124, 24); LD(88, 125, 280); LD(89, 126, 25); LD(90, 127, 281); LD(91, 128, 26); LD(92, 129, 282); LD(93, 131, 152); LD(94, 132, 408); LD(95, 133, 153); LD(96, 134, 409); LD(97, 135, 154); LD(98, 136, 410); LD(99, 145, 80); LD(100, 146, 336); LD(101, 147, 81); LD(102, 148, 337); LD(103, 149, 82); LD(104, 150, 338); LD(105, 152, 208); LD(106, 153, 464); LD(107, 154, 466); LD(108, 156, 84); LD(109, 157, 340); LD(110, 158, 342); LD(111, 160, 212); LD(112, 161, 468); LD(113, 162, 470); LD(114, 164, 88); LD(115, 165, 344); LD(116, 166, 89); LD(117, 167, 345); LD(118, 168, 90); LD(119, 169, 346); LD(120, 171, 216); LD(121, 172, 472); LD(122, 173, 217); LD(123, 174, 473); LD(124, 175, 218); LD(125, 176, 474); LD(126, 185, 32); LD(127, 186, 288); LD(128, 187, 33); LD(129, 188, 289); LD(130, 189, 34); LD(131, 190, 290); LD(132, 192, 160); LD(133, 193, 416); LD(134, 194, 418); LD(135, 196, 36); LD(136, 197, 292); LD(137, 198, 294); LD(138, 200, 164); LD(139, 201, 420); LD(140, 202, 422); LD(141, 204, 40); LD(142, 205, 296); LD(143, 206, 41); LD(144, 207, 297); LD(145, 208, 42); LD(146, 209, 298); LD(147, 211, 168); LD(148, 212, 424); LD(149, 213, 169); LD(150, 214, 425); LD(151, 215, 170); LD(152, 216, 426); LD(153, 225, 96); LD(154, 226, 352); LD(155, 227, 97); LD(156, 228, 353); LD(157, 229, 98); LD(158, 230, 354); LD(159, 232, 224); LD(160, 233, 480); LD(161, 234, 225); LD(162, 235, 481); LD(163, 236, 226); LD(164, 237, 482); LD(165, 239, 100); LD(166, 240, 356); LD(167, 241, 101); LD(168, 242, 357); LD(169, 243, 102); LD(170, 244, 358); LD(171, 246, 228); LD(172, 247, 484); LD(173, 248, 229); LD(174, 249, 485); LD(175, 250, 230); LD(176, 251, 486); LD(177, 253, 104); LD(178, 254, 360); LD(179, 255, 105); LD(180, 256, 361); LD(181, 257, 106); LD(182, 258, 362); LD(183, 260, 232); LD(184, 261, 488); LD(185, 262, 233); LD(186, 263, 489); LD(187, 264, 234); LD(188, 265, 490)
#define B2_LOADXYZ6() LD(0, 49, 0); LD(1, 50, 256); LD(2, 51, 1); LD(3, 52, 257); LD(4, 53, 2); LD(5, 54, 258); LD(6, 56, 128); LD(7, 57, 384); LD(8, 58, 129); LD(9, 59, 385); LD(10, 60, 130); LD(11, 61, 386); LD(12, 63, 4); LD(13, 64, 260); LD(14, 65, 5); LD(15, 66, 261); LD(16, 67, 6); LD(17, 68, 262); LD(18, 70, 132); LD(19, 71, 388); LD(20, 72, 133); LD(21, 73, 389); LD(22, 74, 134); LD(23, 75, 390); LD(24, 77, 8); LD(25, 78, 264); LD(26, 79, 9); LD(27, 80, 265); LD(28, 81, 10); LD(29, 82, 266); LD(30, 84, 136); LD(31, 85, 392); LD(32, 86, 137); LD(33, 87, 393); LD(34, 88, 138); LD(35, 89, 394); LD(36, 98, 64); LD(37, 99, 320); LD(38, 100, 65); LD(39, 101, 321); LD(40, 102, 66); LD(41, 103, 322); LD(42, 105, 192); LD(43, 106, 448); LD(44, 107, 193); LD(45, 108, 449); LD(46, 109, 194); LD(47, 110, 450); LD(48, 112, 68); LD(49, 113, 324); LD(50, 114, 326); LD(51, 116, 196); LD(52, 117, 452); LD(53, 118, 454); LD(54, 120, 72); LD(55, 121, 328); LD(56, 122, 330); LD(57, 124, 200); LD(58, 125, 456); LD(59, 126, 201); LD(60, 127, 457); LD(61, 128, 202); LD(62, 129, 458); LD(63, 138, 16); LD(64, 139, 272); LD(65, 140, 17); LD(66, 141, 273); LD(67, 142, 18); LD(68, 143, 274); LD(69, 145, 144); LD(70, 146, 400); LD(71, 147, 145); LD(72, 148, 401); LD(73, 149, 146); LD(74, 150, 402); LD(75, 152, 20); LD(76, 153, 276); LD(77, 154, 278); LD(78, 156, 148); LD(79, 157, 404); LD(80, 158, 406); LD(81, 160, 24); LD(82, 161, 280); LD(83, 162, 282); LD(84, 164, 152); LD(85, 165, 408); LD(86, 166, 153); LD(87, 167, 409); LD(88, 168, 154); LD(89, 169, 410); LD(90, 178, 80); LD(91, 179, 336); LD(92, 180, 81); LD(93, 181, 337); LD(94, 182, 82); LD(95, 183, 338); LD(96, 185, 208); LD(97, 186, 464); LD(98, 187, 209); LD(99, 188, 465); LD(100, 189, 210); LD(101, 190, 466); LD(102, 192, 84); LD(103, 193, 340); LD(104, 194, 342); LD(105, 196, 212); LD(106, 197, 468); LD(107, 198, 470); LD(108, 200, 88); LD(109, 201, 344); LD(110, 202, 346); LD(111, 204, 216); LD(112, 205, 472); LD(113, 206, 217); LD(114, 207, 473); LD(115, 208, 218); LD(116, 209, 474); LD(117, 218, 32); LD(118, 219, 288); LD(119, 220, 33); LD(120, 221, 289); LD(121, 222, 34); LD(122, 223, 290); LD(123, 225, 160); LD(124, 226, 416); LD(125, 227, 161); LD(126, 228, 417); LD(127, 229, 162); LD(128, 230, 418); LD(129, 232, 36); LD(130, 233, 292); LD(131, 234, 37); LD(132, 235, 293); LD(133, 236, 38); LD(134, 237, 294); LD(135, 239, 164); LD(136, 240, 420); LD(137, 241, 165); LD(138, 242, 421); LD(139, 243, 166); LD(140, 244, 422); LD(141, 246, 40); LD(142, 247, 296); LD(143, 248, 41); LD(144, 249, 297); LD(145, 250, 42); LD(146, 251, 298); LD(147, 253, 168); LD(148, 254, 424); LD(149, 255, 169); LD(150, 256, 425); LD(151, 257, 170); LD(152, 258, 426); LD(153, 267, 96); LD(154, 268, 352); LD(155, 269, 97); LD(156, 270, 353); LD(157, 271, 98); LD(158, 272, 354); LD(159, 274, 224); LD(160, 275, 480); LD(161, 276, 225); LD(162, 277, 481); LD(163, 278, 226); LD(164, 279, 482); LD(165, 281, 100); LD(166, 282, 356); LD(167, 283, 101); LD(168, 284, 357); LD(169, 285, 102); LD(170, 286, 358); LD(171, 288, 228); LD(172, 289, 484); LD(173, 290, 229); LD(174, 291, 485); LD(175, 292, 230); LD(176, 293, 486); LD(177, 295, 104); LD(178, 296, 360); LD(179, 297, 105); LD(180, 298, 361); LD(181, 299, 106); LD(182, 300, 362); LD(183, 302, 232); LD(184, 303, 488); LD(185, 304, 233); LD(186, 305, 489); LD(187, 306, 234); LD(188, 307, 490)
#define B2_LOADXYZ7() LD(0, 0, 0); LD(1, 1, 256); LD(2, 2, 1); LD(3, 3, 257); LD(4, 4, 2); LD(5, 5, 258); LD(6, 7, 128); LD(7, 8, 384); LD(8, 9, 129); LD(9, 10, 385); LD(10, 11, 130); LD(11, 12, 386); LD(12, 14, 4); LD(13, 15, 260); LD(14, 16, 5); LD(15, 17, 261); LD(16, 18, 6); LD(17, 19, 262); LD(18, 21, 132); LD(19, 22, 388); LD(20, 23, 133); LD(21, 24, 389); LD(22, 25, 134); LD(23, 26, 390); LD(24, 28, 8); LD(25, 29, 264); LD(26, 30, 9); LD(27, 31, 265); LD(28, 32, 10); LD(29, 33, 266); LD(30, 35, 136); LD(31, 36, 392); LD(32, 37, 137); LD(33, 38, 393); LD(34, 39, 138); LD(35, 40, 394); LD(36, 49, 64); LD(37, 50, 320); LD(38, 51, 65); LD(39, 52, 321); LD(40, 53, 66); LD(41, 54, 322); LD(42, 56, 192); LD(43, 57, 448); LD(44, 58, 193); LD(45, 59, 449); LD(46, 60, 194); LD(47, 61, 450); LD(48, 63, 68); LD(49, 64, 324); LD(50, 65, 69); LD(51, 66, 325); LD(52, 67, 70); LD(53, 68, 326); LD(54, 70, 196); LD(55, 71, 452); LD(56, 72, 197); LD(57, 73, 453); LD(58, 74, 198); LD(59, 75, 454); LD(60, 77, 72); LD(61, 78, 328); LD(62, 79, 73); LD(63, 80, 329); LD(64, 81, 74); LD(65, 82, 330); LD(66, 84, 200); LD(67, 85, 456); LD(68, 86, 201); LD(69, 87, 457); LD(70, 88, 202); LD(71, 89, 458); LD(72, 98, 16); LD(73, 99, 272); LD(74, 100, 17); LD(75, 101, 273); LD(76, 102, 18); LD(77, 103, 274); LD(78, 105, 144); LD(79, 106, 400); LD(80, 107, 145); LD(81, 108, 401); LD(82, 109, 146); LD(83, 110, 402); LD(84, 112, 20); LD(85, 113, 276); LD(86, 114, 278); LD(87, 116, 148); LD(88, 117, 404); LD(89, 118, 406); LD(90, 120, 24); LD(91, 121, 280); LD(92, 122, 282); LD(93, 124, 152); LD(94, 125, 408); LD(95, 126, 153); LD(96, 127, 409); LD(97, 128, 154); LD(98, 129, 410); LD(99, 138, 80); LD(100, 139, 336); LD(101, 140, 81); LD(102, 141, 337); LD(103, 142, 82); LD(104, 143, 338); LD(105, 145, 208); LD(106, 146, 464); LD(107, 147, 209); LD(108, 148, 465); LD(109, 149, 210); LD(110, 150, 466); LD(111, 152, 84); LD(112, 153, 340); LD(113, 154, 342); LD(114, 156, 212); LD(115, 157, 468); LD(116, 158, 470); LD(117, 160, 88); LD(118, 161, 344); LD(119, 162, 346); LD(120, 164, 216); LD(121, 165, 472); LD(122, 166, 217); LD(123, 167, 473); LD(124, 168, 218); LD(125, 169, 474); LD(126, 178, 32); LD(127, 179, 288); LD(128, 180, 33); LD(129, 181, 289); LD(130, 182, 34); LD(131, 183, 290); LD(132, 185, 160); LD(133, 186, 416); LD(134, 187, 161); LD(135, 188, 417); LD(136, 189, 162); LD(137, 190, 418); LD(138, 192, 36); LD(139, 193, 292); LD(140, 194, 294); LD(141, 196, 164); LD(142, 197, 420); LD(143, 198, 422); LD(144, 200, 40); LD(145, 201, 296); LD(146, 202, 298); LD(147, 204, 168); LD(148, 205, 424); LD(149, 206, 169); LD(150, 207, 425); LD(151, 208, 170); LD(152, 209, 426); LD(153, 218, 96); LD(154, 219, 352); LD(155, 220, 97); LD(156, 221, 353); LD(157, 222, 98); LD(158, 223, 354); LD(159, 225, 224); LD(160, 226, 480); LD(161, 227, 225); LD(162, 228, 481); LD(163, 229, 226); LD(164, 230, 482); LD(165, 232, 100); LD(166, 233, 356); LD(167, 234, 101); LD(168, 235, 357); LD(169, 236, 102); LD(170, 237, 358); LD(171, 239, 228); LD(172, 240, 484); LD(173, 241, 229); LD(174, 242, 485); LD(175, 243, 230); LD(176, 244, 486); LD(177, 246, 104); LD(178, 247, 360); LD(179, 248, 105); LD(180, 249, 361); LD(181, 250, 106); LD(182, 251, 362); LD(183, 253, 232); LD(184, 254, 488); LD(185, 255, 233); LD(186, 256, 489); LD(187, 257, 234); LD(188, 258, 490)

static void accumulate(real *Kijtmp, real *Mjtmp, real *Lptr)
{
  real Lij = ZERO;
  for (int k = 0; k < 189; k ++) { // LOOP WAS VECTORIZED.
    Lij += Kijtmp[k] * Mjtmp[k];
  }
  *Lptr += Lij;
}

static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) with the same
     sibling-index along each direction for this level */
  const int ncpec = POW2(level - 1) + 2;

  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);

  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  const int cx = bx % nch;
  const int cy = (bx % (nch * nch)) / nch;
  const int cz = bx / (nch * nch);
  
  /* Set a pointer to K; K[j][i][k], where i=j=k=0*/
  real *Kptr = K + (0 * cutoff + 0) * 316 + 0;

  /* Set a pointer to M wrt this chunk;
     M[level][j][s][B*cz+iz][B*cy+iy][B*cx+ix], where j=s=ix=iy=iz=0 */
  real *Mptr = M + Mstart + (((0 * 8 + 0) * ncpec + (B * cz + 0)) * ncpec + (B * cy + 0)) * ncpec + (B * cx + 0);

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[8][B + 2][B + 2][B + 2]; // cached?
    
    for (int s = 0; s < 8; s ++) { // sibling-index for source cells
      for (int iz = 0; iz < B + 2; iz ++) {
	for (int iy = 0; iy < B + 2; iy ++) {
	  for (int ix = 0; ix < B + 2; ix ++) {
	    //	    Mj[s][iz][iy][ix] = Mptr[(((j * 8 + s) * ncpec + iz) * ncpec + iy) * ncpec + ix];
	    Mj[s][iz][iy][ix] = Mptr[((s * ncpec + iz) * ncpec + iy) * ncpec + ix];
	  }
	}
      }
    }

    /* Point to next j */
    Mptr += 8 * ncpec * ncpec * ncpec;
    
    /* Set a pointer to L;
       L[chunk][i][iz][iy][ix][sib], where chunk=bx and i=iz=iy=ix=sib=0 */
    real *Lptr = L + ((((bx * cutoff + 0) * B + 0) * B + 0) * B + 0) * 8 + 0;
    
    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Load Kij */
      real Kij[316]; // cached?
      for (int k = 0; k < 316; k ++) {
	//	Kij[k] = K[(j * cutoff + i) * 316 + k];
	Kij[k] = Kptr[k];
      }
      Kptr += 316;
     
      //      /* Set a pointer to L;
      //	 L[chunk][i][iz][iy][ix][sib], where chunk=bx and iz=iy=ix=sib=0 */
      //      real *Lptr = L + ((((bx * cutoff + i) * B + 0) * B + 0) * B + 0) * 8 + 0;

      /* Loop over target cells with the same sibling-index */
      for (int iz = 0; iz < B; iz ++) {
	for (int iy = 0; iy < B; iy ++) {
	  for (int ix = 0; ix < B; ix ++) {
	    
	    /* Offset */
	    const int Mjshift = (iz * (B + 2) + iy) * (B + 2) + ix;

	    /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	       S) and accumulate Lij(F) to Li(F) (reduction for j) */
	    real Kijtmp[189], Mjtmp[189];
	    const real *Mjptr = (real *)Mj + Mjshift;
	    
	    /* Loop over sibling-indices of target cells */

	    if (B == 4) {
	      B4_LOADXYZ0();
	    } else if (B == 2) {
	      B2_LOADXYZ0();
	    }
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    if (B == 4) {
	      B4_LOADXYZ1();
	    } else if (B == 2) {
	      B2_LOADXYZ1();
	    }
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    if (B == 4) {
	      B4_LOADXYZ2();
	    } else if (B == 2) {
	      B2_LOADXYZ2();
	    }
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    if (B == 4) {
	      B4_LOADXYZ3();
	    } else if (B == 2) {
	      B2_LOADXYZ3();
	    }
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    if (B == 4) {
	      B4_LOADXYZ4();
	    } else if (B == 2) {
	      B2_LOADXYZ4();
	    }
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    if (B == 4) {
	      B4_LOADXYZ5();
	    } else if (B == 2) {
	      B2_LOADXYZ5();
	    }
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    if (B == 4) {
	      B4_LOADXYZ6();
	    } else if (B == 2) {
	      B2_LOADXYZ6();
	    }
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	    if (B == 4) {
	      B4_LOADXYZ7();
	    } else if (B == 2) {
	      B2_LOADXYZ7();
	    }
	    accumulate(Kijtmp, Mjtmp, Lptr);
	    Lptr ++;

	  } // ix
	} // iy
      } // iz

    } // i
  } // j
}

/**************************************************************************/
#elif defined(CPU9A)
/**************************************************************************/

#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Kijptr += Kijoff_diff;				\
  Lij += (*Kijptr) * (*Mjptr);

/* Created by aux_CPU9A.c */
#define B4_COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)

#define B2_COMPXYZ0() COMP(57, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ1() COMP(8, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ2() COMP(50, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ3() COMP(1, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ4() COMP(56, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ5() COMP(7, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ6() COMP(49, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ7() COMP(0, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)


static void m2l_kern_ij_blocking(real *L, real *K, real *M, const int cutoff, const int level, const int B, const int Mstart, const int bx)
{
  /* Number of cells (including two ghost cells) with the same
     sibling-index along each direction for this level */
  const int ncpec = POW2(level - 1) + 2;

  /* Number of chunks along each direction for this level */
  const int nch = POW2(level) / (2 * B);

  /* Compute the coordinates (cx,cy,cz) of this chunk, where
     0<=cx,cy,cz<2^l/(2*B) */
  const int cx = bx % nch;
  const int cy = (bx % (nch * nch)) / nch;
  const int cz = bx / (nch * nch);
  
  /* Set a pointer to M wrt this chunk;
     M[level][j][s][B*cz+iz][B*cy+iy][B*cx+ix], where j=s=ix=iy=iz=0 */
  real *Mptr = M + Mstart + (((0 * 8 + 0) * ncpec + (B * cz + 0)) * ncpec + (B * cy + 0)) * ncpec + (B * cx + 0);

  /* Loop over columns j */
  for (int j = 0; j < cutoff; j ++) {

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk */
    real Mj[8][B + 2][B + 2][B + 2]; // cached?
    
    for (int s = 0; s < 8; s ++) { // sibling-index for source cells
      for (int iz = 0; iz < B + 2; iz ++) {
	for (int iy = 0; iy < B + 2; iy ++) {
	  for (int ix = 0; ix < B + 2; ix ++) {
	    Mj[s][iz][iy][ix] = Mptr[(((j * 8 + s) * ncpec + iz) * ncpec + iy) * ncpec + ix];
	  }
	}
      }
    }
    
    /* Loop over rows i */
    for (int i = 0; i < cutoff; i ++) {

      /* Load Kij */
      real Kij[316]; // cached?
      for (int k = 0; k < 316; k ++) {
	Kij[k] = K[(j * cutoff + i) * 316 + k];
      }
     
      /* Set a pointer to L;
	 L[chunk][i][iz][iy][ix][sib], where chunk=bx and iz=iy=ix=sib=0 */
      real *Lptr = L + ((((bx * cutoff + i) * B + 0) * B + 0) * B + 0) * 8 + 0;

      /* Loop over target cells with the same sibling-index */
      for (int iz = 0; iz < B; iz ++) {
	for (int iy = 0; iy < B; iy ++) {
	  for (int ix = 0; ix < B; ix ++) {
	    
	    /* Offset */
	    const int Mjoff = (iz * (B + 2) + iy) * (B + 2) + ix;

	    /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for
	       S) and accumulate Lij(F) to Li(F) (reduction for j) */
	    real Lij, *Kijptr, *Mjptr;
	    
	    /* Loop over sibling-indices of target cells */
	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjoff;
	    if (B == 4)	{
	      B4_COMPXYZ0();
	    } else {
	      B2_COMPXYZ0();
	    }
	    *Lptr += Lij;
	    Lptr ++;
	    
	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjoff;
	    if (B == 4)	{
	      B4_COMPXYZ1();
	    } else {
	      B2_COMPXYZ1();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjoff;
	    if (B == 4)	{
	      B4_COMPXYZ2();
	    } else {
	      B2_COMPXYZ2();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjoff;
	    if (B == 4)	{
	      B4_COMPXYZ3();
	    } else {
	      B2_COMPXYZ3();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjoff;
	    if (B == 4)	{
	      B4_COMPXYZ4();
	    } else {
	      B2_COMPXYZ4();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjoff;
	    if (B == 4)	{
	      B4_COMPXYZ5();
	    } else {
	      B2_COMPXYZ5();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjoff;
	    if (B == 4)	{
	      B4_COMPXYZ6();
	    } else {
	      B2_COMPXYZ6();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	    Lij = ZERO;
	    Kijptr = Kij;
	    Mjptr = (real *)Mj + Mjoff;
	    if (B == 4)	{
	      B4_COMPXYZ7();
	    } else {
	      B2_COMPXYZ7();
	    }
	    *Lptr += Lij;
	    Lptr ++;

	  } // ix
	} // iy
      } // iz

    } // i
  } // j
}

/**************************************************************************/
#else
/**************************************************************************/
#error No minor version was specified.
/**************************************************************************/
#endif
/**************************************************************************/
/* Based on CUDA_VER45 */

#include "m2l_aux.h"
#include "auxAnotherFMMInteraction.h"
#include "m2l_aux_cpu.h"

#if !defined(LEVEL_TO_SWITCH_B2_TO_B4) // 3 or more
#define LEVEL_TO_SWITCH_B2_TO_B4 3 // default
#endif

#if !defined(LEVEL_TO_SWITCH_B4_TO_B8) // 4 or more
#define LEVEL_TO_SWITCH_B4_TO_B8 9999 // default (meaning this never happens)
#endif

void anotherFMMInteraction(anotherTree **atree, real *E, int *Ktable, 
			   real *U, real *Kweights, int n, int dof,
			   int cutoff, real homogen)
{
  timerType *timer_all;
  allocTimer(&timer_all);
  initTimer(timer_all);
  startTimer(timer_all);

  int n3 = n * n * n;
  int dofn3 = dof * n3;

  int ncell = (*atree)->ncell;
  int minlev = (*atree)->minlev;
  int maxlev = (*atree)->maxlev;
  int *levsta = (*atree)->levsta;
  int *levend = (*atree)->levend;
  real *celeng = (*atree)->celeng;
  cell *c = (*atree)->c;
  real3 *center = c->center;

  /* Allocate and initialise field values of all the real cells */
  (*atree)->fieldval = (real *)calloc(ncell * dofn3, sizeof(real));

  /* Shortcut for proxy source values (moment) */
  real *PS = (*atree)->proxysval;

  /* Allocate and initialise proxy field values of all the real cells */
  real *PF = (real *)calloc(ncell * dofn3, sizeof(real));

  /* Keep M2L transfer matrices as column-major */
  real *K = E;
    
  /* Allocate another K-matrix */
  real *Kanother = (real *)malloc(316 * cutoff * cutoff * sizeof(real));
  m2l_aux_convert_K_to_Kanother_for_ij_blocking_row1_col1(cutoff, Ktable, K, Kanother);

  /* Compute the number of all the real and ghost cells */
  const int ncellanother = m2l_aux_get_number_of_real_and_ghost_cells_for_ij_blocking(minlev, maxlev);

  /* Allocate and initialize another M-vector for both real and ghost
     cells */
  real *Manother = (real *)calloc(ncellanother * cutoff, sizeof(real));
#if defined(CPU9F) || defined(CPU9G) || defined(CPU9H) || defined(CPU9I) || defined(CPU9J) || defined(CPU9K) || defined(CPU9L) || defined(CPU9M) || defined(CPU9N) || defined(CPU9O) || defined(CPU9P) || defined(CPU9Q) || defined(CPU9R) || defined(CPU9S) || defined(CPU9T) || defined(CPU9U)
  m2l_aux_convert_M_to_Manother_for_ij_blocking_col1_CPU(cutoff, minlev, maxlev, center, celeng[0], PS, Manother);
#else
  m2l_aux_convert_M_to_Manother_for_ij_blocking_col1(cutoff, minlev, maxlev, center, celeng[0], PS, Manother);
#endif

  INFO("LEVEL_TO_SWITCH_B2_TO_B4=%d\n", LEVEL_TO_SWITCH_B2_TO_B4);
  INFO("LEVEL_TO_SWITCH_B4_TO_B8=%d\n", LEVEL_TO_SWITCH_B4_TO_B8);

  timerType *timer_kernel;
  allocTimer(&timer_kernel);
  initTimer(timer_kernel);

  /* Loop over levels */
  for (int level = minlev; level <= maxlev; level ++) {
      
    /* Indices of the first and last cells in this level */
    const int Asta = levsta[level];
    const int Aend = levend[level];

    /* Number of cells in this level */
    const int nc = Aend - Asta + 1;

    /* Compute the starting index of Manother */    
    const int Manotherstart = m2l_aux_get_starting_index_of_Manother_for_ij_blocking(cutoff, minlev, level);

    /* Allocate and initialise another L for this level */
    real *Lanother = (real *)calloc(nc * cutoff, sizeof(real));

    /* Set chunk size */
    int B;
    //    if (level < LEVEL_TO_SWITCH_B2_TO_B4) {
    //      B = 2;
    //    } else {
    //      B = 4;
    //    }

    if (level >= LEVEL_TO_SWITCH_B4_TO_B8) {
      B = 8;
    } else if (level >= LEVEL_TO_SWITCH_B2_TO_B4) {
      B = 4;
    } else {
      B = 2;
    }

    /* Number of chunks in this level */
    const int nchunk = CUBE(POW2(level) / (2 * B));
    INFO("level=%d B=%d nchunk=%d\n", level, B, nchunk);

    /* Loop over target chunks */

    startTimer(timer_kernel);

#if defined(CPU9O) || defined(CPU9R)

    for (int chunk = 0; chunk < nchunk; chunk ++) {
      /* Compute another L */
      m2l_kern_ij_blocking(Lanother, Kanother, Manother, cutoff, level, B, Manotherstart, chunk);
    }

#else

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int chunk = 0; chunk < nchunk; chunk ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      /* Compute another L */
#if defined(CPU9C)
      if (B == 2) {
	m2l_kern_ij_blocking_b2(Lanother, Kanother, Manother, cutoff, level, B, Manotherstart, chunk);
      } else {
	m2l_kern_ij_blocking_b4(Lanother, Kanother, Manother, cutoff, level, B, Manotherstart, chunk);
      }
#else
      m2l_kern_ij_blocking(Lanother, Kanother, Manother, cutoff, level, B, Manotherstart, chunk);
#endif
    }

#endif

    stopTimer(timer_kernel);

    /* Convert Lanother to L */
#if defined(CPU9I) || defined(CPU9J) || defined(CPU9K) || defined(CPU9L) || defined(CPU9M) || defined(CPU9N) || defined(CPU9O) || defined(CPU9P) || defined(CPU9Q) || defined(CPU9R) || defined(CPU9S) || defined(CPU9T) || defined(CPU9U)
    m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1_CPU2(cutoff, center, celeng[0], level, B, Lanother, &(PF[cutoff * Asta]));
#else
    m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1_CPU(cutoff, center, celeng[0], level, B, Lanother, &(PF[cutoff * Asta]));
#endif

    /* Free Lanother */
    free(Lanother);

  } // level

  /* Free */
  free(Manother);
  free(Kanother);

  /* Free proxy source value */
  free((*atree)->proxysval);

  /* Check kernel performance */
#if defined(CHECK_PERFORMANCE)
  double perf = m2l_aux_comp_kernel_performance_in_Gflops(cutoff, minlev, maxlev, levsta, levend, c->iinter, getTimer(*timer_kernel));
  INFO("calc_performance: kernel = %f [Gflop/s]\n", perf);
#endif

  /*
    Translate proxy field value to field value (post M2L)
  */
  
  timerType *timer_kernel2;
  allocTimer(&timer_kernel2);
  initTimer(timer_kernel2);

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
    startTimer(timer_kernel2);
    postm2l(levsta[level], levend[level], dofn3, cutoff, Ur, &(PF[cutoff * levsta[level]]), scale, adjust, (*atree)->fieldval);
    stopTimer(timer_kernel2);
  }

  free(Ur);
  free(adjust);
  free(PF);

  /* Finalise timers */
  printTimer(stderr, "kernel", timer_kernel);
  printTimer(stderr, "kernel2", timer_kernel2);

  freeTimer(&timer_kernel);
  freeTimer(&timer_kernel2);

  stopTimer(timer_all);
  printTimer(stderr, "all", timer_all);
  freeTimer(&timer_all);
}
/**************************************************************************/
#elif defined(CPU7)
/**************************************************************************/
/* Define 27 interction-kinds. See Table 4 (left) in the paper; this
   is from m2l_kern_sibling_blocking.cu */
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

/**************************************************************************/
#if defined(CPU7A)
/**************************************************************************/
/* Based on CPU6D */

#include "m2l_aux.h"

#ifndef PITCH_SOURCECLUSTERS
#define PITCH_SOURCECLUSTERS 32 // 27 or more
#endif

static void px8(const int cutoff, const real *K,
		real *L0, real *L1, real *L2, real *L3,
		real *L4, real *L5, real *L6, real *L7,
		const real *M0, const real *M1, const real *M2, const real *M3,
		const real *M4, const real *M5, const real *M6, const real *M7, const int kbase)
{
  {

    const int n = cutoff;
    const real *Kptr = K + kbase * n * n; // row-major

    for (int i = 0; i < n; i ++) { // no unrolling

      real sum0a = ZERO;
      real sum1a = ZERO;
      real sum2a = ZERO;
      real sum3a = ZERO;
      real sum4a = ZERO;
      real sum5a = ZERO;
      real sum6a = ZERO;
      real sum7a = ZERO;
      
      const real *Ki0 = Kptr + i * n;

      for (int j = 0; j < n; j ++) { // no message?
	const real tmpa = *(Ki0 + j);
	sum0a += tmpa * *(M0 + j);
	sum1a += tmpa * *(M1 + j);
	sum2a += tmpa * *(M2 + j);
	sum3a += tmpa * *(M3 + j);
	sum4a += tmpa * *(M4 + j);
	sum5a += tmpa * *(M5 + j);
	sum6a += tmpa * *(M6 + j);
	sum7a += tmpa * *(M7 + j);
      } // j
     
     *(L0 + i    ) += sum0a;
     *(L1 + i    ) += sum1a;
     *(L2 + i    ) += sum2a;
     *(L3 + i    ) += sum3a;
     *(L4 + i    ) += sum4a;
     *(L5 + i    ) += sum5a;
     *(L6 + i    ) += sum6a;
     *(L7 + i    ) += sum7a;
    } // i

  }
}

static void px4(const int cutoff, const real *K,
		real *L0, real *L1, real *L2, real *L3,
		const real *M0, const real *M1, const real *M2, const real *M3, const int kbase)
{
  if (kbase != NULL_KINDEX) {

    const int n = cutoff;
    const real *Kptr = K + kbase * n * n; // row-major

    for (int i = 0; i < n; i += 2) { // unrolling 2x

      real sum0a = ZERO;
      real sum1a = ZERO;
      real sum2a = ZERO;
      real sum3a = ZERO;
      real sum0b = ZERO;
      real sum1b = ZERO;
      real sum2b = ZERO;
      real sum3b = ZERO;

      const real *Ki0 = Kptr + i * n;
      const real *Ki1 = Ki0 + n;

      for (int j = 0; j < n; j ++) { // PARTIAL LOOP WAS VECTORIZED.
	const real tmpa = *(Ki0 + j);
	sum0a += tmpa * *(M0 + j);
	sum1a += tmpa * *(M1 + j);
	sum2a += tmpa * *(M2 + j);
	sum3a += tmpa * *(M3 + j);
	const real tmpb = *(Ki1 + j);
	sum0b += tmpb * *(M0 + j);
	sum1b += tmpb * *(M1 + j);
	sum2b += tmpb * *(M2 + j);
	sum3b += tmpb * *(M3 + j);
      } // j

      *(L0 + i    ) += sum0a;
      *(L0 + i + 1) += sum0b;
      *(L1 + i    ) += sum1a;
      *(L1 + i + 1) += sum1b;
      *(L2 + i    ) += sum2a;
      *(L2 + i + 1) += sum2b;
      *(L3 + i    ) += sum3a;
      *(L3 + i + 1) += sum3b;

    } // i

  }
}

static void px2(const int cutoff, const real *K,
		real *L0, real *L1,
		const real *M0, const real *M1, const int kbase)
{
  if (kbase != NULL_KINDEX) {

    const int n = cutoff;
    const real *Kptr = K + kbase * n * n; // row-major

    for (int i = 0; i < n; i += 4) { // unrolling 4x

      real sum0a = ZERO;
      real sum1a = ZERO;
      real sum0b = ZERO;
      real sum1b = ZERO;
      real sum0c = ZERO;
      real sum1c = ZERO;
      real sum0d = ZERO;
      real sum1d = ZERO;

      const real *Ki0 = Kptr + i * n;
      const real *Ki1 = Ki0 + n;
      const real *Ki2 = Ki1 + n;
      const real *Ki3 = Ki2 + n;

      for (int j = 0; j < n; j ++) { // LOOP WAS VECTORIZED.
	const real M0j = *(M0 + j);
	const real M1j = *(M1 + j);
	const real tmpa = *(Ki0 + j);
	sum0a += tmpa * M0j;
	sum1a += tmpa * M1j;
	const real tmpb = *(Ki1 + j);
	sum0b += tmpb * M0j;
	sum1b += tmpb * M1j;
	const real tmpc = *(Ki2 + j);
	sum0c += tmpc * M0j;
	sum1c += tmpc * M1j;
	const real tmpd = *(Ki3 + j);
	sum0d += tmpd * M0j;
	sum1d += tmpd * M1j;
      } // j

      *(L0 + i    ) += sum0a;
      *(L0 + i + 1) += sum0b;
      *(L0 + i + 2) += sum0c;
      *(L0 + i + 3) += sum0d;
      *(L1 + i    ) += sum1a;
      *(L1 + i + 1) += sum1b;
      *(L1 + i + 2) += sum1c;
      *(L1 + i + 3) += sum1d;

    } // i

  }
}

static void px1(const int cutoff, const real *K, real *L0, const real *M0, const int kbase)
{
  if (kbase != NULL_KINDEX) {

    const int n = cutoff;
    const real *Kptr = K + kbase * n * n; // row-major

#if defined(SINGLE)
    for (int i = 0; i < n; i += 8) { // unrolling 8x; best for single-precision
#else
    for (int i = 0; i < n; i += 4) { // unrolling 4x; best for double-precision
#endif
      real sum0a = ZERO;
      real sum0b = ZERO;
      real sum0c = ZERO;
      real sum0d = ZERO;
#if defined(SINGLE)
      real sum0e = ZERO;
      real sum0f = ZERO;
      real sum0g = ZERO;
      real sum0h = ZERO;
#endif

      const real *Ki0 = Kptr + i * n;
      const real *Ki1 = Ki0 + n;
      const real *Ki2 = Ki1 + n;
      const real *Ki3 = Ki2 + n;
#if defined(SINGLE)
      const real *Ki4 = Ki3 + n;
      const real *Ki5 = Ki4 + n;
      const real *Ki6 = Ki5 + n;
      const real *Ki7 = Ki6 + n;
#endif

      for (int j = 0; j < n; j ++) { // LOOP WAS VECTORIZED.
	const real M0j = *(M0 + j);
	sum0a += *(Ki0 + j) * M0j;
	sum0b += *(Ki1 + j) * M0j;
	sum0c += *(Ki2 + j) * M0j;
	sum0d += *(Ki3 + j) * M0j;
#if defined(SINGLE)
	sum0e += *(Ki4 + j) * M0j;
	sum0f += *(Ki5 + j) * M0j;
	sum0g += *(Ki6 + j) * M0j;
	sum0h += *(Ki7 + j) * M0j;
#endif
      } // j

      *(L0 + i    ) += sum0a;
      *(L0 + i + 1) += sum0b;
      *(L0 + i + 2) += sum0c;
      *(L0 + i + 3) += sum0d;
#if defined(SINGLE)
      *(L0 + i + 4) += sum0e;
      *(L0 + i + 5) += sum0f;
      *(L0 + i + 6) += sum0g;
      *(L0 + i + 7) += sum0h;
#endif

    } // i

  }
}


static void comp_Kindex0_Kindex1(int *Kindex0, int *Kindex1)
{
  /* K-index is determined from source-cluster index and
     interaction-kind. Namely, the index is given by
     Ktable[kindex0[source-cluster index
     0:26]+kindex1[interaction-kind 0:26]] */
  Kindex0[ 0] =  57; Kindex1[ 0] =  57;
  Kindex0[ 1] = 155; Kindex1[ 1] =  55;
  Kindex0[ 2] = 253; Kindex1[ 2] =  43;
  Kindex0[ 3] =  71; Kindex1[ 3] =  41;
  Kindex0[ 4] = 169; Kindex1[ 4] = -41;
  Kindex0[ 5] = 267; Kindex1[ 5] = -43;
  Kindex0[ 6] =  85; Kindex1[ 6] = -55;
  Kindex0[ 7] = 183; Kindex1[ 7] = -57;
  Kindex0[ 8] = 281; Kindex1[ 8] =  56;
  Kindex0[ 9] =  59; Kindex1[ 9] =  50;
  Kindex0[10] = 157; Kindex1[10] =   8;
  Kindex0[11] = 255; Kindex1[11] =  48;
  Kindex0[12] =  73; Kindex1[12] =   6;
  Kindex0[13] = 171; Kindex1[13] =  42;
  Kindex0[14] = 269; Kindex1[14] =  -6;
  Kindex0[15] =  87; Kindex1[15] =  -8;
  Kindex0[16] = 185; Kindex1[16] = -42;
  Kindex0[17] = 283; Kindex1[17] = -48;
  Kindex0[18] =  61; Kindex1[18] = -50;
  Kindex0[19] = 159; Kindex1[19] = -56;
  Kindex0[20] = 257; Kindex1[20] =  49;
  Kindex0[21] =  75; Kindex1[21] =   7;
  Kindex0[22] = 173; Kindex1[22] =   1;
  Kindex0[23] = 271; Kindex1[23] =  -1;
  Kindex0[24] =  89; Kindex1[24] =  -7;
  Kindex0[25] = 187; Kindex1[25] = -49;
  Kindex0[26] = 285; Kindex1[26] =   0;
}

static void m2lx8421(const int A0, const int F0, const int cutoff, real *L, const real *K, const real *M,
		     const int *sourceclusters, const int pitch_sourceclusters,
		     const int *Ktable, const int *Kindex0, const int *Kindex1)
{
  /* Set the pointers to the local-coefficients of the eight children
     in the relevant field cluster (FC) */

  real *L0 = L + A0 * cutoff;
  real *L1 = L0 + cutoff;
  real *L2 = L1 + cutoff;
  real *L3 = L2 + cutoff;
  real *L4 = L3 + cutoff;
  real *L5 = L4 + cutoff;
  real *L6 = L5 + cutoff;
  real *L7 = L6 + cutoff;
  
  /* Load the list of 0th siblings (B0) in source clusters (SC) that
     interact with FC */
  const int *B0 = &(sourceclusters[pitch_sourceclusters * F0]); // B0[0:26]

  /* Loop over source-cluster indices */
  for (int d = 0; d < 27; d ++) {
    
    /* If B0, which is the 0th sibling of SC, does not exist in the
       hierachy (then, the other seven siblings in SC neither exist)
       or SC coincides with FC, we can skip the computaion for SC */
    if (B0[d] != NULL_CELL) {

      /* Set the pointer to the moments of the eight children in SC */
      const real *M0 = M + B0[d] * cutoff;
      const real *M1 = M0 + cutoff;
      const real *M2 = M1 + cutoff;
      const real *M3 = M2 + cutoff;
      const real *M4 = M3 + cutoff;
      const real *M5 = M4 + cutoff;
      const real *M6 = M5 + cutoff;
      const real *M7 = M6 + cutoff;

      px1(cutoff, K, L0, M7, Ktable[Kindex0[d] + Kindex1[F0S7]]);
      px1(cutoff, K, L1, M6, Ktable[Kindex0[d] + Kindex1[F1S6]]);
      px1(cutoff, K, L2, M5, Ktable[Kindex0[d] + Kindex1[F2S5]]);
      px1(cutoff, K, L3, M4, Ktable[Kindex0[d] + Kindex1[F3S4]]);
      px1(cutoff, K, L4, M3, Ktable[Kindex0[d] + Kindex1[F4S3]]);
      px1(cutoff, K, L5, M2, Ktable[Kindex0[d] + Kindex1[F5S2]]);
      px1(cutoff, K, L6, M1, Ktable[Kindex0[d] + Kindex1[F6S1]]);
      px1(cutoff, K, L7, M0, Ktable[Kindex0[d] + Kindex1[F7S0]]);

      px2(cutoff, K, L0, L1, M6, M7, Ktable[Kindex0[d] + Kindex1[F0S6]]);
      px2(cutoff, K, L0, L2, M5, M7, Ktable[Kindex0[d] + Kindex1[F0S5]]);
      px2(cutoff, K, L0, L4, M3, M7, Ktable[Kindex0[d] + Kindex1[F0S3]]);
      px2(cutoff, K, L1, L3, M4, M6, Ktable[Kindex0[d] + Kindex1[F1S4]]);
      px2(cutoff, K, L1, L5, M2, M6, Ktable[Kindex0[d] + Kindex1[F1S2]]);
      px2(cutoff, K, L2, L3, M4, M5, Ktable[Kindex0[d] + Kindex1[F2S4]]);
      px2(cutoff, K, L2, L6, M1, M5, Ktable[Kindex0[d] + Kindex1[F2S1]]);
      px2(cutoff, K, L3, L7, M0, M4, Ktable[Kindex0[d] + Kindex1[F3S0]]);
      px2(cutoff, K, L4, L5, M2, M3, Ktable[Kindex0[d] + Kindex1[F4S2]]);
      px2(cutoff, K, L4, L6, M1, M3, Ktable[Kindex0[d] + Kindex1[F4S1]]);
      px2(cutoff, K, L5, L7, M0, M2, Ktable[Kindex0[d] + Kindex1[F5S0]]);
      px2(cutoff, K, L6, L7, M0, M1, Ktable[Kindex0[d] + Kindex1[F6S0]]);
      
      px4(cutoff, K, L0, L1, L2, L3, M4, M5, M6, M7, Ktable[Kindex0[d] + Kindex1[F0S4]]);
      px4(cutoff, K, L0, L1, L4, L5, M2, M3, M6, M7, Ktable[Kindex0[d] + Kindex1[F0S2]]);
      px4(cutoff, K, L0, L2, L4, L6, M1, M3, M5, M7, Ktable[Kindex0[d] + Kindex1[F0S1]]);
      px4(cutoff, K, L1, L3, L5, L7, M0, M2, M4, M6, Ktable[Kindex0[d] + Kindex1[F1S0]]);
      px4(cutoff, K, L2, L3, L6, L7, M0, M1, M4, M5, Ktable[Kindex0[d] + Kindex1[F2S0]]);
      px4(cutoff, K, L4, L5, L6, L7, M0, M1, M2, M3, Ktable[Kindex0[d] + Kindex1[F4S0]]);
      
      px8(cutoff, K, L0, L1, L2, L3, L4, L5, L6, L7, M0, M1, M2, M3, M4, M5, M6, M7, Ktable[Kindex0[d] + Kindex1[F0S0]]);
    }
  } // d
}
/**************************************************************************/
#else
/**************************************************************************/
#error No minor version was specified.
/**************************************************************************/
#endif
/**************************************************************************/

#include "auxAnotherFMMInteraction.h"

void anotherFMMInteraction(anotherTree **atree, real *E, int *Ktable, 
			    real *U, real *Kweights, int n, int dof,
			    int cutoff, real homogen)
{
  timerType *timer_all;
  allocTimer(&timer_all);
  initTimer(timer_all);
  startTimer(timer_all);

  const int cutoff2 = cutoff * cutoff;
  const int n3 = n * n * n;                      // n3 = n^3
  const int dofn3 = dof * n3;

  const int ncell = (*atree)->ncell;
  const int minlev = (*atree)->minlev;
  const int maxlev = (*atree)->maxlev;
  const int *levsta = (*atree)->levsta;
  const int *levend = (*atree)->levend;
  const real *celeng = (*atree)->celeng;
  cell *c = (*atree)->c;

#if !defined(FAST_HOST_CODE)
  const real zero = 0;
  const char trans[] = "n";
  const int incr = 1;
#endif

  timerType *timer_conv, *timer_kernel, *timer_kernel2;
  allocTimer(&timer_conv);
  allocTimer(&timer_kernel);
  allocTimer(&timer_kernel2);
  initTimer(timer_conv);
  initTimer(timer_kernel);
  initTimer(timer_kernel2);

  /* Convert M2L transfer matrices from column-major to row-major */
  startTimer(timer_conv);
  real *K = (real *)malloc(316 * cutoff2 * sizeof(real));
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int k = 0; k < 316; k ++) {
    for (int row = 0; row < cutoff; row ++) {
      for (int col = 0; col < cutoff; col ++) {
	K[cutoff2 * k + cutoff * row + col] = E[cutoff2 * k + row + cutoff * col];
      }
    }
  }
  stopTimer(timer_conv);

  /* Allocate and initialise proxy field values at field Chebyshev
     nodes */
  (*atree)->fieldval = (real *)calloc(ncell * dofn3, sizeof(real));

  /* Allocate and initialise proxy field values to zero */
  real *Pf = (real *)calloc(cutoff * ncell, sizeof(real));

  /* Shortcut to proxy source value */
  real *Ps = (*atree)->proxysval;

#if defined(FAST_HOST_CODE)
  /* Convert U from column-major to row-major */
  real *Ur = (real *)malloc(dofn3 * cutoff * sizeof(real));
  startTimer(timer_conv);
  postm2l_convert_U_from_column_major_to_row_major(U, Ur, dofn3, cutoff);
  stopTimer(timer_conv);

  /* Precompute adjusting vector */
  real *adjust = (real *)malloc(dofn3 * sizeof(real));
  startTimer(timer_conv);
  postm2l_compute_adjusting_vector(Kweights, adjust, dof, n3);
  stopTimer(timer_conv);
#endif

  /* Load the auxiliary tables for Ktable */
  int Kindex0[27], Kindex1[27];
  comp_Kindex0_Kindex1(Kindex0, Kindex1);

  /* Compute number of cells with child-type of 0 between levels
     minlev(>=2) and maxlev, assuming to use a complete oct-tree. */
  const int nallcluster = (POW8(maxlev) - POW8(minlev - 1)) / 7;

  /* Compute source clusters */
  int pitch_sourceclusters = PITCH_SOURCECLUSTERS;
  int *sourceclusters = (int *)malloc(pitch_sourceclusters * nallcluster * sizeof(int));
#if defined(ENABLE_USE_PARENT_LEAVES_ARRAYS)
  m2l_aux_comp_sourceclusters(minlev, maxlev, celeng[0], c->center, c->parent, c->leaves,
			      c->ineigh, c->pitch_neighbors, c->neighbors,
			      pitch_sourceclusters, sourceclusters);
#else
  m2l_aux_comp_sourceclusters(minlev, maxlev, celeng[0], c->center,
			      c->ineigh, c->pitch_neighbors, c->neighbors,
			      pitch_sourceclusters, sourceclusters);
#endif

  /* Loop over levels */
  for (int level = minlev; level <= maxlev; level ++) {

    const real L = celeng[level];              // Length of cell
    const real iL = 1 / L;                     // Inverse-length  
    const real scale = POW((real)iL, homogen); // Scaling factor for SVD

    /*
      Translate proxy source values to proxy field values (M2L)
    */
    
    startTimer(timer_kernel); // Perhaps, this must be outside of the omp directive...

    /* Loop over field clusters */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int F = 0; F < (levend[level] - levsta[level] + 1) / 8; F ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.

      /* Obtain the index of field cell of child-type 0 */
      const int A0 = levsta[level] + 8 * F;
      
      /* Obtain the index of field cluster (# of field clusters in
	 levels between minlev and level-1 plus the index of the
	 relevant level) */
      const int F0 = (POW8(level - 1) - POW8(minlev - 1)) / 7 + F;
      
      /* Perform M2L */
      m2lx8421(A0, F0, cutoff, Pf, K, Ps,
      	       sourceclusters, pitch_sourceclusters, Ktable, Kindex0, Kindex1);

    } // F

    stopTimer(timer_kernel);

    /*
      Translate proxy field values to field values (post-M2L)
    */

    startTimer(timer_kernel2); // Perhaps, this must be outside of the omp directive...

#if defined(FAST_HOST_CODE)

    postm2l(levsta[level], levend[level], dofn3, cutoff,
	    Ur, &(Pf[cutoff * levsta[level]]), scale, adjust, (*atree)->fieldval);

#else

    /* Loop over cell A */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      
      /* Initialize pointer to the field values of A */
      real *F = &((*atree)->fieldval[dofn3 * A]);
      
      /* Compute the field values at the field Chebyshev nodes */
      agemv_(trans, &dofn3, &cutoff, &scale, U, &dofn3, &Pf[cutoff * A],
	     &incr, &zero, F, &incr);
      
      /* Adjust the field values by the appropriate weight */
      int l = 0;
      for (int i = 0; i < n3; i++) {
	const real tmp = Kweights[i];
	for (int j = 0; j < dof; j++) { // LOOP WAS VECTORIZED.
	  F[l] *= tmp;
	  l++;
	}
      }
    }

#endif

    stopTimer(timer_kernel2);

  } /* level */

  free(sourceclusters);

  /* Free local variables */
#if defined(FAST_HOST_CODE)
  free(Ur);
  free(adjust);
#endif
  free(Pf);
  free(K);

  /* Free proxysval (keep fieldval) */
  free((*atree)->proxysval);

#if defined(CHECK_PERFORMANCE)
  /* Floating-operation count for one interaction (L=K*M) */
  double flop_per_interaction = cutoff * (2 * cutoff - 1); /* mul: cutoff^2, add: cutoff*(cutoff-1) */
  /* Count the number of interactions */
  double num_interactions = 0;
  for (int level = minlev; level <= maxlev; level++) {
    for (int A = levsta[level]; A <= levend[level]; A++) {
      num_interactions += (double)(c->iinter[A]);
    }
  }
  INFO("num_interactions = %f\n", num_interactions);
  /* Compute performance [Gflop/s] */
  double flop = flop_per_interaction * num_interactions;
  INFO("flop = %f\n", flop);
  calc_performance("kernel", flop, getTimer(*timer_kernel));
#endif

  /* Finalise timer */
  printTimer(stderr, "conv", timer_conv);
  printTimer(stderr, "kernel", timer_kernel);
  printTimer(stderr, "kernel2", timer_kernel2);

  freeTimer(&timer_conv);
  freeTimer(&timer_kernel);
  freeTimer(&timer_kernel2);

  stopTimer(timer_all);
  printTimer(stderr, "all", timer_all);
  freeTimer(&timer_all);
}

/**************************************************************************/
#else
/**************************************************************************/
void anotherFMMInteraction(anotherTree **atree, real *E, int *Ktable, 
			   real *U, real *Kweights, int n, int dof,
			   int cutoff, real homogen)
{
  /* This is necessary only for compiling direct method */
  exit(EXIT_FAILURE);
}
/**************************************************************************/
#endif
/**************************************************************************/
