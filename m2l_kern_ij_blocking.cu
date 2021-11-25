#ifndef M2L_KERN_IJ_BLOCKING_CU
#define M2L_KERN_IJ_BLOCKING_CU

#ifndef POW2
#define POW2(n) (1 << (n))
#endif

#ifndef POW4
#define POW4(i) (1 << ((i) << 1))
#endif

/**************************************************************************/
#if defined(CUDA_VER45)
/**************************************************************************/
/**************************************************************************/
#if defined(CUDA_VER45I)
/**************************************************************************/
/* Based on VER45G */

#include "real.h"

#if !defined(K_IS_1X2_TEXTURE) && !defined(K_IS_4X1_TEXTURE) && !defined(K_IS_4X1) && !defined(K_IS_8X1) && !defined(K_IS_16X1) && !defined(K_IS_8X2) && !defined(K_IS_4X4) && !defined(K_IS_4X2) && !defined(K_IS_1X1)
#error Set an appropriate macro.
#endif

/* Dx, Dy, and Dz denotes the dimensions of a chunk. Namely, each
   chunk consists of Dx*Dy*Dz clusters or (2*Dx)*(2*Dy)*(2*Dz)
   cells. In this code, Dx=Dy=Dz=4 is assumed and this number
   corresponds to 'B' in the paper and manual */

#define bx blockIdx.x   // chunk index
#define by blockIdx.y   // row-group index

#define tx threadIdx.x  // 0<=tx<Dx*Dz
#define ty threadIdx.y  // 0<=ty<Dy
#define tz threadIdx.z  // 0<=tz<8, where tz is sibling-index of field cell

/* cutoff stands for the dimension of M-vector, L-vector, and
   K-matrix. This corresponds to 'r' in the paper and manual.  In this
   code, r is either 256 (high-precision version) or 32 (low-precision
   version) */
#define CUTOFF_H     256
#define LOG_CUTOFF_H   8
#define CUTOFF_L      32
#define LOG_CUTOFF_L   5

/* Set the number of rows per row-group. This parameter corresponds to
   'P' in the paper and manual */
#if !defined(NUM_ROW_GROUPS_IJ)
//110312#define NUM_ROW_GROUPS_IJ 8 // 8 is better for C2050+SDK3.2
#define NUM_ROW_GROUPS_IJ 4
#endif
#if (NUM_ROW_GROUPS_IJ == 1)
#define NROWS_H 256
#define NROWS_L  32
#elif (NUM_ROW_GROUPS_IJ == 2)
#define NROWS_H 128
#define NROWS_L  16
#elif (NUM_ROW_GROUPS_IJ == 4)
#define NROWS_H  64
#define NROWS_L   8
#elif (NUM_ROW_GROUPS_IJ == 8)
#define NROWS_H  32
#define NROWS_L   4
#elif (NUM_ROW_GROUPS_IJ == 16)
#define NROWS_H  16
#define NROWS_L   2
#elif (NUM_ROW_GROUPS_IJ == 32)
#define NROWS_H   8
#define NROWS_L   1
#elif (NUM_ROW_GROUPS_IJ == 64)
#define NROWS_H   4
#define NROWS_L   0
#else
#error Unsupposed NUM_ROW_GROUPS_IJ.
#endif

/* Macros to perform Li+=Kij*Mj for all the 316 Kij */
#if defined(K_IS_1X2_TEXTURE)
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real2 Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real Kijtmp = *Kijptr;			\
    Lij += Kijtmp.x * Mjtmp.x;			\
    Lij += Kijtmp.y * Mjtmp.y;			\
  }
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real4 Kijtmp = *Kijptr;			\
    Lij.x += Kijtmp.x * Mjtmp;			\
    Lij.y += Kijtmp.y * Mjtmp;			\
    Lij.z += Kijtmp.z * Mjtmp;			\
    Lij.w += Kijtmp.w * Mjtmp;			\
  }
#elif defined(K_IS_8X1)
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real8 Kijtmp = *Kijptr;			\
    Lij.a += Kijtmp.a * Mjtmp;			\
    Lij.b += Kijtmp.b * Mjtmp;			\
    Lij.c += Kijtmp.c * Mjtmp;			\
    Lij.d += Kijtmp.d * Mjtmp;			\
    Lij.e += Kijtmp.e * Mjtmp;			\
    Lij.f += Kijtmp.f * Mjtmp;			\
    Lij.g += Kijtmp.g * Mjtmp;			\
    Lij.h += Kijtmp.h * Mjtmp;			\
  }
#elif defined(K_IS_16X1)
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real16 Kijtmp = *Kijptr;			\
    Lij.a += Kijtmp.a * Mjtmp;			\
    Lij.b += Kijtmp.b * Mjtmp;			\
    Lij.c += Kijtmp.c * Mjtmp;			\
    Lij.d += Kijtmp.d * Mjtmp;			\
    Lij.e += Kijtmp.e * Mjtmp;			\
    Lij.f += Kijtmp.f * Mjtmp;			\
    Lij.g += Kijtmp.g * Mjtmp;			\
    Lij.h += Kijtmp.h * Mjtmp;			\
    Lij.i += Kijtmp.i * Mjtmp;			\
    Lij.j += Kijtmp.j * Mjtmp;			\
    Lij.k += Kijtmp.k * Mjtmp;			\
    Lij.l += Kijtmp.l * Mjtmp;			\
    Lij.m += Kijtmp.m * Mjtmp;			\
    Lij.n += Kijtmp.n * Mjtmp;			\
    Lij.o += Kijtmp.o * Mjtmp;			\
    Lij.p += Kijtmp.p * Mjtmp;			\
  }
#elif defined(K_IS_8X2)
#if(1) // original (ptx has 16 FMA)
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real2 Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real8x2 Kijtmp = *Kijptr;			\
    Lij.a += Kijtmp.aa * Mjtmp.x;		\
    Lij.a += Kijtmp.ab * Mjtmp.y;		\
    Lij.b += Kijtmp.ba * Mjtmp.x;		\
    Lij.b += Kijtmp.bb * Mjtmp.y;		\
    Lij.c += Kijtmp.ca * Mjtmp.x;		\
    Lij.c += Kijtmp.cb * Mjtmp.y;		\
    Lij.d += Kijtmp.da * Mjtmp.x;		\
    Lij.d += Kijtmp.db * Mjtmp.y;		\
    Lij.e += Kijtmp.ea * Mjtmp.x;		\
    Lij.e += Kijtmp.eb * Mjtmp.y;		\
    Lij.f += Kijtmp.fa * Mjtmp.x;		\
    Lij.f += Kijtmp.fb * Mjtmp.y;		\
    Lij.g += Kijtmp.ga * Mjtmp.x;		\
    Lij.g += Kijtmp.gb * Mjtmp.y;		\
    Lij.h += Kijtmp.ha * Mjtmp.x;		\
    Lij.h += Kijtmp.hb * Mjtmp.y;		\
  }
#endif
#if(0) // same as original in ptx level
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real2 Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real8x2 Kijtmp = *Kijptr;			\
    real tmp;					\
    tmp = Lij.a + Kijtmp.aa * Mjtmp.x;		\
    Lij.a = tmp + Kijtmp.ab * Mjtmp.y;		\
    tmp = Lij.b + Kijtmp.ba * Mjtmp.x;		\
    Lij.b = tmp + Kijtmp.bb * Mjtmp.y;		\
    tmp = Lij.c + Kijtmp.ca * Mjtmp.x;		\
    Lij.c = tmp + Kijtmp.cb * Mjtmp.y;		\
    tmp = Lij.d + Kijtmp.da * Mjtmp.x;		\
    Lij.d = tmp + Kijtmp.db * Mjtmp.y;		\
    tmp = Lij.e + Kijtmp.ea * Mjtmp.x;		\
    Lij.e = tmp + Kijtmp.eb * Mjtmp.y;		\
    tmp = Lij.f + Kijtmp.fa * Mjtmp.x;		\
    Lij.f = tmp + Kijtmp.fb * Mjtmp.y;		\
    tmp = Lij.g + Kijtmp.ga * Mjtmp.x;		\
    Lij.g = tmp + Kijtmp.gb * Mjtmp.y;		\
    tmp = Lij.h + Kijtmp.ha * Mjtmp.x;		\
    Lij.h = tmp + Kijtmp.hb * Mjtmp.y;		\
  }
#endif
#if(0) // worse than the original
#define COMP(Kijoff_diff, Mjoff_diff)			\
  {							\
    Mjptr += Mjoff_diff;				\
    real2 Mjtmp = *Mjptr;				\
    Kijptr += Kijoff_diff;				\
    real8x2 Kijtmp = *Kijptr;				\
    Lij.a += Kijtmp.aa * Mjtmp.x + Kijtmp.ab * Mjtmp.y;	\
    Lij.b += Kijtmp.ba * Mjtmp.x + Kijtmp.bb * Mjtmp.y;	\
    Lij.c += Kijtmp.ca * Mjtmp.x + Kijtmp.cb * Mjtmp.y;	\
    Lij.d += Kijtmp.da * Mjtmp.x + Kijtmp.db * Mjtmp.y;	\
    Lij.e += Kijtmp.ea * Mjtmp.x + Kijtmp.eb * Mjtmp.y;	\
    Lij.f += Kijtmp.fa * Mjtmp.x + Kijtmp.fb * Mjtmp.y;	\
    Lij.g += Kijtmp.ga * Mjtmp.x + Kijtmp.gb * Mjtmp.y;	\
    Lij.h += Kijtmp.ha * Mjtmp.x + Kijtmp.hb * Mjtmp.y;	\
  }
#endif
#if(0) // intended to execute MAD and MUL at the same time; same as the original
#define COMP(Kijoff_diff, Mjoff_diff)				\
  {								\
    Mjptr += Mjoff_diff;					\
    real2 Mjtmp = *Mjptr;					\
    Kijptr += Kijoff_diff;					\
    real8x2 Kijtmp = *Kijptr;					\
    real tmp;							\
    Lij.a += Kijtmp.aa * Mjtmp.x; tmp = Kijtmp.ab * Mjtmp.y;	\
    Lij.a += tmp;						\
    Lij.b += Kijtmp.ba * Mjtmp.x; tmp = Kijtmp.bb * Mjtmp.y;	\
    Lij.b += tmp;						\
    Lij.c += Kijtmp.ca * Mjtmp.x; tmp = Kijtmp.cb * Mjtmp.y;	\
    Lij.c += tmp;						\
    Lij.d += Kijtmp.da * Mjtmp.x; tmp = Kijtmp.db * Mjtmp.y;	\
    Lij.d += tmp;						\
    Lij.e += Kijtmp.ea * Mjtmp.x; tmp = Kijtmp.eb * Mjtmp.y;	\
    Lij.e += tmp;						\
    Lij.f += Kijtmp.fa * Mjtmp.x; tmp = Kijtmp.fb * Mjtmp.y;	\
    Lij.f += tmp;						\
    Lij.g += Kijtmp.ga * Mjtmp.x; tmp = Kijtmp.gb * Mjtmp.y;	\
    Lij.g += tmp;						\
    Lij.h += Kijtmp.ha * Mjtmp.x; tmp = Kijtmp.hb * Mjtmp.y;	\
    Lij.h += tmp;						\
  }
#endif
#if(0) // original (ptx has 16 FMA)
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real2 Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real8x2 Kijtmp = *Kijptr;			\
    Lij.a += Kijtmp.aa * Mjtmp.x;		\
    Lij.a += Kijtmp.ab * Mjtmp.y;		\
    Lij.b += Kijtmp.ba * Mjtmp.x;		\
    Lij.b += Kijtmp.bb * Mjtmp.y;		\
    Lij.c += Kijtmp.ca * Mjtmp.x;		\
    Lij.c += Kijtmp.cb * Mjtmp.y;		\
    Lij.d += Kijtmp.da * Mjtmp.x;		\
    Lij.d += Kijtmp.db * Mjtmp.y;		\
    Lij.e += Kijtmp.ea * Mjtmp.x;		\
    Lij.e += Kijtmp.eb * Mjtmp.y;		\
    Lij.f += Kijtmp.fa * Mjtmp.x;		\
    Lij.f += Kijtmp.fb * Mjtmp.y;		\
    Lij.g += Kijtmp.ga * Mjtmp.x;		\
    Lij.g += Kijtmp.gb * Mjtmp.y;		\
    Lij.h += Kijtmp.ha * Mjtmp.x;		\
    Lij.h += Kijtmp.hb * Mjtmp.y;		\
  }
#endif
#elif defined(K_IS_4X4)
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real4 Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real4x4 Kijtmp = *Kijptr;			\
    Lij.x += Kijtmp.xx * Mjtmp.x;		\
    Lij.x += Kijtmp.xy * Mjtmp.y;		\
    Lij.x += Kijtmp.xz * Mjtmp.z;		\
    Lij.x += Kijtmp.xw * Mjtmp.w;		\
    Lij.y += Kijtmp.yx * Mjtmp.x;		\
    Lij.y += Kijtmp.yy * Mjtmp.y;		\
    Lij.y += Kijtmp.yz * Mjtmp.z;		\
    Lij.y += Kijtmp.yw * Mjtmp.w;		\
    Lij.z += Kijtmp.zx * Mjtmp.x;		\
    Lij.z += Kijtmp.zy * Mjtmp.y;		\
    Lij.z += Kijtmp.zz * Mjtmp.z;		\
    Lij.z += Kijtmp.zw * Mjtmp.w;		\
    Lij.w += Kijtmp.wx * Mjtmp.x;		\
    Lij.w += Kijtmp.wy * Mjtmp.y;		\
    Lij.w += Kijtmp.wz * Mjtmp.z;		\
    Lij.w += Kijtmp.ww * Mjtmp.w;		\
  }
#elif defined(K_IS_4X2)
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real2 Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real4x2 Kijtmp = *Kijptr;			\
    Lij.x += Kijtmp.xx * Mjtmp.x;		\
    Lij.x += Kijtmp.xy * Mjtmp.y;		\
    Lij.y += Kijtmp.yx * Mjtmp.x;		\
    Lij.y += Kijtmp.yy * Mjtmp.y;		\
    Lij.z += Kijtmp.zx * Mjtmp.x;		\
    Lij.z += Kijtmp.zy * Mjtmp.y;		\
    Lij.w += Kijtmp.wx * Mjtmp.x;		\
    Lij.w += Kijtmp.wy * Mjtmp.y;		\
  }
#elif defined(K_IS_1X1)
#define COMP(Kijoff_diff, Mjoff_diff)		\
  {						\
    Mjptr += Mjoff_diff;			\
    real Mjtmp = *Mjptr;			\
    Kijptr += Kijoff_diff;			\
    real Kijtmp = *Kijptr;			\
    Lij += Kijtmp * Mjtmp;			\
  }
#endif
/* Created by aux_scuda45I.c */
#define B4_COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define B4_COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)


#if defined(K_IS_1X2_TEXTURE)
/* Declare the global variable of type texture to use float2 CUDA
   array (texture) that contains 316 K-matrices */
texture<float2, 3, cudaReadModeElementType> texRefK;
#elif defined(K_IS_4X1_TEXTURE)
/* Declare the global variable of type texture to use float4 CUDA
   array (texture) that contains 316 K-matrices */
texture<float4, 3, cudaReadModeElementType> texRefK;
#endif


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r256b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X2)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8x2 *K, real2 *M, int level, int Mstart)
#elif defined(K_IS_4X4)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4x4 *K, real4 *M, int level, int Mstart)
#elif defined(K_IS_4X2)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4x2 *K, real2 *M, int level, int Mstart)
#elif defined(K_IS_1X1)
__global__ void m2l_kern_ij_blocking_r256b4(real *L, real *K, real *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set the pointer to M */
#if defined(K_IS_4X4)
  real4 *Mptr;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_4X4)
    /* M[level][j4=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 4) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart       + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_4X4)
  for (int j4 = 0; j4 < CUTOFF_H / 4; j4 ++) { // unrolling 4x
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  for (int j2 = 0; j2 < CUTOFF_H / 2; j2 ++) { // unrolling 2x
#else
  for (int j = 0; j < CUTOFF_H; j ++) { // no unrolling
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_4X4)
    __shared__ real4 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real  Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_4X4)
	real4 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real  *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to next column(s) */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
    /* Set the row index and the pointer to L (L[chunk][row=NROWS*by][cell=id]) */
    int i = NROWS_H * by;
    real *Lptr = L + (((bx << LOG_CUTOFF_H) + i) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    /* Set the row index and the pointer to L (L[chunk][row=NROWS*by/4][cell=id]) */
    int i4 = ((NROWS_H * by) >> 2);
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + i4) << 9) + id;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set the row index and the pointer to L (L[chunk][row=NROWS*by/8][cell=id]) */
    int i8 = ((NROWS_H * by) >> 3);
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_H - 3)) + i8) << 9) + id;
#elif defined(K_IS_16X1)
    /* Set the row index and the pointer to L (L[chunk][row=NROWS*by/16][cell=id]) */
    int i16 = ((NROWS_H * by) >> 4);
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_H - 4)) + i16) << 9) + id;
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_H != 1)
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
    for (int iloc = 0; iloc < NROWS_H; iloc ++) // no unrolling
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    for (int iloc = 0; iloc < NROWS_H; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    for (int iloc = 0; iloc < NROWS_H; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 16) // unrolling 16x
#endif
#endif
    {
      /* Load Kij[z][y][x] */
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316];
      if (id < 316) Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
      __shared__ real4 Kij[316];
      if (id < 316) Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
      __shared__ real4 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_H / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_H / 16) + i16) * 316 + id);
#elif defined(K_IS_8X2)
      __shared__ real8x2 Kij[316];
      if (id < 316) Kij[id] = *(K + (j2 * (CUTOFF_H / 8) + i8) * 316 + id);
#elif defined(K_IS_4X4)
      __shared__ real4x4 Kij[316];
      if (id < 316) Kij[id] = *(K + (j4 * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_4X2)
      __shared__ real4x2 Kij[316];
      if (id < 316) Kij[id] = *(K + (j2 * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_1X1)
      __shared__ real Kij[316];
      if (id < 316) Kij[id] = *(K + (j * CUTOFF_H + i) * 316 + id);
#endif

      /* Advance row index */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      i4 ++;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      i8 ++;
#elif defined(K_IS_16X1)
      i16 ++;
#endif

      /* Ensure that Kij (and Mj if iloc=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X2)
      real8x2 *Kijptr = (real8x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X4)
      real4x4 *Kijptr = (real4x4 *)Kij;
      real4 *Mjptr = (real4 *)Mj + Mjoff;
#elif defined(K_IS_4X2)
      real4x2 *Kijptr = (real4x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_1X1)
      real *Kijptr = (real *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
      if (tz == 0) {
	B4_COMPXYZ0();
      }	else if (tz == 1) {
	B4_COMPXYZ1();
      }	else if (tz == 2) {
	B4_COMPXYZ2();
      }	else if (tz == 3) {
	B4_COMPXYZ3();
      }	else if (tz == 4) {
	B4_COMPXYZ4();
      }	else if (tz == 5) {
	B4_COMPXYZ5();
      }	else if (tz == 6) {
	B4_COMPXYZ6();
      }	else if (tz == 7) {
	B4_COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr to next row(s) */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if iloc is the last) is no longer
	 used */
      __syncthreads();

    } /* i */
  } /* j */
}


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r32b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X2)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8x2 *K, real2 *M, int level, int Mstart)
#elif defined(K_IS_4X4)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4x4 *K, real4 *M, int level, int Mstart)
#elif defined(K_IS_4X2)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4x2 *K, real2 *M, int level, int Mstart)
#elif defined(K_IS_1X1)
__global__ void m2l_kern_ij_blocking_r32b4(real *L, real *K, real *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set the pointer to M */
#if defined(K_IS_4X4)
  real4 *Mptr;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_4X4)
    /* M[level][j4=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 4) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart       + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_4X4)
  for (int j4 = 0; j4 < CUTOFF_L / 4; j4 ++) { // unrolling 4x
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  for (int j2 = 0; j2 < CUTOFF_L / 2; j2 ++) { // unrolling 2x
#else
  for (int j = 0; j < CUTOFF_L; j ++) { // no unrolling
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_4X4)
    __shared__ real4 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real  Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_4X4)
	real4 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real  *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to next column(s) */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
    /* Set the row index and the pointer to L (L[chunk][row=NROWS*by][cell=id]) */
    int i = NROWS_L * by;
    real *Lptr = L + (((bx << LOG_CUTOFF_L) + i) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    /* Set the row index and the pointer to L (L[chunk][row=NROWS*by/4][cell=id]) */
    int i4 = ((NROWS_L * by) >> 2);
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + i4) << 9) + id;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set the row index and the pointer to L (L[chunk][row=NROWS*by/8][cell=id]) */
    int i8 = ((NROWS_L * by) >> 3);
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_L - 3)) + i8) << 9) + id;
#elif defined(K_IS_16X1)
    /* Set the row index and the pointer to L (L[chunk][row=NROWS*by/16][cell=id]) */
    int i16 = ((NROWS_L * by) >> 4);
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_L - 4)) + i16) << 9) + id;
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_L != 1)
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
    for (int iloc = 0; iloc < NROWS_L; iloc ++) // no unrolling
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    for (int iloc = 0; iloc < NROWS_L; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    for (int iloc = 0; iloc < NROWS_L; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 16) // unrolling 16x
#endif
#endif
    {
      /* Load Kij[z][y][x] */
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316];
      if (id < 316) Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
      __shared__ real4 Kij[316];
      if (id < 316) Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
      __shared__ real4 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_L / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_L / 16) + i16) * 316 + id);
#elif defined(K_IS_8X2)
      __shared__ real8x2 Kij[316];
      if (id < 316) Kij[id] = *(K + (j2 * (CUTOFF_L / 8) + i8) * 316 + id);
#elif defined(K_IS_4X4)
      __shared__ real4x4 Kij[316];
      if (id < 316) Kij[id] = *(K + (j4 * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_4X2)
      __shared__ real4x2 Kij[316];
      if (id < 316) Kij[id] = *(K + (j2 * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_1X1)
      __shared__ real Kij[316];
      if (id < 316) Kij[id] = *(K + (j * CUTOFF_L + i) * 316 + id);
#endif

      /* Advance row index */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      i4 ++;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      i8 ++;
#elif defined(K_IS_16X1)
      i16 ++;
#endif

      /* Ensure that Kij (and Mj if iloc=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X2)
      real8x2 *Kijptr = (real8x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X4)
      real4x4 *Kijptr = (real4x4 *)Kij;
      real4 *Mjptr = (real4 *)Mj + Mjoff;
#elif defined(K_IS_4X2)
      real4x2 *Kijptr = (real4x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_1X1)
      real *Kijptr = (real *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
      if (tz == 0) {
	B4_COMPXYZ0();
      }	else if (tz == 1) {
	B4_COMPXYZ1();
      }	else if (tz == 2) {
	B4_COMPXYZ2();
      }	else if (tz == 3) {
        B4_COMPXYZ3();
      }	else if (tz == 4) {
	B4_COMPXYZ4();
      }	else if (tz == 5) {
	B4_COMPXYZ5();
      }	else if (tz == 6) {
	B4_COMPXYZ6();
      }	else if (tz == 7) {
	B4_COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr to next row(s) */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if iloc is the last) is no longer
	 used */
      __syncthreads();

    } /* i */
  } /* j */
}


#if defined(USE_ANY_R)

__global__ void m2l_kern_ij_blocking_b4(int r, real *L, real *K, real *M, int level, int Mstart)
{
  /* Read the index of the underlying level */
  int lev = level;
  
  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2
  
  /* Set the pointer to M */
  real *Mptr;
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*B), 0<=cy<2^l/(2*B), 0<=cz<2^l/(2*B) */
    int u = POW2(lev - 1) / 4; // 2^l/(2*B)
    int cx = bx % u;
    int cy = (bx % (u * u)) / u;
    int cz = bx / (u * u);
    
    /* M[level][j=0][sib=tz][cell=(B*cx,B*cy,B*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (4 * cz)) * ncpec + (4 * cy)) * ncpec + (4 * cx);
  }
  
  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = tx % 4;  // tx%B
    int hy = ty;      // ty
    int hz = tx / 4;  // tx/B
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(B+2)*(hy+(B+2)*hz)
  }
  
  /* Compute the unique cell index */
  int id = tx + 16 * (ty + 4 * tz); // tx+B*B*(ty+B*tz)
  
  /* Loop over columns j */
  for (int j = 0; j < r; j ++) { // no unrolling
    
    /* Load Mj of (2*B+4)^3 source cells in/around this chunk. Those
       cells are classified by their sibling-indice. */
    __shared__ real  Mj[8][6][6][6]; // Mj[8][B+2][B+2][B+2]
    
    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
        Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to next column */
    Mptr += 8 * ncpec * ncpec * ncpec;
    
    /* Number of rows per row-group */
    int nrows = r % gridDim.y == 0 ? r / gridDim.y : r / gridDim.y + 1;
    
    /* Loop over rows */
    for (int  i = nrows * by; i < MIN(nrows * (by + 1), r); i ++) {

      /* Load Kij */
      __shared__ real Kij[316];
      if (id < 316) Kij[id] = *(K + (j * r + i) * 316 + id);
      
      /* Ensure that Kij (and Mj if i is the first) was loaded */
      __syncthreads();
      
      /* Initialise Lij(F) */
      real Lij = ZERO;
      
      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
      real *Kijptr = (real *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
      
      if (tz == 0) {
	B4_COMPXYZ0();
      }	else if (tz == 1) {
	B4_COMPXYZ1();
      }	else if (tz == 2) {
	B4_COMPXYZ2();
      }	else if (tz == 3) {
	B4_COMPXYZ3();
      }	else if (tz == 4) {
	B4_COMPXYZ4();
      }	else if (tz == 5) {
	B4_COMPXYZ5();
      }	else if (tz == 6) {
	B4_COMPXYZ6();
      }	else if (tz == 7) {
	B4_COMPXYZ7();
      }
      
      /* Set the pointer to Li(F) */
      real *Lptr = L + (bx * r + i) * 512 + id; // 8*B^3
      
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
      *Lptr += Lij;
      
      /* Ensure that Kij (and Mj if i is the last) is no longer
	 used */
      __syncthreads();
      
    } /* i */
  } /* j */
}

#define B2_COMPXYZ0() COMP(57, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ1() COMP(8, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ2() COMP(50, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ3() COMP(1, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 2); COMP(1, 256); COMP(2, -382); COMP(1, 2); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ4() COMP(56, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ5() COMP(7, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ6() COMP(49, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)
#define B2_COMPXYZ7() COMP(0, 0); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -442); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, 2); COMP(2, -382); COMP(1, 256); COMP(1, 2); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(9, -330); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -382); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(2, -130); COMP(1, 256); COMP(1, -255); COMP(1, 256); COMP(1, -255); COMP(1, 256)

__global__ void m2l_kern_ij_blocking_b2(int r, real *L, real *K, real *M, int level, int Mstart)
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set the pointer to M */
  real *Mptr;
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*B), 0<=cy<2^l/(2*B), 0<=cz<2^l/(2*B) */
    int u = POW2(lev - 1) / 2; // 2^l/(2*B)
    int cx = bx % u;
    int cy = (bx % (u * u)) / u;
    int cz = bx / (u * u);

    /* M[level][j=0][sib=tz][cell=(B*cx,B*cy,B*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (2 * cz)) * ncpec + (2 * cy)) * ncpec + (2 * cx);
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = tx % 2;
    int hy = ty;
    int hz = tx / 2;
    Mjoff = hx + 4 * (hy + 4 * hz); // hx+(B+2)*(hy+(B+2)*hz)
  }

  /* Compute the unique cell index */
  int id = tx + 4 * (ty + 2 * tz); // tx+B*B*(ty+B*tz)<64
  
  /* Loop over columns j */
  for (int j = 0; j < r; j ++) { // no unrolling
    
    /* Load Mj of (2*B+4)^3 source cells in/around this chunk. Those
       cells are classified by their sibling-indices */
    __shared__ real  Mj[8][4][4][4]; // Mj[8][B+2][B+2][B+2]

    for (int iz = 0; iz < 4; iz ++) { // B+2
      Mj[tz][iz][ty    ][tx] = *(Mptr + (iz * ncpec + (ty    )) * ncpec + tx); // tx<B*B(=B+2), ty<B
      Mj[tz][iz][ty + 2][tx] = *(Mptr + (iz * ncpec + (ty + 2)) * ncpec + tx); // tx<B*B(=B+2), ty<B
    }
    
    /* Advance Mptr to next column */
    Mptr += 8 * ncpec * ncpec * ncpec;

    /* Number of rows per row-group */
    int nrows = r % gridDim.y == 0 ? r / gridDim.y : r / gridDim.y + 1;

    /* Loop over rows */
    for (int  i = nrows * by; i < MIN(nrows * (by + 1), r); i ++) {

      /* Load Kij */
      __shared__ real Kij[316];
      {
	real *Kptr = K + (j * r + i) * 316 + id;
	Kij[id      ] = *Kptr; Kptr += 64; // 8*B^3=64
	Kij[id +  64] = *Kptr; Kptr += 64;
	Kij[id + 128] = *Kptr; Kptr += 64;
	Kij[id + 192] = *Kptr; Kptr += 64;
	if (id + 256 < 316) Kij[id + 256] = *Kptr;
      }
      
      /* Ensure that Kij (and Mj if i is the first) was loaded */
      __syncthreads();
      
      /* Initialise Lij(F) */
      real Lij = ZERO;
      
      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
      real *Kijptr = (real *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
      
      if (tz == 0) {
	B2_COMPXYZ0();
      }	else if (tz == 1) {
	B2_COMPXYZ1();
      }	else if (tz == 2) {
	B2_COMPXYZ2();
      }	else if (tz == 3) {
        B2_COMPXYZ3();
      }	else if (tz == 4) {
	B2_COMPXYZ4();
      }	else if (tz == 5) {
        B2_COMPXYZ5();
      }	else if (tz == 6) {
	B2_COMPXYZ6();
      }	else if (tz == 7) {
	B2_COMPXYZ7();
      }
      
      /* Set the pointer to Li(F) */
      real *Lptr = L + (bx * r + i) * 64 + id; // 8*B^3
      
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
      *Lptr += Lij;
      
      /* Ensure that Kij (and Mj if i is the last) is no longer
	 used */
      __syncthreads();
      
    } /* i */
  } /* j */
}
#endif /* defined(USE_ANY_R) */
/**************************************************************************/
#elif defined(CUDA_VER45H)
/**************************************************************************/
/* Based on VER45H */

#include "real.h"

/* Dx, Dy, and Dz denote the size of chunk. Namely, each chunk
   consists of Dx*Dy*Dz clusters. In this code, Dx=Dy=Dz=4 is assumed
   and this number corresponds to 'B' in the paper and manual */

#define bx blockIdx.x   // chunk index
#define by blockIdx.y   // row-group index

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

/* cutoff stands for the dimension of M-vector, L-vector, and
   K-matrix. This corresponds to 'r' in the paper and manual.  In this
   code, r is either 256 (high-precision version) or 32 (low-precision
   version) */
#define CUTOFF_H     256
#define LOG_CUTOFF_H   8
#define CUTOFF_L      32
#define LOG_CUTOFF_L   5

/* Set the number of rows per row-group. This parameter corresponds to
   'P' in the paper and manual */
#if !defined(NUM_ROW_GROUPS_IJ)
#define NUM_ROW_GROUPS_IJ 8 // 8 is better for C2050+SDK3.2
#endif
#if (NUM_ROW_GROUPS_IJ == 1)
#define NROWS_H 256 // cutoff=256
#define NROWS_L  32 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 2)
#define NROWS_H 128 // cutoff=256
#define NROWS_L  16 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 4)
#define NROWS_H  64 // cutoff=256
#define NROWS_L   8 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 8)
#define NROWS_H  32 // cutoff=256
#define NROWS_L   4 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 16)
#define NROWS_H  16 // cutoff=256
#define NROWS_L   2 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 32)
#define NROWS_H   8 // cutoff=256
#define NROWS_L   1 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 64)
#define NROWS_H   4 // cutoff=256
#define NROWS_L   0 // cutoff=32  IMPOSSIBLE
#else
#error Unsupposed NUM_ROW_GROUPS_IJ.
#endif


//////////////////////////////////////////
#if (CUDA_ARCH >= 20) && (CUDA_ARCH < 30)
//////////////////////////////////////////

#if !defined(K_IS_4X2)
#error Set an appropriate macro.
#endif

/* 0<=tx<2*Dx*Dz, 0<=ty<Dy, and 0<=tz<4, where tz denotes the first or
   second half of eight sibling-indexes */

/* Macros to perform Li+=Kij*Mj for all the 189 Kij */
#if defined(K_IS_4X2)
#define COMP(Kijoff_diff, Mjoff_diff)					\
  {									\
    Mjptr += Mjoff_diff;						\
    real Mjtmp = *Mjptr;						\
    Kijptr += Kijoff_diff;						\
    real4 Kijtmp = *Kijptr;						\
    Lij[id].x += Kijtmp.x * Mjtmp;					\
    Lij[id].y += Kijtmp.y * Mjtmp;					\
    Lij[id].z += Kijtmp.z * Mjtmp;					\
    Lij[id].w += Kijtmp.w * Mjtmp;					\
  }
#endif
/* Created by aux_scuda45H.c */
#define COMPXYZ0() COMP(114, 0); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728)
#define COMPXYZ1() COMP(16, 0); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728)
#define COMPXYZ2() COMP(100, 0); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728)
#define COMPXYZ3() COMP(2, 0); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 4); COMP(2, 1728); COMP(4, -2584); COMP(2, 4); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728)
#define COMPXYZ4() COMP(112, 0); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728)
#define COMPXYZ5() COMP(14, 0); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728)
#define COMPXYZ6() COMP(98, 0); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728)
#define COMPXYZ7() COMP(0, 0); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2980); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, 4); COMP(4, -2584); COMP(2, 1728); COMP(2, 4); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(18, -2188); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -2584); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(4, -868); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728); COMP(2, -1726); COMP(2, 1728)


#if defined(K_IS_4X2)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4x2 *K, real2 *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_4X2)
  real2 *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of this field chunk, to
       which the bx-th thread-block is assigned, where
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_4X2)
    /* M[level][j2=0][sib=0][cell=(B*cx,B*cy,B*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + 0) * ncpec + (4 * cz)) * ncpec + (4 * cy)) * ncpec + (4 * cx);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int j  = tx / 16;        // 0<=j<2
    int hx = (tx % 16) % 4;  // 0<=hx<4
    int hy = ty;             // 0<=hy<4
    int hz = (tx % 16) / 4;  // 0<=hz<4
    Mjoff = j + 2 * (hx + 6 * (hy + 6 * hz));
  }

  /* Compute the unique cell index */
  int id = tx + 32 * (ty + 4 * tz); // 0<=id<8*B^3
  
  /* Loop over columns j */
#if defined(K_IS_4X2)
  for (int j2 = 0; j2 < CUTOFF_H / 2; j2 ++) { // unrolling 2x
#endif

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk. Those
       cells are classified by their sibling-indices */
#if defined(K_IS_4X2)
    __shared__ real Mj[8][6][6][6][2]; // Mj[sibling-index][B+2][B+2][B+2][j or j+1]
#endif
    {
      int px = id % 16;         // 0<=px<16
      int py = (id % 64) / 16;  // 0<=py<4
      int pz = id / 64;         // 0<=pz<8; sibling-index

      int ncpec2 = ncpec * ncpec;
      int wid = (py >> 1);         // 0, 0, 1, 1 for py=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for py=0, 1, 2, 3
      int ysta = 3 * py - 6 * wid; // 0, 3, 0, 3 for py=0, 1, 2, 3
      if (px < 6) {
#if defined(K_IS_4X2)
	real2 *ptmp = Mptr + ((pz * ncpec + zsta) * ncpec + ysta) * ncpec + px;
	real2 Mtmp;
#endif
	for (int l = 0; l < 3; l ++) {
	  Mtmp = *ptmp;
	  Mj[pz][l + zsta][0 + ysta][px][0] = Mtmp.x;
	  Mj[pz][l + zsta][0 + ysta][px][1] = Mtmp.y;
	  Mtmp = *(ptmp + ncpec);
	  Mj[pz][l + zsta][1 + ysta][px][0] = Mtmp.x;
	  Mj[pz][l + zsta][1 + ysta][px][1] = Mtmp.y;
	  Mtmp = *(ptmp + ncpec * 2);
	  Mj[pz][l + zsta][2 + ysta][px][0] = Mtmp.x;
	  Mj[pz][l + zsta][2 + ysta][px][1] = Mtmp.y;
	  ptmp += ncpec2;
	}
      }
    }
    
    /* Advance Mptr to next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_4X2)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by/4][cell=id] for unrolling 4x */
    int i4 = ((NROWS_H * by) >> 2);
    real4 *Lptr;
    if (tx < 16) { // only first half-warp; 0<=tx<16
      Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + i4) << 9) + (tx + 16 * (ty + 4 * tz)); // not 32, but 16
    }
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_H != 1)
#if defined(K_IS_4X2)
    for (int iloc = 0; iloc < NROWS_H; iloc += 4) // unrolling 4x
#endif
#endif
    {
      /* Load Kij[z][y][x] */
#if defined(K_IS_4X2)
      __shared__ real4 Kij[316][2];
      if (id < 316) {
	real4x2 tmp = *(K + (j2 * (CUTOFF_H / 4) + i4) * 316 + id);
	Kij[id][0].x = tmp.xx;
	Kij[id][1].x = tmp.xy;
	Kij[id][0].y = tmp.yx;
	Kij[id][1].y = tmp.yy;
	Kij[id][0].z = tmp.zx;
	Kij[id][1].z = tmp.zy;
	Kij[id][0].w = tmp.wx;
	Kij[id][1].w = tmp.wy;
      }
#endif

      /* Ensure that Kij (and Mj if iloc=0) was loaded */
      __syncthreads();

      /* Allocate Lij and Li,j+1 of 256 Fs */
#if defined(K_IS_4X2)
      __shared__ real4 Lij[512];
#endif

      /*
	256 Fs with sibling-indexes 0 to 3
      */

      /* Initialise Lij(F) */
#if defined(K_IS_4X2)
      Lij[id] = make_real4(ZERO, ZERO, ZERO, ZERO);
#endif
      
      /* Compute Lij(F)+=Kij(F,S)*Mj(S) (reduction for S) */
      {
#if defined(K_IS_4X2)
	real4 *Kijptr = (real4 *)Kij + (tx / 16);
	real *Mjptr = (real *)Mj + Mjoff;
#endif
	if (tz == 0) {
	  COMPXYZ0();
	} else if (tz == 1) {
	  COMPXYZ1();
	} else if (tz == 2) {
	  COMPXYZ2();
	} else if (tz == 3) {
	  COMPXYZ3();
	}
      }

      /* Ensure that Lij is computed */
      __syncthreads();
      
      /* Sum of j and j+1 and store */
      if (tx < 16) { // Let the first half-warp compute the sum
#if defined(K_IS_4X2)
	Lptr->x += Lij[id].x + Lij[id + 16].x;
	Lptr->y += Lij[id].y + Lij[id + 16].y;
	Lptr->z += Lij[id].z + Lij[id + 16].z;
	Lptr->w += Lij[id].w + Lij[id + 16].w;
#endif
      }

      /* Ensure that Lij may be overwritten */
      __syncthreads();

      /*
	256 Fs with sibling-indexes 4 to 7
      */

      /* Initialise Lij(F) */
#if defined(K_IS_4X2)
      Lij[id] = make_real4(ZERO, ZERO, ZERO, ZERO);
#endif
      
      /* Compute Lij(F)+=Kij(F,S)*Mj(S) (reduction for S) */
      {
#if defined(K_IS_4X2)
	real4 *Kijptr = (real4 *)Kij + (tx / 16);
	real *Mjptr = (real *)Mj + Mjoff;
#endif
	if (tz == 0) {
	  COMPXYZ4();
	} else if (tz == 1) {
	  COMPXYZ5();
	} else if (tz == 2) {
	  COMPXYZ6();
	} else if (tz == 3) {
	  COMPXYZ7();
	}
      }

      /* Ensure that Lij is computed */
      __syncthreads();
      
      /* Sum of j and j+1 and store */
      if (tx < 16) { // Let the first half-warp compute the sum
#if defined(K_IS_4X2)
	(Lptr + 256)->x += Lij[id].x + Lij[id + 16].x;
	(Lptr + 256)->y += Lij[id].y + Lij[id + 16].y;
	(Lptr + 256)->z += Lij[id].z + Lij[id + 16].z;
	(Lptr + 256)->w += Lij[id].w + Lij[id + 16].w;

	/* Advance Lptr to next i */
	Lptr += 512; // 8*B^3
#endif
      }

      /* advance row index to next i */
#if defined(K_IS_4X2)
      i4 ++;
#endif

      //      /* Advance Lptr to next i */
      //      Lptr += 512; // 8*B^3
      
      /* Ensure that Kij (and Mj if iloc is the last) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}

#if defined(K_IS_4X2)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4x2 *K, real2 *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_4X2)
  real2 *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of this field chunk, to
       which the bx-th thread-block is assigned, where
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_4X2)
    /* M[level][j2=0][sib=0][cell=(B*cx,B*cy,B*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + 0) * ncpec + (4 * cz)) * ncpec + (4 * cy)) * ncpec + (4 * cx);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int j  = tx / 16;        // 0<=j<2
    int hx = (tx % 16) % 4;  // 0<=hx<4
    int hy = ty;             // 0<=hy<4
    int hz = (tx % 16) / 4;  // 0<=hz<4
    Mjoff = j + 2 * (hx + 6 * (hy + 6 * hz));
  }

  /* Compute the unique cell index */
  int id = tx + 32 * (ty + 4 * tz); // 0<=id<8*B^3
  
  /* Loop over columns j */
#if defined(K_IS_4X2)
  for (int j2 = 0; j2 < CUTOFF_L / 2; j2 ++) { // unrolling 2x
#endif

    /* Load Mj of (2*B+4)^3 source cells in/around this chunk. Those
       cells are classified by their sibling-indices */
#if defined(K_IS_4X2)
    __shared__ real Mj[8][6][6][6][2]; // Mj[sibling-index][B+2][B+2][B+2][j or j+1]
#endif
    {
      int px = id % 16;         // 0<=px<16
      int py = (id % 64) / 16;  // 0<=py<4
      int pz = id / 64;         // 0<=pz<8; sibling-index

      int ncpec2 = ncpec * ncpec;
      int wid = (py >> 1);         // 0, 0, 1, 1 for py=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for py=0, 1, 2, 3
      int ysta = 3 * py - 6 * wid; // 0, 3, 0, 3 for py=0, 1, 2, 3
      if (px < 6) {
#if defined(K_IS_4X2)
	real2 *ptmp = Mptr + ((pz * ncpec + zsta) * ncpec + ysta) * ncpec + px;
	real2 Mtmp;
#endif
	for (int l = 0; l < 3; l ++) {
	  Mtmp = *ptmp;
	  Mj[pz][l + zsta][0 + ysta][px][0] = Mtmp.x;
	  Mj[pz][l + zsta][0 + ysta][px][1] = Mtmp.y;
	  Mtmp = *(ptmp + ncpec);
	  Mj[pz][l + zsta][1 + ysta][px][0] = Mtmp.x;
	  Mj[pz][l + zsta][1 + ysta][px][1] = Mtmp.y;
	  Mtmp = *(ptmp + ncpec * 2);
	  Mj[pz][l + zsta][2 + ysta][px][0] = Mtmp.x;
	  Mj[pz][l + zsta][2 + ysta][px][1] = Mtmp.y;
	  ptmp += ncpec2;
	}
      }
    }
    
    /* Advance Mptr to next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_4X2)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by/4][cell=id] for unrolling 4x */
    int i4 = ((NROWS_L * by) >> 2);
    real4 *Lptr;
    if (tx < 16) { // only first half-warp; 0<=tx<16
      Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + i4) << 9) + (tx + 16 * (ty + 4 * tz)); // not 32, but 16
    }
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_L != 1)
#if defined(K_IS_4X2)
    for (int iloc = 0; iloc < NROWS_L; iloc += 4) // unrolling 4x
#endif
#endif
    {
      /* Load Kij[z][y][x] */
#if defined(K_IS_4X2)
      __shared__ real4 Kij[316][2];
      if (id < 316) {
	real4x2 tmp = *(K + (j2 * (CUTOFF_L / 4) + i4) * 316 + id);
	Kij[id][0].x = tmp.xx;
	Kij[id][1].x = tmp.xy;
	Kij[id][0].y = tmp.yx;
	Kij[id][1].y = tmp.yy;
	Kij[id][0].z = tmp.zx;
	Kij[id][1].z = tmp.zy;
	Kij[id][0].w = tmp.wx;
	Kij[id][1].w = tmp.wy;
      }
#endif

      /* Ensure that Kij (and Mj if iloc=0) was loaded */
      __syncthreads();

      /* Allocate Lij and Li,j+1 of 256 Fs */
#if defined(K_IS_4X2)
      __shared__ real4 Lij[512];
#endif

      /*
	256 Fs with sibling-indexes 0 to 3
      */

      /* Initialise Lij(F) */
#if defined(K_IS_4X2)
      Lij[id] = make_real4(ZERO, ZERO, ZERO, ZERO);
#endif
      
      /* Compute Lij(F)+=Kij(F,S)*Mj(S) (reduction for S) */
      {
#if defined(K_IS_4X2)
	real4 *Kijptr = (real4 *)Kij + (tx / 16);
	real *Mjptr = (real *)Mj + Mjoff;
#endif
	if (tz == 0) {
	  COMPXYZ0();
	} else if (tz == 1) {
	  COMPXYZ1();
	} else if (tz == 2) {
	  COMPXYZ2();
	} else if (tz == 3) {
	  COMPXYZ3();
	}
      }

      /* Ensure that Lij is computed */
      __syncthreads();
      
      /* Sum of j and j+1 and store */
      if (tx < 16) { // Let the first half-warp compute the sum
#if defined(K_IS_4X2)
	Lptr->x += Lij[id].x + Lij[id + 16].x;
	Lptr->y += Lij[id].y + Lij[id + 16].y;
	Lptr->z += Lij[id].z + Lij[id + 16].z;
	Lptr->w += Lij[id].w + Lij[id + 16].w;
#endif
      }

      /* Ensure that Lij may be overwritten */
      __syncthreads();

      /*
	256 Fs with sibling-indexes 4 to 7
      */

      /* Initialise Lij(F) */
#if defined(K_IS_4X2)
      Lij[id] = make_real4(ZERO, ZERO, ZERO, ZERO);
#endif
      
      /* Compute Lij(F)+=Kij(F,S)*Mj(S) (reduction for S) */
      {
#if defined(K_IS_4X2)
	real4 *Kijptr = (real4 *)Kij + (tx / 16);
	real *Mjptr = (real *)Mj + Mjoff;
#endif
	if (tz == 0) {
	  COMPXYZ4();
	} else if (tz == 1) {
	  COMPXYZ5();
	} else if (tz == 2) {
	  COMPXYZ6();
	} else if (tz == 3) {
	  COMPXYZ7();
	}
      }

      /* Ensure that Lij is computed */
      __syncthreads();
      
      /* Sum of j and j+1 and store */
      if (tx < 16) { // Let the first half-warp compute the sum
#if defined(K_IS_4X2)
	(Lptr + 256)->x += Lij[id].x + Lij[id + 16].x;
	(Lptr + 256)->y += Lij[id].y + Lij[id + 16].y;
	(Lptr + 256)->z += Lij[id].z + Lij[id + 16].z;
	(Lptr + 256)->w += Lij[id].w + Lij[id + 16].w;

	/* Advance Lptr to next i */
	Lptr += 512; // 8*B^3
#endif
      }

      /* advance row index to next i */
#if defined(K_IS_4X2)
      i4 ++;
#endif

      //      /* Advance Lptr to next i */
      //      Lptr += 512; // 8*B^3
      
      /* Ensure that Kij (and Mj if iloc is the last) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}

//////////////////////////////////////////
#elif (CUDA_ARCH >= 10) && (CUDA_ARCH < 20)
//////////////////////////////////////////

#if !defined(K_IS_1X2_TEXTURE) && !defined(K_IS_4X1_TEXTURE) && !defined(K_IS_4X1) && !defined(K_IS_8X1) && !defined(K_IS_16X1) && !defined(K_IS_8X2) && !defined(K_IS_4X4) && !defined(K_IS_4X2)
#error Set an appropriate macro.
#endif

/* 0<=tx<2*Dx*Dz, 0<=ty<Dy, and 0<=tz<4, where tz denotes eight
   sibling-indexes */

/* Macros to perform Li+=Kij*Mj for all the 189 Kij */
#if defined(K_IS_1X2_TEXTURE)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij += Kijtmp.x * Mjtmp.x;				\
  Lij += Kijtmp.y * Mjtmp.y;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.x * Mjtmp;				\
  Lij.y += Kijtmp.y * Mjtmp;				\
  Lij.z += Kijtmp.z * Mjtmp;				\
  Lij.w += Kijtmp.w * Mjtmp
#elif defined(K_IS_8X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.a * Mjtmp;				\
  Lij.b += Kijtmp.b * Mjtmp;				\
  Lij.c += Kijtmp.c * Mjtmp;				\
  Lij.d += Kijtmp.d * Mjtmp;				\
  Lij.e += Kijtmp.e * Mjtmp;				\
  Lij.f += Kijtmp.f * Mjtmp;				\
  Lij.g += Kijtmp.g * Mjtmp;				\
  Lij.h += Kijtmp.h * Mjtmp
#elif defined(K_IS_16X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.a * Mjtmp;				\
  Lij.b += Kijtmp.b * Mjtmp;				\
  Lij.c += Kijtmp.c * Mjtmp;				\
  Lij.d += Kijtmp.d * Mjtmp;				\
  Lij.e += Kijtmp.e * Mjtmp;				\
  Lij.f += Kijtmp.f * Mjtmp;				\
  Lij.g += Kijtmp.g * Mjtmp;				\
  Lij.h += Kijtmp.h * Mjtmp;				\
  Lij.i += Kijtmp.i * Mjtmp;				\
  Lij.j += Kijtmp.j * Mjtmp;				\
  Lij.k += Kijtmp.k * Mjtmp;				\
  Lij.l += Kijtmp.l * Mjtmp;				\
  Lij.m += Kijtmp.m * Mjtmp;				\
  Lij.n += Kijtmp.n * Mjtmp;				\
  Lij.o += Kijtmp.o * Mjtmp;				\
  Lij.p += Kijtmp.p * Mjtmp
#elif defined(K_IS_8X2)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.aa * Mjtmp.x;				\
  Lij.a += Kijtmp.ab * Mjtmp.y;				\
  Lij.b += Kijtmp.ba * Mjtmp.x;				\
  Lij.b += Kijtmp.bb * Mjtmp.y;				\
  Lij.c += Kijtmp.ca * Mjtmp.x;				\
  Lij.c += Kijtmp.cb * Mjtmp.y;				\
  Lij.d += Kijtmp.da * Mjtmp.x;				\
  Lij.d += Kijtmp.db * Mjtmp.y;				\
  Lij.e += Kijtmp.ea * Mjtmp.x;				\
  Lij.e += Kijtmp.eb * Mjtmp.y;				\
  Lij.f += Kijtmp.fa * Mjtmp.x;				\
  Lij.f += Kijtmp.fb * Mjtmp.y;				\
  Lij.g += Kijtmp.ga * Mjtmp.x;				\
  Lij.g += Kijtmp.gb * Mjtmp.y;				\
  Lij.h += Kijtmp.ha * Mjtmp.x;				\
  Lij.h += Kijtmp.hb * Mjtmp.y
#elif defined(K_IS_4X4)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.xx * Mjtmp.x;				\
  Lij.x += Kijtmp.xy * Mjtmp.y;				\
  Lij.x += Kijtmp.xz * Mjtmp.z;				\
  Lij.x += Kijtmp.xw * Mjtmp.w;				\
  Lij.y += Kijtmp.yx * Mjtmp.x;				\
  Lij.y += Kijtmp.yy * Mjtmp.y;				\
  Lij.y += Kijtmp.yz * Mjtmp.z;				\
  Lij.y += Kijtmp.yw * Mjtmp.w;				\
  Lij.z += Kijtmp.zx * Mjtmp.x;				\
  Lij.z += Kijtmp.zy * Mjtmp.y;				\
  Lij.z += Kijtmp.zz * Mjtmp.z;				\
  Lij.z += Kijtmp.zw * Mjtmp.w;				\
  Lij.w += Kijtmp.wx * Mjtmp.x;				\
  Lij.w += Kijtmp.wy * Mjtmp.y;				\
  Lij.w += Kijtmp.wz * Mjtmp.z;				\
  Lij.w += Kijtmp.ww * Mjtmp.w
#elif defined(K_IS_4X2)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.xx * Mjtmp.x;				\
  Lij.x += Kijtmp.xy * Mjtmp.y;				\
  Lij.y += Kijtmp.yx * Mjtmp.x;				\
  Lij.y += Kijtmp.yy * Mjtmp.y;				\
  Lij.z += Kijtmp.zx * Mjtmp.x;				\
  Lij.z += Kijtmp.zy * Mjtmp.y;				\
  Lij.w += Kijtmp.wx * Mjtmp.x;				\
  Lij.w += Kijtmp.wy * Mjtmp.y
#endif
/* Created by aux_scuda38BH.c */
#define COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)

#if defined(K_IS_1X2_TEXTURE)
/* Declare the global variable of type texture to use float2 CUDA
   array (texture) that contains 316 K-matrices */
texture<float2, 3, cudaReadModeElementType> texRefK;
#elif defined(K_IS_4X1_TEXTURE)
/* Declare the global variable of type texture to use float4 CUDA
   array (texture) that contains 316 K-matrices */
texture<float4, 3, cudaReadModeElementType> texRefK;
#endif


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r256b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X2)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8x2 *K, real2 *M, int level, int Mstart)
#elif defined(K_IS_4X4)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4x4 *K, real4 *M, int level, int Mstart)
#elif defined(K_IS_4X2)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4x2 *K, real2 *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_4X4)
  real4 *Mptr;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_4X4)
    /* M[level][j4=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 4) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart       + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_4X4)
  for (int j4 = 0; j4 < CUTOFF_H / 4; j4 ++) { // unrolling 4x
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  for (int j2 = 0; j2 < CUTOFF_H / 2; j2 ++) { // unrolling 2x
#else
  for (int j = 0; j < CUTOFF_H; j ++) { // no unrolling
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_4X4)
    __shared__ real4 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real  Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif
    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_4X4)
	real4 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real  *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by][cell=id] for no unrolling */
    int i = NROWS_H * by;
    real *Lptr = L + (((bx << LOG_CUTOFF_H) + i) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by/4][cell=id] for unrolling 4x */
    int i4 = ((NROWS_H * by) >> 2);
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + i4) << 9) + id;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by/8][cell=id] for unrolling 8x */
    int i8 = ((NROWS_H * by) >> 3);
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_H - 3)) + i8) << 9) + id;
#elif defined(K_IS_16X1)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by/16][cell=id] for unrolling 16x */
    int i16 = ((NROWS_H * by) >> 4);
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_H - 4)) + i16) << 9) + id;
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_H != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_H; iloc ++) // no unrolling
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    for (int iloc = 0; iloc < NROWS_H; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    for (int iloc = 0; iloc < NROWS_H; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 16) // unrolling 16x
#endif
#endif
    {
      /* Load Kij[z][y][x] */
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316];
      if (id < 316) Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
      __shared__ real4 Kij[316];
      if (id < 316) Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
      __shared__ real4 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_H / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_H / 16) + i16) * 316 + id);
#elif defined(K_IS_8X2)
      __shared__ real8x2 Kij[316];
      if (id < 316) Kij[id] = *(K + (j2 * (CUTOFF_H / 8) + i8) * 316 + id);
#elif defined(K_IS_4X4)
      __shared__ real4x4 Kij[316];
      if (id < 316) Kij[id] = *(K + (j4 * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_4X2)
      __shared__ real4x2 Kij[316];
      if (id < 316) Kij[id] = *(K + (j2 * (CUTOFF_H / 4) + i4) * 316 + id);
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij, Kijtmp;
      real2 *Mjptr = (real2 *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij, Kijtmp;
      real *Mjptr = (real *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij, Kijtmp;
      real *Mjptr = (real *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij, Kijtmp;
      real *Mjptr = (real *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_8X2)
      real8x2 *Kijptr = (real8x2 *)Kij, Kijtmp;
      real2 *Mjptr = (real2 *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_4X4)
      real4x4 *Kijptr = (real4x4 *)Kij, Kijtmp;
      real4 *Mjptr = (real4 *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_4X2)
      real4x2 *Kijptr = (real4x2 *)Kij, Kijtmp;
      real2 *Mjptr = (real2 *)Mj + Mjoff, Mjtmp;
#endif

      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */ 
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
      i ++; // advance row index to next i
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
      i4 ++; // advance row index to next i
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
      i8 ++; // advance row index to next i
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
      i16 ++; // advance row index to next i
#endif

      /* Advance Lptr to next i */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r32b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X2)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8x2 *K, real2 *M, int level, int Mstart)
#elif defined(K_IS_4X4)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4x4 *K, real4 *M, int level, int Mstart)
#elif defined(K_IS_4X2)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4x2 *K, real2 *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_4X4)
  real4 *Mptr;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_4X4)
    /* M[level][j4=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 4) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart       + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_4X4)
  for (int j4 = 0; j4 < CUTOFF_L / 4; j4 ++) { // unrolling 4x
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  for (int j2 = 0; j2 < CUTOFF_L / 2; j2 ++) { // unrolling 2x
#else
  for (int j = 0; j < CUTOFF_L; j ++) { // no unrolling
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_4X4)
    __shared__ real4 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real  Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif
    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_4X4)
	real4 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real  *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by][cell=id] for no unrolling */
    int i = NROWS_L * by;
    real *Lptr = L + (((bx << LOG_CUTOFF_L) + i) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by/4][cell=id] for unrolling 4x */
    int i4 = ((NROWS_L * by) >> 2);
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + i4) << 9) + id;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by/8][cell=id] for unrolling 8x */
    int i8 = ((NROWS_L * by) >> 3);
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_L - 3)) + i8) << 9) + id;
#elif defined(K_IS_16X1)
    /* Initialise row index and set pointer to
       L[chunk][row=NROWS*by/16][cell=id] for unrolling 16x */
    int i16 = ((NROWS_L * by) >> 4);
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_L - 4)) + i16) << 9) + id;
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_L != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_L; iloc ++) // no unrolling
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    for (int iloc = 0; iloc < NROWS_L; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    for (int iloc = 0; iloc < NROWS_L; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 16) // unrolling 16x
#endif
#endif
    {
      /* Load Kij[z][y][x] */
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316];
      if (id < 316) Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
      __shared__ real4 Kij[316];
      if (id < 316) Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
      __shared__ real4 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_L / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316];
      if (id < 316) Kij[id] = *(K + (j * (CUTOFF_L / 16) + i16) * 316 + id);
#elif defined(K_IS_8X2)
      __shared__ real8x2 Kij[316];
      if (id < 316) Kij[id] = *(K + (j2 * (CUTOFF_L / 8) + i8) * 316 + id);
#elif defined(K_IS_4X4)
      __shared__ real4x4 Kij[316];
      if (id < 316) Kij[id] = *(K + (j4 * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_4X2)
      __shared__ real4x2 Kij[316];
      if (id < 316) Kij[id] = *(K + (j2 * (CUTOFF_L / 4) + i4) * 316 + id);
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij, Kijtmp;
      real2 *Mjptr = (real2 *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij, Kijtmp;
      real *Mjptr = (real *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij, Kijtmp;
      real *Mjptr = (real *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij, Kijtmp;
      real *Mjptr = (real *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_8X2)
      real8x2 *Kijptr = (real8x2 *)Kij, Kijtmp;
      real2 *Mjptr = (real2 *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_4X4)
      real4x4 *Kijptr = (real4x4 *)Kij, Kijtmp;
      real4 *Mjptr = (real4 *)Mj + Mjoff, Mjtmp;
#elif defined(K_IS_4X2)
      real4x2 *Kijptr = (real4x2 *)Kij, Kijtmp;
      real2 *Mjptr = (real2 *)Mj + Mjoff, Mjtmp;
#endif

      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */ 
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
      i ++; // advance row index to next i
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
      i4 ++; // advance row index to next i
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
      i8 ++; // advance row index to next i
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
      i16 ++; // advance row index to next i
#endif

      /* Advance Lptr to next i */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}
//////////////////////////////////////////
#else
//////////////////////////////////////////
#error Unsupported architecture.
//////////////////////////////////////////
#endif
//////////////////////////////////////////
/**************************************************************************/
#elif defined(CUDA_VER45G)
/**************************************************************************/
/* Based on VER45F */

#include "real.h"

#if !defined(K_IS_1X2_TEXTURE) && !defined(K_IS_4X1_TEXTURE) && !defined(K_IS_4X1) && !defined(K_IS_8X1) && !defined(K_IS_16X1) && !defined(K_IS_8X2) && !defined(K_IS_4X4) && !defined(K_IS_4X2)
#error Set an appropriate macro.
#endif

/* In general, the symbols Dx, Dy, and Dz defines the size of chunk,
   that is, each chunk consists of Dx*Dy*Dz clusters. In this code,
   Dx=Dy=Dz=4 is assumed and this number corresponds to 'B' in the
   paper and manual */

#define bx blockIdx.x   // chunk index
#define by blockIdx.y   // row-group index

#define tx threadIdx.x  // 0<=tx<Dx*Dz, where Dx=4 and Dz=4.
#define ty threadIdx.y  // 0<=ty<Dy, where Dy=4.
#define tz threadIdx.z  // 0<=tz<8, where tz is sibling-index of field cell

/* cutoff stands for the dimension of M-vector, L-vector, and
   K-matrix. This corresponds to 'r' in the paper and manual.  In this
   code, r is either 256 (high-precision version) or 32 (low-precision
   version) */
#define CUTOFF_H     256
#define LOG_CUTOFF_H   8
#define CUTOFF_L      32
#define LOG_CUTOFF_L   5

/* Set the number of rows per row-group. This parameter corresponds to
   'P' in the paper and manual */
#if !defined(NUM_ROW_GROUPS_IJ)
#define NUM_ROW_GROUPS_IJ 8 // 8 is better for C2050+SDK3.2
#endif
#if (NUM_ROW_GROUPS_IJ == 1)
#define NROWS_H 256 // cutoff=256
#define NROWS_L  32 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 2)
#define NROWS_H 128 // cutoff=256
#define NROWS_L  16 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 4)
#define NROWS_H  64 // cutoff=256
#define NROWS_L   8 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 8)
#define NROWS_H  32 // cutoff=256
#define NROWS_L   4 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 16)
#define NROWS_H  16 // cutoff=256
#define NROWS_L   2 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 32)
#define NROWS_H   8 // cutoff=256
#define NROWS_L   1 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 64)
#define NROWS_H   4 // cutoff=256
#define NROWS_L   0 // cutoff=32  IMPOSSIBLE
#else
#error Unsupposed NUM_ROW_GROUPS_IJ.
#endif

/* Macros to perform Li+=Kij*Mj for all the 316 Kij */
#if defined(K_IS_1X2_TEXTURE)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij += Kijtmp.x * Mjtmp.x;				\
  Lij += Kijtmp.y * Mjtmp.y;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.x * Mjtmp;				\
  Lij.y += Kijtmp.y * Mjtmp;				\
  Lij.z += Kijtmp.z * Mjtmp;				\
  Lij.w += Kijtmp.w * Mjtmp
#elif defined(K_IS_8X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.a * Mjtmp;				\
  Lij.b += Kijtmp.b * Mjtmp;				\
  Lij.c += Kijtmp.c * Mjtmp;				\
  Lij.d += Kijtmp.d * Mjtmp;				\
  Lij.e += Kijtmp.e * Mjtmp;				\
  Lij.f += Kijtmp.f * Mjtmp;				\
  Lij.g += Kijtmp.g * Mjtmp;				\
  Lij.h += Kijtmp.h * Mjtmp
#elif defined(K_IS_16X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.a * Mjtmp;				\
  Lij.b += Kijtmp.b * Mjtmp;				\
  Lij.c += Kijtmp.c * Mjtmp;				\
  Lij.d += Kijtmp.d * Mjtmp;				\
  Lij.e += Kijtmp.e * Mjtmp;				\
  Lij.f += Kijtmp.f * Mjtmp;				\
  Lij.g += Kijtmp.g * Mjtmp;				\
  Lij.h += Kijtmp.h * Mjtmp;				\
  Lij.i += Kijtmp.i * Mjtmp;				\
  Lij.j += Kijtmp.j * Mjtmp;				\
  Lij.k += Kijtmp.k * Mjtmp;				\
  Lij.l += Kijtmp.l * Mjtmp;				\
  Lij.m += Kijtmp.m * Mjtmp;				\
  Lij.n += Kijtmp.n * Mjtmp;				\
  Lij.o += Kijtmp.o * Mjtmp;				\
  Lij.p += Kijtmp.p * Mjtmp
#elif defined(K_IS_8X2)
#if(0) // worse than the original
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.aa * Mjtmp.x + Kijtmp.ab * Mjtmp.y;	\
  Lij.b += Kijtmp.ba * Mjtmp.x + Kijtmp.bb * Mjtmp.y;	\
  Lij.c += Kijtmp.ca * Mjtmp.x + Kijtmp.cb * Mjtmp.y;	\
  Lij.d += Kijtmp.da * Mjtmp.x + Kijtmp.db * Mjtmp.y;	\
  Lij.e += Kijtmp.ea * Mjtmp.x + Kijtmp.eb * Mjtmp.y;	\
  Lij.f += Kijtmp.fa * Mjtmp.x + Kijtmp.fb * Mjtmp.y;	\
  Lij.g += Kijtmp.ga * Mjtmp.x + Kijtmp.gb * Mjtmp.y;	\
  Lij.h += Kijtmp.ha * Mjtmp.x + Kijtmp.hb * Mjtmp.y
#endif
#if(0) // intended to execute MAD and MUL at the same time; same as the original
#define COMP(Kijoff_diff, Mjoff_diff)					\
  {									\
    Mjptr += Mjoff_diff;						\
    Mjtmp = *Mjptr;							\
    Kijptr += Kijoff_diff;						\
    Kijtmp = *Kijptr;							\
    real tmp;								\
    Lij.a += Kijtmp.aa * Mjtmp.x; tmp = Kijtmp.ab * Mjtmp.y;		\
    Lij.a += tmp;							\
    Lij.b += Kijtmp.ba * Mjtmp.x; tmp = Kijtmp.bb * Mjtmp.y;		\
    Lij.b += tmp;							\
    Lij.c += Kijtmp.ca * Mjtmp.x; tmp = Kijtmp.cb * Mjtmp.y;		\
    Lij.c += tmp;							\
    Lij.d += Kijtmp.da * Mjtmp.x; tmp = Kijtmp.db * Mjtmp.y;		\
    Lij.d += tmp;							\
    Lij.e += Kijtmp.ea * Mjtmp.x; tmp = Kijtmp.eb * Mjtmp.y;		\
    Lij.e += tmp;							\
    Lij.f += Kijtmp.fa * Mjtmp.x; tmp = Kijtmp.fb * Mjtmp.y;		\
    Lij.f += tmp;							\
    Lij.g += Kijtmp.ga * Mjtmp.x; tmp = Kijtmp.gb * Mjtmp.y;		\
    Lij.g += tmp;							\
    Lij.h += Kijtmp.ha * Mjtmp.x; tmp = Kijtmp.hb * Mjtmp.y;		\
    Lij.h += tmp;							\
  }
#endif
#if(1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.aa * Mjtmp.x;				\
  Lij.a += Kijtmp.ab * Mjtmp.y;				\
  Lij.b += Kijtmp.ba * Mjtmp.x;				\
  Lij.b += Kijtmp.bb * Mjtmp.y;				\
  Lij.c += Kijtmp.ca * Mjtmp.x;				\
  Lij.c += Kijtmp.cb * Mjtmp.y;				\
  Lij.d += Kijtmp.da * Mjtmp.x;				\
  Lij.d += Kijtmp.db * Mjtmp.y;				\
  Lij.e += Kijtmp.ea * Mjtmp.x;				\
  Lij.e += Kijtmp.eb * Mjtmp.y;				\
  Lij.f += Kijtmp.fa * Mjtmp.x;				\
  Lij.f += Kijtmp.fb * Mjtmp.y;				\
  Lij.g += Kijtmp.ga * Mjtmp.x;				\
  Lij.g += Kijtmp.gb * Mjtmp.y;				\
  Lij.h += Kijtmp.ha * Mjtmp.x;				\
  Lij.h += Kijtmp.hb * Mjtmp.y
#endif
#elif defined(K_IS_4X4)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.xx * Mjtmp.x;				\
  Lij.x += Kijtmp.xy * Mjtmp.y;				\
  Lij.x += Kijtmp.xz * Mjtmp.z;				\
  Lij.x += Kijtmp.xw * Mjtmp.w;				\
  Lij.y += Kijtmp.yx * Mjtmp.x;				\
  Lij.y += Kijtmp.yy * Mjtmp.y;				\
  Lij.y += Kijtmp.yz * Mjtmp.z;				\
  Lij.y += Kijtmp.yw * Mjtmp.w;				\
  Lij.z += Kijtmp.zx * Mjtmp.x;				\
  Lij.z += Kijtmp.zy * Mjtmp.y;				\
  Lij.z += Kijtmp.zz * Mjtmp.z;				\
  Lij.z += Kijtmp.zw * Mjtmp.w;				\
  Lij.w += Kijtmp.wx * Mjtmp.x;				\
  Lij.w += Kijtmp.wy * Mjtmp.y;				\
  Lij.w += Kijtmp.wz * Mjtmp.z;				\
  Lij.w += Kijtmp.ww * Mjtmp.w
#elif defined(K_IS_4X2)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.xx * Mjtmp.x;				\
  Lij.x += Kijtmp.xy * Mjtmp.y;				\
  Lij.y += Kijtmp.yx * Mjtmp.x;				\
  Lij.y += Kijtmp.yy * Mjtmp.y;				\
  Lij.z += Kijtmp.zx * Mjtmp.x;				\
  Lij.z += Kijtmp.zy * Mjtmp.y;				\
  Lij.w += Kijtmp.wx * Mjtmp.x;				\
  Lij.w += Kijtmp.wy * Mjtmp.y
#endif
#define COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)

#if defined(K_IS_1X2_TEXTURE)
/* Declare the global variable of type texture to use float2 CUDA
   array (texture) that contains 316 K-matrices */
texture<float2, 3, cudaReadModeElementType> texRefK;
#elif defined(K_IS_4X1_TEXTURE)
/* Declare the global variable of type texture to use float4 CUDA
   array (texture) that contains 316 K-matrices */
texture<float4, 3, cudaReadModeElementType> texRefK;
#endif


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r256b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X2)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8x2 *K, real2 *M, int level, int Mstart)
#elif defined(K_IS_4X4)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4x4 *K, real4 *M, int level, int Mstart)
#elif defined(K_IS_4X2)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4x2 *K, real2 *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_4X4)
  real4 *Mptr;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_4X4)
    /* M[level][j4=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 4) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart       + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_4X4)
  for (int j4 = 0; j4 < CUTOFF_H / 4; j4 ++) { // unrolling 4x
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  for (int j2 = 0; j2 < CUTOFF_H / 2; j2 ++) { // unrolling 2x
#else
  for (int j = 0; j < CUTOFF_H; j ++) { // no unrolling
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_4X4)
    __shared__ real4 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real  Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_4X4)
	real4 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real  *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_H) + (NROWS_H * by)) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + ((NROWS_H * by) >> 2)) << 9) + id;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set a pointer to L (L[chunk][row=NROWS*by/8][sib=tz][cell=id]) */
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_H - 3)) + ((NROWS_H * by) >> 3)) << 9) + id;
#elif defined(K_IS_16X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/16][sib=tz][cell=id]) */
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_H - 4)) + ((NROWS_H * by) >> 4)) << 9) + id;
#endif

#if defined(K_IS_1X2_TEXTURE)
    /* Set row index for no unrolling */
    int i = NROWS_H * by;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    /* Set row index for unrolling x4 */
    int i4 = ((NROWS_H * by) >> 2);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set row index for unrolling x8 */
    int i8 = ((NROWS_H * by) >> 3);
#elif defined(K_IS_16X1)
    /* Set row index for unrolling x16 */
    int i16 = ((NROWS_H * by) >> 4);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_H != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_H; iloc ++) // no unrolling
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    for (int iloc = 0; iloc < NROWS_H; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    for (int iloc = 0; iloc < NROWS_H; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 16) // unrolling 16x
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X2)
      __shared__ real8x2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X4)
      __shared__ real4x4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X2)
      __shared__ real4x2 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 16) + i16) * 316 + id);
#elif defined(K_IS_8X2)
	Kij[id] = *(K + (j2 * (CUTOFF_H / 8) + i8) * 316 + id);
#elif defined(K_IS_4X4)
	Kij[id] = *(K + (j4 * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_4X2)
	Kij[id] = *(K + (j2 * (CUTOFF_H / 4) + i4) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      /* Advance row index from i to i+4 */
      i4 ++;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      /* Advance row index from i to i+8 */
      i8 ++;
#elif defined(K_IS_16X1)
      /* Advance row index from i to i+16 */
      i16 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X2)
      real8x2 *Kijptr = (real8x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X4)
      real4x4 *Kijptr = (real4x4 *)Kij;
      real4 *Mjptr = (real4 *)Mj + Mjoff;
#elif defined(K_IS_4X2)
      real4x2 *Kijptr = (real4x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X1)
      real8 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_16X1)
      real16 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X2)
      real8x2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X4)
      real4x4 Kijtmp;
      real4 Mjtmp;
#elif defined(K_IS_4X2)
      real4x2 Kijtmp;
      real2 Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r32b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X2)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8x2 *K, real2 *M, int level, int Mstart)
#elif defined(K_IS_4X4)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4x4 *K, real4 *M, int level, int Mstart)
#elif defined(K_IS_4X2)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4x2 *K, real2 *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_4X4)
  real4 *Mptr;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_4X4)
    /* M[level][j4=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 4) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart       + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_4X4)
  for (int j4 = 0; j4 < CUTOFF_L / 4; j4 ++) { // unrolling 4x
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  for (int j2 = 0; j2 < CUTOFF_L / 2; j2 ++) { // unrolling 2x
#else
  for (int j = 0; j < CUTOFF_L; j ++) { // no unrolling
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_4X4)
    __shared__ real4 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real  Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_4X4)
	real4 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real  *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_L) + (NROWS_L * by)) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + ((NROWS_L * by) >> 2)) << 9) + id;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set a pointer to L (L[chunk][row=NROWS*by/8][sib=tz][cell=id]) */
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_L - 3)) + ((NROWS_L * by) >> 3)) << 9) + id;
#elif defined(K_IS_16X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/16][sib=tz][cell=id]) */
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_L - 4)) + ((NROWS_L * by) >> 4)) << 9) + id;
#endif

#if defined(K_IS_1X2_TEXTURE)
    /* Set row index for no unrolling */
    int i = NROWS_L * by;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    /* Set row index for unrolling x4 */
    int i4 = ((NROWS_L * by) >> 2);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set row index for unrolling x8 */
    int i8 = ((NROWS_L * by) >> 3);
#elif defined(K_IS_16X1)
    /* Set row index for unrolling x16 */
    int i16 = ((NROWS_L * by) >> 4);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_L != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_L; iloc ++) // no unrolling
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    for (int iloc = 0; iloc < NROWS_L; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    for (int iloc = 0; iloc < NROWS_L; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 16) // unrolling 16x
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X2)
      __shared__ real8x2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X4)
      __shared__ real4x4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X2)
      __shared__ real4x2 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 16) + i16) * 316 + id);
#elif defined(K_IS_8X2)
	Kij[id] = *(K + (j2 * (CUTOFF_L / 8) + i8) * 316 + id);
#elif defined(K_IS_4X4)
	Kij[id] = *(K + (j4 * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_4X2)
	Kij[id] = *(K + (j2 * (CUTOFF_L / 4) + i4) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      /* Advance row index from i to i+4 */
      i4 ++;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      /* Advance row index from i to i+8 */
      i8 ++;
#elif defined(K_IS_16X1)
      /* Advance row index from i to i+16 */
      i16 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X2)
      real8x2 *Kijptr = (real8x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X4)
      real4x4 *Kijptr = (real4x4 *)Kij;
      real4 *Mjptr = (real4 *)Mj + Mjoff;
#elif defined(K_IS_4X2)
      real4x2 *Kijptr = (real4x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X1)
      real8 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_16X1)
      real16 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X2)
      real8x2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X4)
      real4x4 Kijtmp;
      real4 Mjtmp;
#elif defined(K_IS_4X2)
      real4x2 Kijtmp;
      real2 Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}
/**************************************************************************/
#elif defined(CUDA_VER45F)
/**************************************************************************/
/* Based on VER45E */

#include "real.h"

#if !defined(K_IS_1X2_TEXTURE) && !defined(K_IS_4X1_TEXTURE) && !defined(K_IS_4X1) && !defined(K_IS_8X1) && !defined(K_IS_16X1) && !defined(K_IS_8X2)
#error Set an appropriate macro.
#endif

/* In general, the symbols Dx, Dy, and Dz defines the size of chunk,
   that is, each chunk consists of Dx*Dy*Dz clusters. In this code,
   Dx=Dy=Dz=4 is assumed and this number corresponds to 'B' in the
   paper and manual */

#define bx blockIdx.x   // chunk index
#define by blockIdx.y   // row-group index

#define tx threadIdx.x  // 0<=tx<Dx*Dz, where Dx=4 and Dz=4.
#define ty threadIdx.y  // 0<=ty<Dy, where Dy=4.
#define tz threadIdx.z  // 0<=tz<8, where tz is sibling-index of field cell

/* cutoff stands for the dimension of M-vector, L-vector, and
   K-matrix. This corresponds to 'r' in the paper and manual.  In this
   code, r is either 256 (high-precision version) or 32 (low-precision
   version) */
#define CUTOFF_H     256
#define LOG_CUTOFF_H   8
#define CUTOFF_L      32
#define LOG_CUTOFF_L   5

/* Set the number of rows per row-group. This parameter corresponds to
   'P' in the paper and manual */
#if !defined(NUM_ROW_GROUPS_IJ)
#define NUM_ROW_GROUPS_IJ 8 // 8 is better for C2050+SDK3.2
#endif
#if (NUM_ROW_GROUPS_IJ == 1)
#define NROWS_H 256 // cutoff=256
#define NROWS_L  32 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 2)
#define NROWS_H 128 // cutoff=256
#define NROWS_L  16 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 4)
#define NROWS_H  64 // cutoff=256
#define NROWS_L   8 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 8)
#define NROWS_H  32 // cutoff=256
#define NROWS_L   4 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 16)
#define NROWS_H  16 // cutoff=256
#define NROWS_L   2 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 32)
#define NROWS_H   8 // cutoff=256
#define NROWS_L   1 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 64)
#define NROWS_H   4 // cutoff=256
#define NROWS_L   0 // cutoff=32  IMPOSSIBLE
#else
#error Unsupposed NUM_ROW_GROUPS_IJ.
#endif

/* Macros to perform Li+=Kij*Mj for all the 316 Kij */
#if defined(K_IS_1X2_TEXTURE)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij += Kijtmp.x * Mjtmp.x;				\
  Lij += Kijtmp.y * Mjtmp.y;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.x * Mjtmp;				\
  Lij.y += Kijtmp.y * Mjtmp;				\
  Lij.z += Kijtmp.z * Mjtmp;				\
  Lij.w += Kijtmp.w * Mjtmp
#elif defined(K_IS_8X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.a * Mjtmp;				\
  Lij.b += Kijtmp.b * Mjtmp;				\
  Lij.c += Kijtmp.c * Mjtmp;				\
  Lij.d += Kijtmp.d * Mjtmp;				\
  Lij.e += Kijtmp.e * Mjtmp;				\
  Lij.f += Kijtmp.f * Mjtmp;				\
  Lij.g += Kijtmp.g * Mjtmp;				\
  Lij.h += Kijtmp.h * Mjtmp
#elif defined(K_IS_16X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.a * Mjtmp;				\
  Lij.b += Kijtmp.b * Mjtmp;				\
  Lij.c += Kijtmp.c * Mjtmp;				\
  Lij.d += Kijtmp.d * Mjtmp;				\
  Lij.e += Kijtmp.e * Mjtmp;				\
  Lij.f += Kijtmp.f * Mjtmp;				\
  Lij.g += Kijtmp.g * Mjtmp;				\
  Lij.h += Kijtmp.h * Mjtmp;				\
  Lij.i += Kijtmp.i * Mjtmp;				\
  Lij.j += Kijtmp.j * Mjtmp;				\
  Lij.k += Kijtmp.k * Mjtmp;				\
  Lij.l += Kijtmp.l * Mjtmp;				\
  Lij.m += Kijtmp.m * Mjtmp;				\
  Lij.n += Kijtmp.n * Mjtmp;				\
  Lij.o += Kijtmp.o * Mjtmp;				\
  Lij.p += Kijtmp.p * Mjtmp
#elif defined(K_IS_8X2)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.aa * Mjtmp.x;				\
  Lij.a += Kijtmp.ab * Mjtmp.y;				\
  Lij.b += Kijtmp.ba * Mjtmp.x;				\
  Lij.b += Kijtmp.bb * Mjtmp.y;				\
  Lij.c += Kijtmp.ca * Mjtmp.x;				\
  Lij.c += Kijtmp.cb * Mjtmp.y;				\
  Lij.d += Kijtmp.da * Mjtmp.x;				\
  Lij.d += Kijtmp.db * Mjtmp.y;				\
  Lij.e += Kijtmp.ea * Mjtmp.x;				\
  Lij.e += Kijtmp.eb * Mjtmp.y;				\
  Lij.f += Kijtmp.fa * Mjtmp.x;				\
  Lij.f += Kijtmp.fb * Mjtmp.y;				\
  Lij.g += Kijtmp.ga * Mjtmp.x;				\
  Lij.g += Kijtmp.gb * Mjtmp.y;				\
  Lij.h += Kijtmp.ha * Mjtmp.x;				\
  Lij.h += Kijtmp.hb * Mjtmp.y
#endif
#define COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)

#if defined(K_IS_1X2_TEXTURE)
/* Declare the global variable of type texture to use float2 CUDA
   array (texture) that contains 316 K-matrices */
texture<float2, 3, cudaReadModeElementType> texRefK;
#elif defined(K_IS_4X1_TEXTURE)
/* Declare the global variable of type texture to use float4 CUDA
   array (texture) that contains 316 K-matrices */
texture<float4, 3, cudaReadModeElementType> texRefK;
#endif


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r256b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X2)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8x2 *K, real2 *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
  for (int j2 = 0; j2 < CUTOFF_H / 2; j2 ++) { // unrolling 2x
#else
  for (int j = 0; j < CUTOFF_H; j ++) { // no unrolling
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_H) + (NROWS_H * by)) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + ((NROWS_H * by) >> 2)) << 9) + id;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set a pointer to L (L[chunk][row=NROWS*by/8][sib=tz][cell=id]) */
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_H - 3)) + ((NROWS_H * by) >> 3)) << 9) + id;
#elif defined(K_IS_16X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/16][sib=tz][cell=id]) */
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_H - 4)) + ((NROWS_H * by) >> 4)) << 9) + id;
#endif

#if defined(K_IS_1X2_TEXTURE)
    /* Set row index for no unrolling */
    int i = NROWS_H * by;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set row index for unrolling x4 */
    int i4 = ((NROWS_H * by) >> 2);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set row index for unrolling x8 */
    int i8 = ((NROWS_H * by) >> 3);
#elif defined(K_IS_16X1)
    /* Set row index for unrolling x16 */
    int i16 = ((NROWS_H * by) >> 4);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_H != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_H; iloc ++)
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    for (int iloc = 0; iloc < NROWS_H; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 16) // unrolling 16x
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X2)
      __shared__ real8x2 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 16) + i16) * 316 + id);
#elif defined(K_IS_8X2)
	Kij[id] = *(K + (j2 * (CUTOFF_H / 8) + i8) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      /* Advance row index from i to i+4 */
      i4 ++;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      /* Advance row index from i to i+8 */
      i8 ++;
#elif defined(K_IS_16X1)
      /* Advance row index from i to i+16 */
      i16 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X2)
      real8x2 *Kijptr = (real8x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X1)
      real8 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_16X1)
      real16 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X2)
      real8x2 Kijtmp;
      real2 Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r32b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X2)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8x2 *K, real2 *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
  for (int j2 = 0; j2 < CUTOFF_L / 2; j2 ++) {
#else
  for (int j = 0; j < CUTOFF_L; j ++) {
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_L) + (NROWS_L * by)) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + ((NROWS_L * by) >> 2)) << 9) + id;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set a pointer to L (L[chunk][row=NROWS*by/8][sib=tz][cell=id]) */
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_L - 3)) + ((NROWS_L * by) >> 3)) << 9) + id;
#elif defined(K_IS_16X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/16][sib=tz][cell=id]) */
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_L - 4)) + ((NROWS_L * by) >> 4)) << 9) + id;
#endif

#if defined(K_IS_1X2_TEXTURE)
    /* Set row index for no unrolling */
    int i = NROWS_L * by;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set row index for unrolling x4 */
    int i4 = ((NROWS_L * by) >> 2);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    /* Set row index for unrolling x8 */
    int i8 = ((NROWS_L * by) >> 3);
#elif defined(K_IS_16X1)
    /* Set row index for unrolling x16 */
    int i16 = ((NROWS_L * by) >> 4);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_L != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_L; iloc ++)
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    for (int iloc = 0; iloc < NROWS_L; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 16) // unrolling 16x
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X2)
      __shared__ real8x2 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 16) + i16) * 316 + id);
#elif defined(K_IS_8X2)
	Kij[id] = *(K + (j2 * (CUTOFF_L / 8) + i8) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      /* Advance row index from i to i+4 */
      i4 ++;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      /* Advance row index from i to i+8 */
      i8 ++;
#elif defined(K_IS_16X1)
      /* Advance row index from i to i+16 */
      i16 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X2)
      real8x2 *Kijptr = (real8x2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X1)
      real8 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_16X1)
      real16 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X2)
      real8x2 Kijtmp;
      real2 Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}
/**************************************************************************/
#elif defined(CUDA_VER45E)
/**************************************************************************/
#error This version is obsolete.

/* Based on VER45D */

#include "real.h"

#if !defined(K_IS_1X2_TEXTURE) && !defined(K_IS_4X1_TEXTURE) && !defined(K_IS_4X1) && !defined(K_IS_8X1) && !defined(K_IS_16X1)
#error Set an appropriate macro.
#endif

/* In general, the symbols Dx, Dy, and Dz defines the size of chunk,
   that is, each chunk consists of Dx*Dy*Dz clusters. In this code,
   Dx=Dy=Dz=4 is assumed and this number corresponds to 'B' in the
   paper and manual */

#define bx blockIdx.x   // chunk index
#define by blockIdx.y   // row-group index

#define tx threadIdx.x  // 0<=tx<Dx*Dz, where Dx=4 and Dz=4.
#define ty threadIdx.y  // 0<=ty<Dy, where Dy=4.
#define tz threadIdx.z  // 0<=tz<8, where tz is sibling-index of field cell

/* cutoff stands for the dimension of M-vector, L-vector, and
   K-matrix. This corresponds to 'r' in the paper and manual.  In this
   code, r is either 256 (high-precision version) or 32 (low-precision
   version) */
#define CUTOFF_H     256
#define LOG_CUTOFF_H   8
#define CUTOFF_L      32
#define LOG_CUTOFF_L   5

/* Set the number of rows per row-group. This parameter corresponds to
   'P' in the paper and manual */
#if !defined(NUM_ROW_GROUPS_IJ)
#define NUM_ROW_GROUPS_IJ 8 // 8 is better for C2050+SDK3.2
#endif
#if (NUM_ROW_GROUPS_IJ == 1)
#define NROWS_H 256 // cutoff=256
#define NROWS_L  32 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 2)
#define NROWS_H 128 // cutoff=256
#define NROWS_L  16 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 4)
#define NROWS_H  64 // cutoff=256
#define NROWS_L   8 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 8)
#define NROWS_H  32 // cutoff=256
#define NROWS_L   4 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 16)
#define NROWS_H  16 // cutoff=256
#define NROWS_L   2 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 32)
#define NROWS_H   8 // cutoff=256
#define NROWS_L   1 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 64)
#define NROWS_H   4 // cutoff=256
#define NROWS_L   0 // cutoff=32  IMPOSSIBLE
#else
#error Unsupposed NUM_ROW_GROUPS_IJ.
#endif

/* Macros to perform Li+=Kij*Mj for all the 316 Kij */
#if defined(K_IS_1X2_TEXTURE)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij += Kijtmp.x * Mjtmp.x;				\
  Lij += Kijtmp.y * Mjtmp.y;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.x * Mjtmp;				\
  Lij.y += Kijtmp.y * Mjtmp;				\
  Lij.z += Kijtmp.z * Mjtmp;				\
  Lij.w += Kijtmp.w * Mjtmp
#elif defined(K_IS_8X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.a * Mjtmp;				\
  Lij.b += Kijtmp.b * Mjtmp;				\
  Lij.c += Kijtmp.c * Mjtmp;				\
  Lij.d += Kijtmp.d * Mjtmp;				\
  Lij.e += Kijtmp.e * Mjtmp;				\
  Lij.f += Kijtmp.f * Mjtmp;				\
  Lij.g += Kijtmp.g * Mjtmp;				\
  Lij.h += Kijtmp.h * Mjtmp
#elif defined(K_IS_16X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.a * Mjtmp;				\
  Lij.b += Kijtmp.b * Mjtmp;				\
  Lij.c += Kijtmp.c * Mjtmp;				\
  Lij.d += Kijtmp.d * Mjtmp;				\
  Lij.e += Kijtmp.e * Mjtmp;				\
  Lij.f += Kijtmp.f * Mjtmp;				\
  Lij.g += Kijtmp.g * Mjtmp;				\
  Lij.h += Kijtmp.h * Mjtmp;				\
  Lij.i += Kijtmp.i * Mjtmp;				\
  Lij.j += Kijtmp.j * Mjtmp;				\
  Lij.k += Kijtmp.k * Mjtmp;				\
  Lij.l += Kijtmp.l * Mjtmp;				\
  Lij.m += Kijtmp.m * Mjtmp;				\
  Lij.n += Kijtmp.n * Mjtmp;				\
  Lij.o += Kijtmp.o * Mjtmp;				\
  Lij.p += Kijtmp.p * Mjtmp
#endif
#define COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)

#if defined(K_IS_1X2_TEXTURE)
/* Declare the global variable of type texture to use float2 CUDA
   array (texture) that contains 316 K-matrices */
texture<float2, 3, cudaReadModeElementType> texRefK;
#elif defined(K_IS_4X1_TEXTURE)
/* Declare the global variable of type texture to use float4 CUDA
   array (texture) that contains 316 K-matrices */
texture<float4, 3, cudaReadModeElementType> texRefK;
#endif

#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r256b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_1X2_TEXTURE)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_1X2_TEXTURE)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_1X2_TEXTURE)
  for (int j2 = 0; j2 < CUTOFF_H / 2; j2 ++) {
#else
  for (int j = 0; j < CUTOFF_H; j ++) {
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_1X2_TEXTURE)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_1X2_TEXTURE)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_H) + (NROWS_H * by)) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + ((NROWS_H * by) >> 2)) << 9) + id;
#elif defined(K_IS_8X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/8][sib=tz][cell=id]) */
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_H - 3)) + ((NROWS_H * by) >> 3)) << 9) + id;
#elif defined(K_IS_16X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/16][sib=tz][cell=id]) */
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_H - 4)) + ((NROWS_H * by) >> 4)) << 9) + id;
#endif

#if defined(K_IS_1X2_TEXTURE)
    /* Set row index for no unrolling */
    int i = NROWS_H * by;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set row index for unrolling x4 */
    int i4 = ((NROWS_H * by) >> 2);
#elif defined(K_IS_8X1)
    /* Set row index for unrolling x8 */
    int i8 = ((NROWS_H * by) >> 3);
#elif defined(K_IS_16X1)
    /* Set row index for unrolling x16 */
    int i16 = ((NROWS_H * by) >> 4);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_H != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_H; iloc ++)
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 16) // unrolling 16x
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 16) + i16) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      /* Advance row index from i to i+4 */
      i4 ++;
#elif defined(K_IS_8X1)
      /* Advance row index from i to i+8 */
      i8 ++;
#elif defined(K_IS_16X1)
      /* Advance row index from i to i+16 */
      i16 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h
		    = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X1)
      real8 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_16X1)
      real16 Kijtmp;
      real Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#elif defined(K_IS_16X1)
__global__ void m2l_kern_ij_blocking_r32b4(real16 *L, real16 *K, real *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_1X2_TEXTURE)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_1X2_TEXTURE)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_1X2_TEXTURE)
  for (int j2 = 0; j2 < CUTOFF_L / 2; j2 ++) {
#else
  for (int j = 0; j < CUTOFF_L; j ++) {
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_1X2_TEXTURE)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_1X2_TEXTURE)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_L) + (NROWS_L * by)) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + ((NROWS_L * by) >> 2)) << 9) + id;
#elif defined(K_IS_8X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/8][sib=tz][cell=id]) */
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_L - 3)) + ((NROWS_L * by) >> 3)) << 9) + id;
#elif defined(K_IS_16X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/16][sib=tz][cell=id]) */
    real16 *Lptr = L + (((bx << (LOG_CUTOFF_L - 4)) + ((NROWS_L * by) >> 4)) << 9) + id;
#endif

#if defined(K_IS_1X2_TEXTURE)
    /* Set row index for no unrolling */
    int i = NROWS_L * by;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set row index for unrolling x4 */
    int i4 = ((NROWS_L * by) >> 2);
#elif defined(K_IS_8X1)
    /* Set row index for unrolling x8 */
    int i8 = ((NROWS_L * by) >> 3);
#elif defined(K_IS_16X1)
    /* Set row index for unrolling x16 */
    int i16 = ((NROWS_L * by) >> 4);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_L != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_L; iloc ++)
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 8) // unrolling 8x
#elif defined(K_IS_16X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 16) // unrolling 16x
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_16X1)
      __shared__ real16 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 8) + i8) * 316 + id);
#elif defined(K_IS_16X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 16) + i16) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      /* Advance row index from i to i+4 */
      i4 ++;
#elif defined(K_IS_8X1)
      /* Advance row index from i to i+8 */
      i8 ++;
#elif defined(K_IS_16X1)
      /* Advance row index from i to i+16 */
      i16 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#elif defined(K_IS_16X1)
      real16 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h
		    = Lij.i = Lij.j = Lij.k = Lij.l = Lij.m = Lij.n = Lij.o = Lij.p = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_16X1)
      real16 *Kijptr = (real16 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X1)
      real8 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_16X1)
      real16 Kijtmp;
      real Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#elif defined(K_IS_16X1)
      real16 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      Ltmp.i += Lij.i;
      Ltmp.j += Lij.j;
      Ltmp.k += Lij.k;
      Ltmp.l += Lij.l;
      Ltmp.m += Lij.m;
      Ltmp.n += Lij.n;
      Ltmp.o += Lij.o;
      Ltmp.p += Lij.p;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}
/**************************************************************************/
#elif defined(CUDA_VER45D)
/**************************************************************************/
#error This version is obsolete.

/* Based on VER45C */

#include "real.h"

#if !defined(K_IS_1X2_TEXTURE) && !defined(K_IS_4X1_TEXTURE) && !defined(K_IS_4X1) && !defined(K_IS_8X1)
#error Set an appropriate macro.
#endif

/* In general, the symbols Dx, Dy, and Dz defines the size of chunk,
   that is, each chunk consists of Dx*Dy*Dz clusters. In this code,
   Dx=Dy=Dz=4 is assumed and this number corresponds to 'B' in the
   paper and manual */

#define bx blockIdx.x   // chunk index
#define by blockIdx.y   // row-group index

#define tx threadIdx.x  // 0<=tx<Dx*Dz, where Dx=4 and Dz=4.
#define ty threadIdx.y  // 0<=ty<Dy, where Dy=4.
#define tz threadIdx.z  // 0<=tz<8, where tz is sibling-index of field cell

/* cutoff stands for the dimension of M-vector, L-vector, and
   K-matrix. This corresponds to 'r' in the paper and manual.  In this
   code, r is either 256 (high-precision version) or 32 (low-precision
   version) */
#define CUTOFF_H     256
#define LOG_CUTOFF_H   8
#define CUTOFF_L      32
#define LOG_CUTOFF_L   5

/* Set the number of rows per row-group. This parameter corresponds to
   'P' in the paper and manual */
#if !defined(NUM_ROW_GROUPS_IJ)
#define NUM_ROW_GROUPS_IJ 8 // 8 is better for C2050+SDK3.2
#endif
#if (NUM_ROW_GROUPS_IJ == 1)
#define NROWS_H 256 // cutoff=256
#define NROWS_L  32 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 2)
#define NROWS_H 128 // cutoff=256
#define NROWS_L  16 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 4)
#define NROWS_H  64 // cutoff=256
#define NROWS_L   8 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 8)
#define NROWS_H  32 // cutoff=256
#define NROWS_L   4 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 16)
#define NROWS_H  16 // cutoff=256
#define NROWS_L   2 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 32)
#define NROWS_H   8 // cutoff=256
#define NROWS_L   1 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 64)
#define NROWS_H   4 // cutoff=256
#define NROWS_L   0 // cutoff=32  IMPOSSIBLE
#else
#error Unsupposed NUM_ROW_GROUPS_IJ.
#endif

/* Macros to perform Li+=Kij*Mj for all the 316 Kij */
#if defined(K_IS_1X2_TEXTURE)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij += Kijtmp.x * Mjtmp.x;				\
  Lij += Kijtmp.y * Mjtmp.y;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.x * Mjtmp;				\
  Lij.y += Kijtmp.y * Mjtmp;				\
  Lij.z += Kijtmp.z * Mjtmp;				\
  Lij.w += Kijtmp.w * Mjtmp
#elif defined(K_IS_8X1)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.a += Kijtmp.a * Mjtmp;				\
  Lij.b += Kijtmp.b * Mjtmp;				\
  Lij.c += Kijtmp.c * Mjtmp;				\
  Lij.d += Kijtmp.d * Mjtmp;				\
  Lij.e += Kijtmp.e * Mjtmp;				\
  Lij.f += Kijtmp.f * Mjtmp;				\
  Lij.g += Kijtmp.g * Mjtmp;				\
  Lij.h += Kijtmp.h * Mjtmp
#endif
#define COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)

#if defined(K_IS_1X2_TEXTURE)
/* Declare the global variable of type texture to use float2 CUDA
   array (texture) that contains 316 K-matrices */
texture<float2, 3, cudaReadModeElementType> texRefK;
#elif defined(K_IS_4X1_TEXTURE)
/* Declare the global variable of type texture to use float4 CUDA
   array (texture) that contains 316 K-matrices */
texture<float4, 3, cudaReadModeElementType> texRefK;
#endif

#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r256b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_1X2_TEXTURE)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_1X2_TEXTURE)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_1X2_TEXTURE)
  for (int j2 = 0; j2 < CUTOFF_H / 2; j2 ++) {
#else
  for (int j = 0; j < CUTOFF_H; j ++) {
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_1X2_TEXTURE)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_1X2_TEXTURE)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_H) + (NROWS_H * by)) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + ((NROWS_H * by) >> 2)) << 9) + id;
#elif defined(K_IS_8X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/8][sib=tz][cell=id]) */
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_H - 3)) + ((NROWS_H * by) >> 3)) << 9) + id;
#endif

    /* Set row index for unrolling x4 */
#if defined(K_IS_1X2_TEXTURE)
    int i = NROWS_H * by;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    int i4 = ((NROWS_H * by) >> 2);
#elif defined(K_IS_8X1)
    int i8 = ((NROWS_H * by) >> 3);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_H != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_H; iloc ++)
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 8) // unrolling 8x
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 8) + i8) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      /* Advance row index from i to i+4 */
      i4 ++;
#elif defined(K_IS_8X1)
      /* Advance row index from i to i+8 */
      i8 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X1)
      real8 Kijtmp;
      real Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}


#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#elif defined(K_IS_8X1)
__global__ void m2l_kern_ij_blocking_r32b4(real8 *L, real8 *K, real *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_1X2_TEXTURE)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_1X2_TEXTURE)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_1X2_TEXTURE)
  for (int j2 = 0; j2 < CUTOFF_L / 2; j2 ++) {
#else
  for (int j = 0; j < CUTOFF_L; j ++) {
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_1X2_TEXTURE)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_1X2_TEXTURE)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_L) + (NROWS_L * by)) << 9) + id;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + ((NROWS_L * by) >> 2)) << 9) + id;
#elif defined(K_IS_8X1)
    /* Set a pointer to L (L[chunk][row=NROWS*by/8][sib=tz][cell=id]) */
    real8 *Lptr = L + (((bx << (LOG_CUTOFF_L - 3)) + ((NROWS_L * by) >> 3)) << 9) + id;
#endif

    /* Set row index for unrolling x4 */
#if defined(K_IS_1X2_TEXTURE)
    int i = NROWS_L * by;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    int i4 = ((NROWS_L * by) >> 2);
#elif defined(K_IS_8X1)
    int i8 = ((NROWS_L * by) >> 3);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_L != 1)
#if defined(K_IS_1X2_TEXTURE)
    for (int iloc = 0; iloc < NROWS_L; iloc ++)
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 4) // unrolling 4x
#elif defined(K_IS_8X1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 8) // unrolling 8x
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#elif defined(K_IS_8X1)
      __shared__ real8 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 4) + i4) * 316 + id);
#elif defined(K_IS_8X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 8) + i8) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      /* Advance row index from i to i+4 */
      i4 ++;
#elif defined(K_IS_8X1)
      /* Advance row index from i to i+8 */
      i8 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#elif defined(K_IS_8X1)
      real8 Lij; Lij.a = Lij.b = Lij.c = Lij.d = Lij.e = Lij.f = Lij.g = Lij.h = ZERO;
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#elif defined(K_IS_8X1)
      real8 *Kijptr = (real8 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Kijtmp;
      real Mjtmp;
#elif defined(K_IS_8X1)
      real8 Kijtmp;
      real Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#elif defined(K_IS_8X1)
      real8 Ltmp = *Lptr;
      Ltmp.a += Lij.a;
      Ltmp.b += Lij.b;
      Ltmp.c += Lij.c;
      Ltmp.d += Lij.d;
      Ltmp.e += Lij.e;
      Ltmp.f += Lij.f;
      Ltmp.g += Lij.g;
      Ltmp.h += Lij.h;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}


/**************************************************************************/
#elif defined(CUDA_VER45C)
/**************************************************************************/
#error This version is obsolete.

/* Based on VER45B */

#include "real.h"

#if !defined(K_IS_1X2_TEXTURE) && !defined(K_IS_4X1_TEXTURE) && !defined(K_IS_4X1)
#error Set an appropriate macro.
#endif

/* In general, the symbols Dx, Dy, and Dz defines the size of chunk,
   that is, each chunk consists of Dx*Dy*Dz clusters. In this code,
   Dx=Dy=Dz=4 is assumed and this number corresponds to 'B' in the
   paper and manual */

#define bx blockIdx.x   // chunk index
#define by blockIdx.y   // row-group index

#define tx threadIdx.x  // 0<=tx<Dx*Dz, where Dx=4 and Dz=4.
#define ty threadIdx.y  // 0<=ty<Dy, where Dy=4.
#define tz threadIdx.z  // 0<=tz<8, where tz is sibling-index of field cell

/* cutoff stands for the dimension of M-vector, L-vector, and
   K-matrix. This corresponds to 'r' in the paper and manual.  In this
   code, r is either 256 (high-precision version) or 32 (low-precision
   version) */
#define CUTOFF_H     256
#define LOG_CUTOFF_H   8
#define CUTOFF_L      32
#define LOG_CUTOFF_L   5

/* Set the number of rows per row-group. This parameter corresponds to
   'P' in the paper and manual */
#if !defined(NUM_ROW_GROUPS_IJ)
#define NUM_ROW_GROUPS_IJ 8 // 8 is better for C2050+SDK3.2
#endif
#if (NUM_ROW_GROUPS_IJ == 1)
#define NROWS_H 256 // cutoff=256
#define NROWS_L  32 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 2)
#define NROWS_H 128 // cutoff=256
#define NROWS_L  16 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 4)
#define NROWS_H  64 // cutoff=256
#define NROWS_L   8 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 8)
#define NROWS_H  32 // cutoff=256
#define NROWS_L   4 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 16)
#define NROWS_H  16 // cutoff=256
#define NROWS_L   2 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 32)
#define NROWS_H   8 // cutoff=256
#define NROWS_L   1 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 64)
#define NROWS_H   4 // cutoff=256
#define NROWS_L   0 // cutoff=32  IMPOSSIBLE
#else
#error Unsupposed NUM_ROW_GROUPS_IJ.
#endif

/* Macros to perform Li+=Kij*Mj for all the 316 Kij */
#if defined(K_IS_1X2_TEXTURE)
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij += Kijtmp.x * Mjtmp.x;				\
  Lij += Kijtmp.y * Mjtmp.y;
#else
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.x * Mjtmp;				\
  Lij.y += Kijtmp.y * Mjtmp;				\
  Lij.z += Kijtmp.z * Mjtmp;				\
  Lij.w += Kijtmp.w * Mjtmp
#endif
#define COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)

#if defined(K_IS_1X2_TEXTURE)
/* Declare the global variable of type texture to use float2 CUDA
   array (texture) that contains 316 K-matrices */
texture<float2, 3, cudaReadModeElementType> texRefK;
#elif defined(K_IS_4X1_TEXTURE)
/* Declare the global variable of type texture to use float4 CUDA
   array (texture) that contains 316 K-matrices */
texture<float4, 3, cudaReadModeElementType> texRefK;
#endif

#if defined(K_IS_1X2_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4 *K, real *M, int level, int Mstart) // real is double
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_1X2_TEXTURE)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_1X2_TEXTURE)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_1X2_TEXTURE)
  for (int j2 = 0; j2 < CUTOFF_H / 2; j2 ++) {
#else
  for (int j = 0; j < CUTOFF_H; j ++) {
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_1X2_TEXTURE)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_1X2_TEXTURE)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_H) + (NROWS_H * by)) << 9) + id;
#else
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + ((NROWS_H * by) >> 2)) << 9) + id;
#endif

    /* Set row index for unrolling x4 */
#if defined(K_IS_1X2_TEXTURE)
    int i = NROWS_H * by;
#else
    int i4 = ((NROWS_H * by) >> 2);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if defined(K_IS_1X2_TEXTURE)
#if (NROWS_H != 1)
    for (int iloc = 0; iloc < NROWS_H; iloc ++)
#endif
#else
      //110304#if (NROWS_H != 4)
#if (NROWS_H != 1)
    for (int iloc = 0; iloc < NROWS_H; iloc += 4)
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#else
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_H / 4) + i4) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#else
      /* Advance row index from i to i+4 (from i4 to i4+1) */
      i4 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#else
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#else
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#else
      real4 Kijtmp;
      real Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#else
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}


#if defined(K_IS_1X2_TEXTURE) 
__global__ void m2l_kern_ij_blocking_r32b4(real *L, real2 *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1_TEXTURE)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real *M, int level, int Mstart) // real is float
#elif defined(K_IS_4X1)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4 *K, real *M, int level, int Mstart) // real is double
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
#if defined(K_IS_1X2_TEXTURE)
  real2 *Mptr;
#else
  real *Mptr;
#endif
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

#if defined(K_IS_1X2_TEXTURE)
    /* M[level][j2=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + (Mstart / 2) + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#else
    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
#endif
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
#if defined(K_IS_1X2_TEXTURE)
  for (int j2 = 0; j2 < CUTOFF_L / 2; j2 ++) {
#else
  for (int j = 0; j < CUTOFF_L; j ++) {
#endif

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */
#if defined(K_IS_1X2_TEXTURE)
    __shared__ real2 Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#else
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
#endif

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
#if defined(K_IS_1X2_TEXTURE)
	real2 *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#else
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
#endif
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

#if defined(K_IS_1X2_TEXTURE)
    /* Set a pointer to L (L[chunk][row=NROWS*by][sib=tz][cell=id]) */
    real *Lptr = L + (((bx << LOG_CUTOFF_L) + (NROWS_L * by)) << 9) + id;
#else
    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + ((NROWS_L * by) >> 2)) << 9) + id;
#endif

    /* Set row index for unrolling x4 */
#if defined(K_IS_1X2_TEXTURE)
    int i = NROWS_L * by;
#else
    int i4 = ((NROWS_L * by) >> 2);
#endif

    /* Loop over local rows in the underlying by-th row-group */
#if defined(K_IS_1X2_TEXTURE)
#if (NROWS_L != 1)
    for (int iloc = 0; iloc < NROWS_L; iloc ++)
#endif
#else
      //110304#if (NROWS_L != 4)
#if (NROWS_L != 1)
    for (int iloc = 0; iloc < NROWS_L; iloc += 4)
#endif
#endif
    {
#if defined(K_IS_1X2_TEXTURE)
      __shared__ real2 Kij[316]; // Kij[z][y][x]
#else
      __shared__ real4 Kij[316]; // Kij[z][y][x]
#endif

      /* Load Kij */
      if (id < 316) {
#if defined(K_IS_1X2_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i, j2);
#elif defined(K_IS_4X1_TEXTURE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#elif defined(K_IS_4X1)
	Kij[id] = *(K + (j * (CUTOFF_L / 4) + i4) * 316 + id);
#endif
      }

#if defined(K_IS_1X2_TEXTURE)
      /* Advance row index from i to i+1 */
      i ++;
#else
      /* Advance row index from i to i+4 (from i4 to i4+1) */
      i4 ++;
#endif

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
#if defined(K_IS_1X2_TEXTURE)
      real Lij = ZERO;
#else
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);
#endif

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
#if defined(K_IS_1X2_TEXTURE)
      real2 *Kijptr = (real2 *)Kij;
      real2 *Mjptr = (real2 *)Mj + Mjoff;
#else
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;
#endif

      /* Perform different computaions according to sibling-index */
#if defined(K_IS_1X2_TEXTURE)
      real2 Kijtmp;
      real2 Mjtmp;
#else
      real4 Kijtmp;
      real Mjtmp;
#endif
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
#if defined(K_IS_1X2_TEXTURE)
      *Lptr += Lij;
#else
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;
#endif

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}
/**************************************************************************/
#elif defined(CUDA_VER45B)
/**************************************************************************/
#error This version is obsolete.

#include "real.h"

/* In general, the symbols Dx, Dy, and Dz defines the size of chunk,
   that is, each chunk consists of Dx*Dy*Dz clusters. In this code,
   Dx=Dy=Dz=4 is assumed and this number corresponds to 'B' in the
   paper and manual */

#define bx blockIdx.x   // chunk index
#define by blockIdx.y   // row-group index

#define tx threadIdx.x  // 0<=tx<Dx*Dz, where Dx=4 and Dz=4.
#define ty threadIdx.y  // 0<=ty<Dy, where Dy=4.
#define tz threadIdx.z  // 0<=tz<8, where tz is sibling-index of field cell

/* cutoff stands for the dimension of M-vector, L-vector, and
   K-matrix. This corresponds to 'r' in the paper and manual.  In this
   code, r is either 256 (high-precision version) or 32 (low-precision
   version) */
#define CUTOFF_H     256
#define LOG_CUTOFF_H   8
#define CUTOFF_L      32
#define LOG_CUTOFF_L   5

/* Set the number of rows per row-group. This parameter corresponds to
   'P' in the paper and manual */
#if !defined(NUM_ROW_GROUPS_IJ)
#define NUM_ROW_GROUPS_IJ 8 // 8 is better for C2050+SDK3.2
#endif
#if (NUM_ROW_GROUPS_IJ == 1)
#define NROWS_H 256 // cutoff=256
#define NROWS_L  32 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 2)
#define NROWS_H 128 // cutoff=256
#define NROWS_L  16 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 4)
#define NROWS_H  64 // cutoff=256
#define NROWS_L   8 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 8)
#define NROWS_H  32 // cutoff=256
#define NROWS_L   4 // cutoff=32
#elif (NUM_ROW_GROUPS_IJ == 16)
#define NROWS_H  16 // cutoff=256
#define NROWS_L   2 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 32)
#define NROWS_H   8 // cutoff=256
#define NROWS_L   1 // cutoff=32  IMPOSSIBLE
#elif (NUM_ROW_GROUPS_IJ == 64)
#define NROWS_H   4 // cutoff=256
#define NROWS_L   0 // cutoff=32  IMPOSSIBLE
#else
#error Unsupposed NUM_ROW_GROUPS_IJ.
#endif

/* Macros to perform Li+=Kij*Mj for all the 316 Kij */
#if(0) // same
#define COMP(Kijoff_diff, Mjoff_diff)				\
  Mjptr += Mjoff_diff;						\
  Kijptr += Kijoff_diff;					\
  Lij.x += (*Kijptr).x * *Mjptr;				\
  Lij.y += (*Kijptr).y * *Mjptr;				\
  Lij.z += (*Kijptr).z * *Mjptr;				\
  Lij.w += (*Kijptr).w * *Mjptr
#else
#define COMP(Kijoff_diff, Mjoff_diff)			\
  Mjptr += Mjoff_diff;					\
  Mjtmp = *Mjptr;					\
  Kijptr += Kijoff_diff;				\
  Kijtmp = *Kijptr;					\
  Lij.x += Kijtmp.x * Mjtmp;				\
  Lij.y += Kijtmp.y * Mjtmp;				\
  Lij.z += Kijtmp.z * Mjtmp;				\
  Lij.w += Kijtmp.w * Mjtmp
#endif
#define COMPXYZ0() COMP(57, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ1() COMP(8, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ2() COMP(50, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ3() COMP(1, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 2); COMP(1, 864); COMP(2, -1292); COMP(1, 2); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ4() COMP(56, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ5() COMP(7, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ6() COMP(49, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)
#define COMPXYZ7() COMP(0, 0); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1490); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, 2); COMP(2, -1292); COMP(1, 864); COMP(1, 2); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(9, -1094); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -1292); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(2, -434); COMP(1, 864); COMP(1, -863); COMP(1, 864); COMP(1, -863); COMP(1, 864)

#if defined(SINGLE)
/* Declare the global variable of type texture to use float4 CUDA
   array (texture) that contains 316 K-matrices */
texture<float4, 3, cudaReadModeElementType> texRefK;
#endif

//110228__global__ void m2l_kern_ij_blocking_r256b4(float4 *L, float *M, int level, int Mstart)
#if defined(SINGLE)
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real *M, int level, int Mstart)
#else
__global__ void m2l_kern_ij_blocking_r256b4(real4 *L, real4 *K, real *M, int level, int Mstart)
//__global__ void m2l_kern_ij_blocking_r256b4(real4 __restrict__ *L, real4 __restrict__ *K, real __restrict__ *M, int level, int Mstart)
#endif
{
  /* Read the index of the underlying level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
  //110228  float *Mptr;
  real *Mptr;
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
  //#pragma unroll 1
  for (int j = 0; j < CUTOFF_H; j ++) {

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */

    //110228    __shared__ float Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
	//110228	float *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    //110228    float4 *Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + ((NROWS_H * by) >> 2)) << 9) + id;
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_H - 2)) + ((NROWS_H * by) >> 2)) << 9) + id;

    /* Set row index for unrolling x4 */
    int i4 = ((NROWS_H * by) >> 2);

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_H != 4)
    //#pragma unroll 1
    for (int iloc = 0; iloc < NROWS_H; iloc += 4)
#endif
    {
      //110228      __shared__ float4 Kij[316]; // Kij[z][y][x]
      __shared__ real4 Kij[316]; // Kij[z][y][x]

      /* Load Kij */
      if (id < 316) {
#if defined(SINGLE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#else
	Kij[id] = *(K + (j * (CUTOFF_H / 4) + i4) * 316 + id);
#endif
      }

      /* Advance row index from i to i+4 (from i4 to i4+1) */
      i4 ++;

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
      //110228      float4 Lij = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
      //110228      float4 *Kijptr = (float4 *)Kij;
      //110228      float *Mjptr = (float *)Mj + Mjoff;
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;

      /* Perform different computaions according to sibling-index */
      //110228      float4 Kijtmp;
      //110228      float Mjtmp;
      real4 Kijtmp;
      real Mjtmp;
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
      //110228      float4 Ltmp = *Lptr;
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
}


//110228__global__ void m2l_kern_ij_blocking_r32b4(float4 *L, float *M, int level, int Mstart)
#if defined(SINGLE)
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real *M, int level, int Mstart)
#else
__global__ void m2l_kern_ij_blocking_r32b4(real4 *L, real4 *K, real *M, int level, int Mstart)
#endif
{
  /* Read the index of level */
  int lev = level;

  /* Number of cells (including two ghost cells) with the same
     sibling-index per direction for this level */
  int ncpec = POW2(lev - 1) + 2; // 2^{l-1}+2

  /* Set a pointer to M */
  //110228  float *Mptr;
  real *Mptr;
  {
    /* Compute the coordinates (cx,cy,cz) of the chunk;
       0<=cx<2^l/(2*Dx), 0<=cy<2^l/(2*Dy), 0<=cz<2^l/(2*Dz) */
    int cx = bx & (POW2(lev - 3) - 1);                  // bx%(2^l/(2*Dx))
    int cy = ((bx & (POW4(lev - 3) - 1)) >> (lev - 3)); // (bx%(2^l/(2*Dx)*2^l/(2*Dy)))/(2^l/(2*Dx))
    int cz = (bx >> ((lev << 1) - 6));                  // bx/(2^l/(2*Dx)*2^l/(2*Dy))

    /* M[level][j=0][sib=tz][cell=(Dx*cx,Dy*cy,Dz*cz)+(ix=0,iy=0,iz=0)] */
    Mptr = M + Mstart + (((0 * 8 + tz) * ncpec + (cz << 2)) * ncpec + (cy << 2)) * ncpec + (cx << 2);
  }

  /* Compute the offset to Mj */
  int Mjoff;
  {
    int hx = (tx & 3);              // tx%Dx
    int hy = ty;                    // ty
    int hz = (tx >> 2);             // tx/Dx
    Mjoff = hx + 6 * (hy + 6 * hz); // hx+(Dx+2)*(hy+(Dy+2)*hz)
  }

  /* Compute the unique cell index */
  int id = (((tz << 2) + ty) << 4) + tx; // 0<=id<=(tz*blockDim.y+ty)*blockDim.x+tx
  
  /* Loop over columns j */
  for (int j = 0; j < CUTOFF_L; j ++) {

    /* Load Mj of (2*Dx+4)*(2*Dy+4)*(2*Dz+4)(=12x12x12=1728) source
       cells in/around this chunk. Those cells are classified by their
       sibling-indices. */

    //110228    __shared__ float Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]
    __shared__ real Mj[8][6][6][6]; // Mj[8][Dz+2][Dy+2][Dx+2]

    {
      int ncpec2 = ncpec * ncpec;
      int wid = (ty >> 1);         // 0, 0, 1, 1 for ty=0, 1, 2, 3
      int zsta = 3 * wid;          // 0, 0, 3, 3 for ty=0, 1, 2, 3
      int ysta = 3 * ty - 6 * wid; // 0, 3, 0, 3 for ty=0, 1, 2, 3
      if (tx < 6) {
	//110228	float *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
	real *ptmp = Mptr + zsta * ncpec2 + ysta * ncpec + tx;
	Mj[tz][0 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][0 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][0 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][1 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][1 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][1 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
	ptmp += ncpec2;
	Mj[tz][2 + zsta][0 + ysta][tx] = *ptmp;
	Mj[tz][2 + zsta][1 + ysta][tx] = *(ptmp + ncpec);
	Mj[tz][2 + zsta][2 + ysta][tx] = *(ptmp + ncpec * 2);
      }
    }
    
    /* Advance Mptr to the next j */
    Mptr += 8 * ncpec * ncpec * ncpec;

    /* Set a pointer to L (L[chunk][row=NROWS*by/4][sib=tz][cell=id]) */
    //110228    float4 *Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + ((NROWS_L * by) >> 2)) << 9) + id;
    real4 *Lptr = L + (((bx << (LOG_CUTOFF_L - 2)) + ((NROWS_L * by) >> 2)) << 9) + id;

    /* Set row index for unrolling x4 */
    int i4 = ((NROWS_L * by) >> 2);

    /* Loop over local rows in the underlying by-th row-group */
#if (NROWS_L != 4)
    for (int iloc = 0; iloc < NROWS_L; iloc += 4)
#endif
    {
      //110228      __shared__ float4 Kij[316]; // Kij[z][y][x]
      __shared__ real4 Kij[316]; // Kij[z][y][x]

      /* Load Kij */
      if (id < 316) {
#if defined(SINGLE)
	Kij[id] = tex3D(texRefK, id, i4, j);
#else
	Kij[id] = *(K + (j * (CUTOFF_L / 4) + i4) * 316 + id);
#endif
      }

      /* Advance row index from i to i+4 (from i4 to i4+1) */
      i4 ++;

      /* Ensure that Kij (and Mj if i=0) was loaded */
      __syncthreads();

      /* Initialise Lij(F) */
      //110228      float4 Lij = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      real4 Lij = make_real4(ZERO, ZERO, ZERO, ZERO);

      /* Compute Lij(F)+=\sum_{S}Kij(F,S)*Mj(S) (reduction for S) */
      //110228      float4 *Kijptr = (float4 *)Kij;
      //110228      float *Mjptr = (float *)Mj + Mjoff;
      real4 *Kijptr = (real4 *)Kij;
      real *Mjptr = (real *)Mj + Mjoff;

      /* Perform different computaions according to sibling-index */
      //110228      float4 Kijtmp;
      //110228      float Mjtmp;
      real4 Kijtmp;
      real Mjtmp;
      if (tz == 0) {
	COMPXYZ0();
      }	else if (tz == 1) {
	COMPXYZ1();
      }	else if (tz == 2) {
	COMPXYZ2();
      }	else if (tz == 3) {
	COMPXYZ3();
      }	else if (tz == 4) {
	COMPXYZ4();
      }	else if (tz == 5) {
	COMPXYZ5();
      }	else if (tz == 6) {
	COMPXYZ6();
      }	else if (tz == 7) {
	COMPXYZ7();
      }
	
      /* Accumulate Lij(F) to Li(F) (reduction for j) */
      //110228      float4 Ltmp = *Lptr;
      real4 Ltmp = *Lptr;
      Ltmp.x += Lij.x;
      Ltmp.y += Lij.y;
      Ltmp.z += Lij.z;
      Ltmp.w += Lij.w;
      *Lptr = Ltmp;

      /* Advance Lptr from i to i+4 */
      Lptr += 512; // (2*Dx)*(2*Dy)*(2*Dz)
      
      /* Ensure that Kij (and Mj if i=cutoff-1) is no longer used */
      __syncthreads();

    } /* i */
  } /* j */
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
#endif /* M2L_KERN_IJ_BLOCKING_CU */
