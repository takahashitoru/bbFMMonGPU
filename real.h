#ifndef REAL_H
#define REAL_H

#include "vec234.h"

#ifdef SINGLE

typedef float real;

#define agesvd_ sgesvd_
#define agemm_ sgemm_
#define agemv_ sgemv_
#define adot_ sdot_
#define acopy_ scopy_

#define SIN(x) sinf(x)
#define COS(x) cosf(x)
#define EXP(x) expf(x)
#define SQRT(x) sqrtf(x)
#define POW(x, y) powf(x, y)
#define ERFC(x) erfcf(x)
#define ABS(x) fabsf(x)
#define FLOOR(x) floorf(x)
#define RINT(x) rintf(x)
#define RSQRT(x) rsqrtf(x)

#ifdef __ICC
#define INVSQRT(x) invsqrtf(x)
#else
#define INVSQRT(x) (1.0f / sqrtf(x))
#endif

typedef float2  real2;
typedef float3  real3;
typedef float4  real4;
typedef float8  real8;
typedef float16 real16;
typedef float8x2 real8x2;
typedef float4x4 real4x4;
typedef float4x2 real4x2;

#define make_real2(x, y) make_float2(x, y)
#define make_real3(x, y, z) make_float3(x, y, z)
#define make_real4(x, y, z, w) make_float4(x, y, z, w)

#define DIVIDE(x, y) fdividef(x, y)

#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
#define THREE 3.0f
#define FOUR 4.0f

#else /* !SINGLE */

typedef double real;

#define agesvd_ dgesvd_
#define agemm_ dgemm_
#define agemv_ dgemv_
#define adot_ ddot_
#define acopy_ dcopy_

#define SIN(x) sin(x)
#define COS(x) cos(x)
#define EXP(x) exp(x)
#define SQRT(x) sqrt(x)
#define POW(x, y) pow(x, y)
#define ERFC(x) erfc(x)
#define ABS(x) fabs(x)
#define FLOOR(x) floor(x)
#define RINT(x) rint(x)
#define RSQRT(x) rsqrt(x)

#ifdef __ICC
#define INVSQRT(x) invsqrt(x)
#else
#define INVSQRT(x) (1.0 / sqrt(x))
#endif

typedef double2  real2;
typedef double3  real3;
typedef double4  real4;
typedef double8  real8;
typedef double16 real16;
typedef double8x2 real8x2;
typedef double4x4 real4x4;
typedef double4x2 real4x2;

#define make_real2(x, y) make_double2(x, y)
#define make_real3(x, y, z) make_double3(x, y, z)
#define make_real4(x, y, z, w) make_double4(x, y, z, w)

#define DIVIDE(x, y) ((x) / (y))

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define THREE 3.0
#define FOUR 4.0

#endif /* !SINGLE */

#endif /* REAL_H */
