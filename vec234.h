#ifndef VEC234_H
#define VEC234_H

/* Define double{2,3,4} and float{2,3,4} for host */

#if !defined(__CUDACC__)

#if(0) // load header files in CUDA Tool kit

#include <host_defines.h>
#include <vector_types.h>

#else // read the relevant codes only

/* cuda/include/host_defines.h */
#define __align__(n) __attribute__((aligned(n)))
/* cuda/include/vector_types.h */
typedef struct _double2 { double x, y; } __align__(16) double2;
typedef struct _double3 { double x, y, z; } double3;
typedef struct _double4 { double x, y, z, w; } __align__(16) double4;

typedef struct _float2 { float x, y; } __align__(8) float2;
typedef struct _float3 { float x, y, z; } float3;
typedef struct _float4 { float x, y, z, w; } __align__(16) float4;

#endif

#endif /* !defined(__CUDACC__) */


/* Define longer vector types */

#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif

typedef struct _double8 { double a, b, c, d, e, f, g, h; } __align__(16) double8;
typedef struct _double16 { double a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p; } __align__(16) double16;
#if defined(VEC234_COLUMN_MAJOR)
typedef struct _double8x2 { double aa, ba, ca, da, ea, fa, ga, ha, ab, bb, cb, db, eb, fb, gb, hb; } __align__(16) double8x2;
typedef struct _double4x4 { double xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww; } __align__(16) double4x4;
typedef struct _double4x2 { double xx, yx, zx, wx, xy, yy, zy, wy; } __align__(16) double4x2;
#else // row major
typedef struct _double8x2 { double aa, ab, ba, bb, ca, cb, da, db, ea, eb, fa, fb, ga, gb, ha, hb; } __align__(16) double8x2;
typedef struct _double4x4 { double xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw, wx, wy, wz, ww; } __align__(16) double4x4;
typedef struct _double4x2 { double xx, xy, yx, yy, zx, zy, wx, wy; } __align__(16) double4x2;
#endif

typedef struct _float8 { float a, b, c, d, e, f, g, h; } __align__(16) float8;
typedef struct _float16 { float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p; } __align__(16) float16;
#if defined(VEC234_COLUMN_MAJOR)
typedef struct _float8x2 { float aa, ba, ca, da, ea, fa, ga, ha, ab, bb, cb, db, eb, fb, gb, hb; } __align__(16) float8x2;
typedef struct _float4x4 { float xx, yx, zx, wx, xy, yy, zy, wy, xz, yz, zz, wz, xw, yw, zw, ww; } __align__(16) float4x4;
typedef struct _float4x2 { float xx, yx, zx, wx, xy, yy, zy, wy; } __align__(16) float4x2;
#else // row major
typedef struct _float8x2 { float aa, ab, ba, bb, ca, cb, da, db, ea, eb, fa, fb, ga, gb, ha, hb; } __align__(16) float8x2;
typedef struct _float4x4 { float xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw, wx, wy, wz, ww; } __align__(16) float4x4;
typedef struct _float4x2 { float xx, xy, yx, yy, zx, zy, wx, wy; } __align__(16) float4x2;
#endif

#endif /* VEC234_H */
