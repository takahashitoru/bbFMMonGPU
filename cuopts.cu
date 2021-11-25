#ifndef CUOPTS_CU
#define CUOPTS_CU

#include <cuda_runtime_api.h>

#include "opts.h"

__host__ void cuopts(void)
{
#ifdef __DEVICE_EMULATION__
  MESSAGE("__DEVICE_EMULATION__");
#endif

#if defined(CUDART_VERSION)
  PRINT("CUDART_VERSION = %d", CUDART_VERSION);
#endif

#ifdef _DEBUG
  MESSAGE("_DEBUG");
#endif

#ifdef DISABLE_TIMING
  MESSAGE("DISABLE_TIMING");
#endif

#ifdef CUDA
  MESSAGE("CUDA");
#endif

#ifdef SINGLE
  MESSAGE("SINGLE");
#else
  MESSAGE("SINGLE is not defined");
#endif


#ifdef CUDA_ARCH
  PRINT("CUDA_ARCH = %d", CUDA_ARCH);
#endif

#ifdef __CUDA_ARCH__ // This macro can be used in device codes only
  PRINT("__CUDA_ARCH__ = %d", __CUDA_ARCH__);
#endif


#ifdef DISABLE_INCREASE_CUTOFF_BY_ONE
  MESSAGE("DISABLE_INCREASE_CUTOFF_BY_ONE");
#endif

#ifdef LAPLACIAN
  MESSAGE("LAPLACIAN");
#elif LAPLACIANFORCE
  MESSAGE("LAPLACIANFORCE");
#elif ONEOVERR4
  MESSAGE("ONEOVERR4");
#else
  ERRMESG("Any kernel is not defined.");
#endif

#ifdef CHECK_PERFORMANCE
  MESSAGE("CHECK_PERFORMANCE");
#endif


#if defined(CUDA_VER46)
  MESSAGE("CUDA_VER46");
#if defined(CUDA_VER46A)
  MESSAGE("CUDA_VER46A");
#elif defined(CUDA_VER46B)
  MESSAGE("CUDA_VER46B");
#elif defined(CUDA_VER46C)
  MESSAGE("CUDA_VER46C");
#elif defined(CUDA_VER46D)
  MESSAGE("CUDA_VER46D");
#elif defined(CUDA_VER46E)
  MESSAGE("CUDA_VER46E");
#elif defined(CUDA_VER46F)
  MESSAGE("CUDA_VER46F");
#elif defined(CUDA_VER46G)
  MESSAGE("CUDA_VER46G");
#elif defined(CUDA_VER46H)
  MESSAGE("CUDA_VER46H");
#elif defined(CUDA_VER46I)
  MESSAGE("CUDA_VER46I");
#elif defined(CUDA_VER46J)
  MESSAGE("CUDA_VER46J");
#elif defined(CUDA_VER46K)
  MESSAGE("CUDA_VER46K");
#elif defined(CUDA_VER46L)
  MESSAGE("CUDA_VER46L");
#elif defined(CUDA_VER46M)
  MESSAGE("CUDA_VER46M");
#elif defined(CUDA_VER46N)
  MESSAGE("CUDA_VER46N");
#elif defined(CUDA_VER46O)
  MESSAGE("CUDA_VER46O");
#elif defined(CUDA_VER46P)
  MESSAGE("CUDA_VER46P");
#endif
#endif /* CUDA_VER46 */


#ifdef _OPENMP
  MESSAGE("_OPENMP");
#endif

#ifdef PBC
  MESSAGE("PBC");
#endif

#ifdef MYDEBUG
  MESSAGE("MYDEBUG");
#endif

#if defined(FAST_HOST_CODE)
  MESSAGE("FAST_HOST_CODE");
#endif

#if defined(DEVICE_QUERY)
  MESSAGE("DEVICE_QUERY");
#endif

#ifdef ENABLE_NEARFIELD_BY_CPU
  MESSAGE("ENABLE_NEARFIELD_BY_CPU");
#endif

#ifdef SCHEME
  PRINT("SCHEME = %d", SCHEME);
#endif

#ifdef DIRECT_GROUP_SIZE
  PRINT("DIRECT_GROUP_SIZE = %d", DIRECT_GROUP_SIZE);
#endif

#ifdef DIRECT_NUM_THREADS_PER_BLOCK
  PRINT("DIRECT_NUM_THREADS_PER_BLOCK = %d", DIRECT_NUM_THREADS_PER_BLOCK);
#endif

#if defined(NUM_ROW_GROUPS_IJ)
  PRINT("NUM_ROW_GROUPS_IJ = %d", NUM_ROW_GROUPS_IJ);
#endif

#ifdef LEVEL_SWITCH_IJ
  PRINT("LEVEL_SWITCH_IJ = %d", LEVEL_SWITCH_IJ);
#endif

}

#endif /* CUOPTS_CU */
