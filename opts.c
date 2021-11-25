#include "opts.h"

void opts(void)
{
#ifdef __DATE__
  SOP("__DATE__", __DATE__);
#endif
#ifdef __TIME__
  SOP("__TIME__", __TIME__);
#endif

#ifdef __linux__
  MESSAGE("__linux__");
#endif

#ifdef __i386__
  MESSAGE("__i386__");
#endif
#ifdef __x86_64__
  MESSAGE("__x86_64__");
#endif

#ifdef __SSE__
  MESSAGE("__SSE__");
#endif
#ifdef __SSE2__
  MESSAGE("__SSE2__");
#endif
#ifdef __SSE3__
  MESSAGE("__SSE3__");
#endif
#ifdef __SSSE3__
  MESSAGE("__SSSE3__");
#endif

#ifdef __ICC
  IOP("__ICC", __ICC);
#endif
#ifdef __INTEL_COMPILER_BUILD_DATE
  IOP("__INTEL_COMPILER_BUILD_DATE", __INTEL_COMPILER_BUILD_DATE);
#endif

#ifdef __GNUC__
  IOP("__GNUC__", __GNUC__);
#endif
#ifdef __GNUC_MINOR__
  IOP("__GNUC_MINOR__", __GNUC_MINOR__);
#endif
#ifdef __GNUC_PATCHLEVEL__
  IOP("__GNUC_PATCHLEVEL__", __GNUC_PATCHLEVEL__);
#endif

#ifdef _OPENMP
  MESSAGE("_OPENMP");
#endif

#ifdef SINGLE
  MESSAGE("SINGLE");
#else
  MESSAGE("SINGLE is not defined");
#endif

#ifdef ENABLE_DIRECT
  MESSAGE("ENABLE_DIRECT");
#endif

#ifdef MYDEBUG
  MESSAGE("MYDEBUG");
#endif

#ifdef DISABLE_TIMING
  MESSAGE("DISABLE_TIMING");
#endif

#ifdef DISABLE_INCREASE_CUTOFF_BY_ONE
  MESSAGE("DISABLE_INCREASE_CUTOFF_BY_ONE");
#endif

#ifdef QUIET
  MESSAGE("QUIET");
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

#ifdef DISABLE_ALIGN
  MESSAGE("DISABLE_ALIGN");
#endif

#ifdef ALIGN_SIZE
  PRINT("ALIGN_SIZE = %d", ALIGN_SIZE);
#endif

#ifdef MINLEV
  PRINT("MINLEV = %d", MINLEV);
#endif

#ifdef CHECK_PERFORMANCE
  MESSAGE("CHECK_PERFORMANCE");
#endif

#ifdef ENABLE_GETTIMEOFDAY
  MESSAGE("ENABLE_GETTIMEOFDAY");
#endif

#ifdef ENABLE_ASCII_IO
  MESSAGE("ENABLE_ASCII_IO");
#endif

#ifdef PBC
  MESSAGE("PBC");
#endif

#ifdef FIELDPOINTS_EQ_SOURCES
  MESSAGE("FIELDPOINTS_EQ_SOURCES");
#endif

#if defined(CPU8)
  MESSAGE("CPU8");
#if defined(CPU8A)
  MESSAGE("CPU8A");
#elif defined(CPU8B)
  MESSAGE("CPU8B");
#endif

#elif defined(CPU9)
  MESSAGE("CPU9");
#if defined(CPU9A)
  MESSAGE("CPU9A");
#elif defined(CPU9B)
  MESSAGE("CPU9B");
#elif defined(CPU9C)
  MESSAGE("CPU9C");
#elif defined(CPU9D)
  MESSAGE("CPU9D");
#elif defined(CPU9E)
  MESSAGE("CPU9E");
#elif defined(CPU9F)
  MESSAGE("CPU9F");
#elif defined(CPU9G)
  MESSAGE("CPU9G");
#elif defined(CPU9H)
  MESSAGE("CPU9H");
#elif defined(CPU9I)
  MESSAGE("CPU9I");
#elif defined(CPU9J)
  MESSAGE("CPU9J");
#elif defined(CPU9K)
  MESSAGE("CPU9K");
#elif defined(CPU9L)
  MESSAGE("CPU9L");
#elif defined(CPU9M)
  MESSAGE("CPU9M");
#elif defined(CPU9N)
  MESSAGE("CPU9N");
#elif defined(CPU9O)
  MESSAGE("CPU9O");
#elif defined(CPU9P)
  MESSAGE("CPU9P");
#elif defined(CPU9Q)
  MESSAGE("CPU9Q");
#elif defined(CPU9R)
  MESSAGE("CPU9R");
#elif defined(CPU9S)
  MESSAGE("CPU9S");
#elif defined(CPU9T)
  MESSAGE("CPU9T");
#elif defined(CPU9U)
  MESSAGE("CPU9U");
#endif
#endif

#ifdef RANDOM_SEED
  MESSAGE("RANDOM_SEED");
#endif
}
