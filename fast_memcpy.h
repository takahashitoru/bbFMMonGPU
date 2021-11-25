#ifndef FAST_MEMCPY_H
#define FAST_MEMCPY_H

#ifdef __INTEL_COMPILER

// It seems that _intel_fast_memcpy is archived in librc.a for ver12.0 accoring to
// nm --print-file-name /opt/intel/composerxe/lib/intel64/*.a | grep _intel_fast_memcpy | grep T

#ifdef __cplusplus
extern "C" {
#endif
  void *_intel_fast_memcpy(void *dest, const void *src, size_t n);
#ifdef __cplusplus
}
#endif
#define fast_memcpy(dest, src, n) ( _intel_fast_memcpy((dest), (src), (n)) )

#else /* other compilers */

#include <string.h>
#define fast_memcpy(dest, src, n) ( memcpy((dest), (src), (n)) )

#endif

#endif /* FAST_MEMCPY_H */
