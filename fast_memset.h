#ifndef FAST_MEMSET_H
#define FAST_MEMSET_H

#ifdef __INTEL_COMPILER

// It seems that _intel_fast_memset is archived in librc.a for ver12.0 accoring to
// nm --print-file-name /opt/intel/composerxe/lib/intel64/*.a | grep _intel_fast_memset | grep T

#ifdef __cplusplus
extern "C" {
#endif
  void *_intel_fast_memset(void *s, int c, size_t n);
#ifdef __cplusplus
}
#endif
#define fast_memset(s, c, n) ( _intel_fast_memset((s), (c), (n)) )

#else /* other compilers */

#include <string.h>
#define fast_memset(s, c, n) ( memset((s), (c), (n)) )

#endif

#endif /* FAST_MEMSET_H */
