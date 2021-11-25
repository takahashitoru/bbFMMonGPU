#ifndef DEBUGMACROS_H
#define DEBUGMACROS_H

#ifdef QUIET

#define INFO(fmt, ...)
#define MESG(message)
#define DBG(fmt, ...)
#define MSG(message)
#define ASSERT(s)

#else

#define INFO(fmt, ...) fprintf(stderr, "# %s: " fmt, __FUNCTION__, __VA_ARGS__)
#define MESG(message) fprintf(stderr, "# %s: %s", __FUNCTION__, message)

#if defined(_DEBUG) || defined(MYDEBUG)

#define DBG(fmt, ...) fprintf(stderr, "### %s: " fmt, __FUNCTION__, __VA_ARGS__)
#define MSG(message) fprintf(stderr, "### %s: %s", __FUNCTION__, message)
#include <assert.h>
#define ASSERT(s) assert(s)

#else

#define DBG(fmt, ...)
#define MSG(message)
#define ASSERT(s)

#endif
#endif

#endif /* DEBUGMACROS_H */
