#ifndef M2L_HOST_H
#define M2L_HOST_H

/* Functions for CUDA event timer */
#include "eventTimer.h"

/* Macro to print to stderr */
#ifndef INFO
#define INFO(fmt, ...) fprintf(stderr, "# %s: " fmt, __FUNCTION__, __VA_ARGS__)
#endif

/* Macro for debugging */
#ifdef _DEBUG
#include <cutil.h>
#define CSC(call) CUDA_SAFE_CALL(call)
#else
#define CSC(call) call
#endif /* _DEBUG */

/* Auxiliary functions */
#include "m2l_aux.h"

/* Global variables to store the kernel execution time */
double kernel_exec_time;

#endif /* M2L_HOST_H */
