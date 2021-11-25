#ifndef DEBUGMACROS_CUDA_H
#define DEBUGMACROS_CUDA_H

#include <assert.h>

#include "cutil.h"

#ifndef DEVICE_ID
#define DEVICE_ID (0) /* Default device ID is zero */
#endif

#if defined(_DEBUG) || defined(MYDEBUG)
#define CHECK_CONFIGURATION(grid, block)				\
  {									\
    cudaDeviceProp deviceProp;						\
    cudaGetDeviceProperties(&deviceProp, DEVICE_ID);			\
    assert(grid.x <= deviceProp.maxGridSize[0] &&			\
	   grid.y <= deviceProp.maxGridSize[1] &&			\
	   grid.z <= deviceProp.maxGridSize[2] &&			\
	   block.x <= deviceProp.maxThreadsDim[0] &&			\
	   block.y <= deviceProp.maxThreadsDim[1] &&			\
	   block.z <= deviceProp.maxThreadsDim[2] &&			\
	   block.x * block.y * block.z <= deviceProp.maxThreadsPerBlock); \
  }
#else
#define CHECK_CONFIGURATION(grid, block) {}
#endif

#endif /* DEBUGMACROS_CUDA_H */
