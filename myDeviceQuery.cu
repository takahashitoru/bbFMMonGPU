#ifndef MYDEVICEQUERY_CU
#define MYDEVICEQUERY_CU

/*
  This file is created from deviceQuery.cpp.  This sample queries the
  properties of the CUDA devices present in the system via CUDA
  Runtime API.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

// includes, project
#include <cutil.h>

void myDeviceQuery()
{
  int deviceCount;

  cudaGetDeviceCount(&deviceCount);

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    MESG("There is no device supporting CUDA.\n");
  }

  for (int dev = 0; dev < deviceCount; dev ++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if (dev == 0) {
      // This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
	MESG("There is no device supporting CUDA.\n");
      } else if (deviceCount == 1) {
	MESG("There is 1 device supporting CUDA.\n");
      }	else {
	INFO("There are %d devices supporting CUDA.\n", deviceCount);
      }
      INFO("Device %d: \"%s\"\n", dev, deviceProp.name);
      INFO("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
      INFO("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
      INFO("  Total amount of global memory:                 %u bytes\n", deviceProp.totalGlobalMem);
#if CUDART_VERSION >= 2000
      INFO("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
      INFO("  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
#endif
      INFO("  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem); 
      INFO("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
      INFO("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
      INFO("  Warp size:                                     %d\n", deviceProp.warpSize);
      INFO("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
      INFO("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
	   deviceProp.maxThreadsDim[0],
	   deviceProp.maxThreadsDim[1],
	   deviceProp.maxThreadsDim[2]);
      INFO("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
	   deviceProp.maxGridSize[0],
	   deviceProp.maxGridSize[1],
	   deviceProp.maxGridSize[2]);
      INFO("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
      INFO("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
      INFO("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
      INFO("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 2020
      INFO("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
      INFO("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
      INFO("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
      INFO("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
	   "Default (multiple host threads can use this device simultaneously)" :
	   deviceProp.computeMode == cudaComputeModeExclusive ?
	   "Exclusive (only one host thread at a time can use this device)" :
	   deviceProp.computeMode == cudaComputeModeProhibited ?
	   "Prohibited (no host thread can use this device)" :
	   "Unknown");
#endif
    }
  } /* dev */
}

#endif /* MYDEVICEQUERY_CU */
