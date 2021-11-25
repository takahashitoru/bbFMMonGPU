#ifndef M2L_HOST_IJ_BLOCKING_CU
#define M2L_HOST_IJ_BLOCKING_CU

/* Include header file */
#include "m2l_host.h"

/* Determine the type of K-matrix (thus, M- and L-vectors) due to
   precision and compute capability in order to realize the bank-free
   access for K-matrix */
////////////////////////////////////////////////////////////////////////
#if defined(CUDA_VER45I)
////////////////////////////////////////////////////////////////////////
#if defined(USE_ANY_R)
#undef CHUNK_SIZE_IJ
//#define CHUNK_SIZE_IJ 4
#define CHUNK_SIZE_IJ 2
#define K_IS_1X1
#else
#if (CUDA_ARCH >= 20) && (CUDA_ARCH < 30) // shared memory bank is 32
#if defined(SINGLE)
#define K_IS_8X2 // best for C2050+SDK3.2
#else // double precision
#define K_IS_8X1 // best for C2050+SDK3.2
#endif

#elif (CUDA_ARCH >= 10) && (CUDA_ARCH < 20) // shared memory bank is 16
#if defined(SINGLE)
#define K_IS_4X1_TEXTURE
#else // double precision, for which CUDA array is unavailable
#define K_IS_4X1
#endif

#else // other architecures
#error Not implemented yet.
#endif
#endif
////////////////////////////////////////////////////////////////////////
#elif defined(CUDA_VER45H)
////////////////////////////////////////////////////////////////////////
#if (CUDA_ARCH >= 20) && (CUDA_ARCH < 30) // shared memory bank is 32
#if defined(SINGLE)
#define K_IS_4X2
//#define K_IS_8X2 // best for C2050+SDK3.2
#else // double precision
#define K_IS_8X1 // best for C2050+SDK3.2
#endif

#elif (CUDA_ARCH >= 10) && (CUDA_ARCH < 20) // shared memory bank is 16
#if defined(SINGLE)
#define K_IS_4X1_TEXTURE
#else // double precision, for which CUDA array is unavailable
#define K_IS_4X1
#endif

#else // other architecures
#error Not implemented yet.
#endif
////////////////////////////////////////////////////////////////////////
#elif defined(CUDA_VER45G)
////////////////////////////////////////////////////////////////////////
#if (CUDA_ARCH >= 20) && (CUDA_ARCH < 30) // shared memory bank is 32
#if defined(SINGLE)
//#define K_IS_4X4 // worse than 8X2
#define K_IS_8X2 // best for C2050+SDK3.2
//#define K_IS_16X1
//#define K_IS_8X1
//#define K_IS_1X2_TEXTURE // only for single precision
//#define K_IS_4X1_TEXTURE // only for single precision
#else // double precision
//#define K_IS_4X2 // worse than 8X1
#define K_IS_8X1 // best for C2050+SDK3.2
//#define K_IS_4X1
#endif

#elif (CUDA_ARCH >= 10) && (CUDA_ARCH < 20) // shared memory bank is 16
#if defined(SINGLE)
#define K_IS_4X1_TEXTURE
#else // double precision, for which CUDA array is unavailable
#define K_IS_4X1
#endif

#else // other architecures
#error Not implemented yet.
#endif
////////////////////////////////////////////////////////////////////////
#elif defined(CUDA_VER45F)
////////////////////////////////////////////////////////////////////////
#if (CUDA_ARCH >= 20) && (CUDA_ARCH < 30) // shared memory bank is 32
#if defined(SINGLE)
#define K_IS_8X2
#else // double precision, for which CUDA array is unavailable
#define K_IS_8X1
#endif
#elif (CUDA_ARCH >= 10) && (CUDA_ARCH < 20) // shared memory bank is 16
#if defined(SINGLE)
#define K_IS_4X1_TEXTURE
#else // double precision, for which CUDA array is unavailable
#define K_IS_4X1
#endif
#else // other architecures
#error Not implemented yet.
#endif
////////////////////////////////////////////////////////////////////////
#elif defined(CUDA_VER45E)
////////////////////////////////////////////////////////////////////////
#if defined(SINGLE)
#if (CUDA_ARCH >= 20) && (CUDA_ARCH < 30) // shared memory bank is 32
#define K_IS_16X1
#elif (CUDA_ARCH >= 10) && (CUDA_ARCH < 20) // shared memory bank is 16
#define K_IS_4X1_TEXTURE
#else
#error Not implemented yet.
#endif
#else // double precision
#define K_IS_4X1 // because CUDA array is undefined for double4
#endif
////////////////////////////////////////////////////////////////////////
#elif defined(CUDA_VER45D)
////////////////////////////////////////////////////////////////////////
#if defined(SINGLE)
#if (CUDA_ARCH >= 20) && (CUDA_ARCH < 30) // shared memory bank is 32
#define K_IS_8X1
#elif (CUDA_ARCH >= 10) && (CUDA_ARCH < 20) // shared memory bank is 16
#define K_IS_4X1_TEXTURE
#else
#error Not implemented yet.
#endif
#else // double precision
#define K_IS_4X1 // because CUDA array is undefined for double4
#endif
////////////////////////////////////////////////////////////////////////
#elif defined(CUDA_VER45C)
////////////////////////////////////////////////////////////////////////
#if defined(SINGLE)
#if (CUDA_ARCH >= 20) && (CUDA_ARCH < 30) // shared memory bank is 32
#define K_IS_1X2_TEXTURE
//#define K_IS_4X1_TEXTURE
#elif (CUDA_ARCH >= 10) && (CUDA_ARCH < 20) // shared memory bank is 16
#define K_IS_4X1_TEXTURE
#else
#error Not implemented yet.
#endif
#else // double precision
#define K_IS_4X1 // because CUDA array is undefined for double4
#endif
////////////////////////////////////////////////////////////////////////
#else
////////////////////////////////////////////////////////////////////////
#if defined(SINGLE)
#define K_IS_4X1_TEXTURE // equivalent to SINGLE
#else // double precision
#define K_IS_4X1 // because CUDA array is undefined for double4
#endif
////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////

/* Parameters for kernel */
#ifndef CHUNK_SIZE_IJ
#define CHUNK_SIZE_IJ 4 // Currently, only 4
#endif

/* Include kernel */
#include "m2l_kern_ij_blocking.cu"

__host__ void m2l_host_ij_blocking(int level_start, int level_end, int r, real *L, real *K, real *M,
				   int *Ktable, real3 *center, real L0)
{
#if defined(K_IS_1X2_TEXTURE)
  MSG("K_IS_1X2_TEXTURE is defined.\n");
#elif defined(K_IS_4X1_TEXTURE)
  MSG("K_IS_4X1_TEXTURE is defined.\n");
#elif defined(K_IS_4X1)
  MSG("K_IS_4X1 is defined.\n");
#elif defined(K_IS_8X1)
  MSG("K_IS_8X1 is defined.\n");
#elif defined(K_IS_16X1)
  MSG("K_IS_16X1 is defined.\n");
#elif defined(K_IS_8X2)
  MSG("K_IS_8X2 is defined.\n");
#elif defined(K_IS_4X4)
  MSG("K_IS_4X4 is defined.\n");
#elif defined(K_IS_4X2)
  MSG("K_IS_4X2 is defined.\n");
#elif defined(K_IS_1X1)
  MSG("K_IS_1X1 is defined.\n");
#endif

  /* Initialize timers */
  eventTimerType timer_m2l_all;
  initEventTimer(&timer_m2l_all);
  startEventTimer(&timer_m2l_all);
  
  eventTimerType timer_m2l_kernel, timer_m2l_set, timer_m2l_get;
  initEventTimer(&timer_m2l_kernel);
  initEventTimer(&timer_m2l_set);
  initEventTimer(&timer_m2l_get);
  
  /* Constants */
  int B = CHUNK_SIZE_IJ;
  int BBB8 = B * B * B * 8; // number of cells per chunk of size B

  /* Compute the indices of starting and final cells for each level */
  int *cellsta = (int *)malloc((level_end + 1) * sizeof(int)); // cellsta[0:level_end]
  int *cellend = (int *)malloc((level_end + 1) * sizeof(int)); // cellend[0:level_end]
  m2l_aux_comp_cellsta_cellend(cellsta, cellend, level_end);

  /* Compute the number of all the real and ghost cells */
  int ncellanother = m2l_aux_get_number_of_real_and_ghost_cells_for_ij_blocking(level_start, level_end);

  /* Allocate and initialize another M-vector for both real and ghost
     cells */
#if defined(K_IS_4X4)
  real4 *Manother = (real4 *)calloc(ncellanother * (r / 4), sizeof(real4));
  m2l_aux_convert_M_to_Manother_for_ij_blocking_col4(r, level_start, level_end, center, L0, M, Manother);
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  real2 *Manother = (real2 *)calloc(ncellanother * (r / 2), sizeof(real2));
  m2l_aux_convert_M_to_Manother_for_ij_blocking_col2(r, level_start, level_end, center, L0, M, Manother);
#else
  real *Manother = (real *)calloc(ncellanother * r, sizeof(real));
  m2l_aux_convert_M_to_Manother_for_ij_blocking_col1(r, level_start, level_end, center, L0, M, Manother);
#endif

  /* Allocate another M-vector on device */
#if defined(K_IS_4X4)
  real4 *d_Manother;
  CSC(cudaMalloc((void **)&d_Manother, ncellanother * (r / 4) * sizeof(real4)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Manother, Manother, ncellanother * (r / 4) * sizeof(real4), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);
#elif defined(K_IS_1X2_TEXTURE) || defined(K_IS_8X2) || defined(K_IS_4X2)
  real2 *d_Manother;
  CSC(cudaMalloc((void **)&d_Manother, ncellanother * (r / 2) * sizeof(real2)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Manother, Manother, ncellanother * (r / 2) * sizeof(real2), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);
#else
  real *d_Manother;
  CSC(cudaMalloc((void **)&d_Manother, ncellanother * r * sizeof(real)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Manother, Manother, ncellanother * r * sizeof(real), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);
#endif

  /* Allocate another K-matrix */
#if defined(K_IS_1X2_TEXTURE)
  real2 *Kanother = (real2 *)malloc(316 * r * (r / 2) * sizeof(real2));
  m2l_aux_convert_K_to_Kanother_for_ij_blocking_row1_col2(r, Ktable, K, Kanother);
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1)
  real4 *Kanother = (real4 *)malloc(316 * (r / 4) * r * sizeof(real4));
  m2l_aux_convert_K_to_Kanother_for_ij_blocking_row4_col1(r, Ktable, K, Kanother);
#elif defined(K_IS_8X1)
  real8 *Kanother = (real8 *)malloc(316 * (r / 8) * r * sizeof(real8));
  m2l_aux_convert_K_to_Kanother_for_ij_blocking_row8_col1(r, Ktable, K, Kanother);
#elif defined(K_IS_16X1)
  real16 *Kanother = (real16 *)malloc(316 * (r / 16) * r * sizeof(real16));
  m2l_aux_convert_K_to_Kanother_for_ij_blocking_row16_col1(r, Ktable, K, Kanother);
#elif defined(K_IS_8X2)
  real8x2 *Kanother = (real8x2 *)malloc(316 * (r / 8) * (r / 2) * sizeof(real8x2));
  m2l_aux_convert_K_to_Kanother_for_ij_blocking_row8_col2(r, Ktable, K, Kanother);
#elif defined(K_IS_4X4)
  real4x4 *Kanother = (real4x4 *)malloc(316 * (r / 4) * (r / 4) * sizeof(real4x4));
  m2l_aux_convert_K_to_Kanother_for_ij_blocking_row4_col4(r, Ktable, K, Kanother);
#elif defined(K_IS_4X2)
  real4x2 *Kanother = (real4x2 *)malloc(316 * (r / 4) * (r / 2) * sizeof(real4x2));
  m2l_aux_convert_K_to_Kanother_for_ij_blocking_row4_col2(r, Ktable, K, Kanother);
#elif defined(K_IS_1X1)
  real *Kanother = (real *)malloc(316 * r * r * sizeof(real));
  m2l_aux_convert_K_to_Kanother_for_ij_blocking_row1_col1(r, Ktable, K, Kanother);
#endif

  /* Allocate another K-matrix on device and copy to device */
#if defined(K_IS_1X2_TEXTURE) // real=float

  cudaArray *d_Kanother;
  cudaChannelFormatDesc myChannelDesc = cudaCreateChannelDesc<float2>();
  cudaExtent myExtent = make_cudaExtent(316, r, r / 2);
#if (CUDART_VERSION >= 3010)
  CSC(cudaMalloc3DArray(&d_Kanother, &myChannelDesc, myExtent, 0));
#else
  CSC(cudaMalloc3DArray(&d_Kanother, &myChannelDesc, myExtent));
#endif
  cudaMemcpy3DParms myParms = {0};
  myParms.extent = myExtent;
  myParms.kind = cudaMemcpyHostToDevice;
  myParms.dstArray = d_Kanother;
  myParms.srcPtr = make_cudaPitchedPtr((void *)Kanother, myExtent.width * sizeof(float2), myExtent.width, myExtent.height); // really??
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy3D(&myParms));
  stopEventTimer(&timer_m2l_set);
  CSC(cudaBindTextureToArray(texRefK, d_Kanother, myChannelDesc));

#elif defined(K_IS_4X1_TEXTURE) // real=float

  cudaArray *d_Kanother;
  cudaChannelFormatDesc myChannelDesc = cudaCreateChannelDesc<float4>(); // real4=float4
  cudaExtent myExtent = make_cudaExtent(316, r / 4, r);
#if (CUDART_VERSION >= 3010)
  CSC(cudaMalloc3DArray(&d_Kanother, &myChannelDesc, myExtent, 0));
#else
  CSC(cudaMalloc3DArray(&d_Kanother, &myChannelDesc, myExtent));
#endif
  cudaMemcpy3DParms myParms = {0};
  myParms.extent = myExtent;
  myParms.kind = cudaMemcpyHostToDevice;
  myParms.dstArray = d_Kanother;
  myParms.srcPtr = make_cudaPitchedPtr((void *)Kanother, myExtent.width * sizeof(float4), myExtent.width, myExtent.height);
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy3D(&myParms));
  stopEventTimer(&timer_m2l_set);
  CSC(cudaBindTextureToArray(texRefK, d_Kanother, myChannelDesc));

#elif defined(K_IS_4X1)

  real4 *d_Kanother;
  CSC(cudaMalloc((void **)&d_Kanother, 316 * (r / 4) * r * sizeof(real4)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Kanother, Kanother, 316 * (r / 4) * r * sizeof(real4), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);

#elif defined(K_IS_8X1)

  real8 *d_Kanother;
  CSC(cudaMalloc((void **)&d_Kanother, 316 * (r / 8) * r * sizeof(real8)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Kanother, Kanother, 316 * (r / 8) * r * sizeof(real8), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);

#elif defined(K_IS_16X1)

  real16 *d_Kanother;
  CSC(cudaMalloc((void **)&d_Kanother, 316 * (r / 16) * r * sizeof(real16)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Kanother, Kanother, 316 * (r / 16) * r * sizeof(real16), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);

#elif defined(K_IS_8X2)

  real8x2 *d_Kanother;
  CSC(cudaMalloc((void **)&d_Kanother, 316 * (r / 8) * (r / 2) * sizeof(real8x2)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Kanother, Kanother, 316 * (r / 8) * (r / 2) * sizeof(real8x2), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);

#elif defined(K_IS_4X4)

  real4x4 *d_Kanother;
  CSC(cudaMalloc((void **)&d_Kanother, 316 * (r / 4) * (r / 4) * sizeof(real4x4)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Kanother, Kanother, 316 * (r / 4) * (r / 4) * sizeof(real4x4), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);

#elif defined(K_IS_4X2)

  real4x2 *d_Kanother;
  CSC(cudaMalloc((void **)&d_Kanother, 316 * (r / 4) * (r / 2) * sizeof(real4x2)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Kanother, Kanother, 316 * (r / 4) * (r / 2) * sizeof(real4x2), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);

#elif defined(K_IS_1X1)

  real *d_Kanother;
  CSC(cudaMalloc((void **)&d_Kanother, 316 * r * r * sizeof(real)));
  startEventTimer(&timer_m2l_set);
  CSC(cudaMemcpy(d_Kanother, Kanother, 316 * r * r * sizeof(real), cudaMemcpyHostToDevice));
  stopEventTimer(&timer_m2l_set);

#endif

  /* Loop over levels */
  for (int level = level_start; level <= level_end; level ++) {
      
    /* Indices of the first and last cells in this level */
    int Fsta = cellsta[level];
    int Fend = cellend[level];

    /* Number of cells in this level */
    int nc = Fend - Fsta + 1;

    /* Compute the starting index of Manother */    
    int Manotherstart = m2l_aux_get_starting_index_of_Manother_for_ij_blocking(r, level_start, level);

    /* Allocate and initialize another L on device */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
    real *d_Lanother;
    CSC(cudaMalloc((void **)&d_Lanother, nc * r * sizeof(real)));
    CSC(cudaMemset(d_Lanother, 0, nc * r * sizeof(real)));
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    real4 *d_Lanother;
    CSC(cudaMalloc((void **)&d_Lanother, nc * (r / 4) * sizeof(real4)));
    CSC(cudaMemset(d_Lanother, 0, nc * (r / 4) * sizeof(real4)));
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    real8 *d_Lanother;
    CSC(cudaMalloc((void **)&d_Lanother, nc * (r / 8) * sizeof(real8)));
    CSC(cudaMemset(d_Lanother, 0, nc * (r / 8) * sizeof(real8)));
#elif defined(K_IS_16X1)
    real16 *d_Lanother;
    CSC(cudaMalloc((void **)&d_Lanother, nc * (r / 16) * sizeof(real16)));
    CSC(cudaMemset(d_Lanother, 0, nc * (r / 16) * sizeof(real16)));
#endif
      
    /* Compute another L */

#if defined(CUDA_VER45H)
#if (CUDA_ARCH >= 20) && (CUDA_ARCH < 30)
    dim3 Db(2 * B * B, B, 4);
    dim3 Dg(POW8(level) / BBB8, NUM_ROW_GROUPS_IJ);
#elif (CUDA_ARCH >= 10) && (CUDA_ARCH < 20)
    dim3 Db(B * B, B, 8);
    dim3 Dg(POW8(level) / BBB8, NUM_ROW_GROUPS_IJ);
#endif
#else
    dim3 Db(B * B, B, 8);
    dim3 Dg(POW8(level) / BBB8, NUM_ROW_GROUPS_IJ);
#endif
    CHECK_CONFIGURATION(Dg, Db);

#if defined(CUDA_VER45I) && defined(USE_ANY_R)
    startEventTimer(&timer_m2l_kernel);
    if (B == 2) {
      m2l_kern_ij_blocking_b2<<<Dg, Db>>>(r, d_Lanother, d_Kanother, d_Manother, level, Manotherstart);
    } else if (B == 4) {
      m2l_kern_ij_blocking_b4<<<Dg, Db>>>(r, d_Lanother, d_Kanother, d_Manother, level, Manotherstart);
    } else {
      INFO("B=%d is not implemented for any r. Exit.\n", B);
    }
    stopEventTimer(&timer_m2l_kernel);
#else
    if (r == 32) {
      startEventTimer(&timer_m2l_kernel);
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_4X1_TEXTURE)
      m2l_kern_ij_blocking_r32b4<<<Dg, Db>>>(d_Lanother, d_Manother, level, Manotherstart);
#elif defined(K_IS_4X1) || defined(K_IS_8X1) || defined(K_IS_16X1) || defined(K_IS_8X2) || defined(K_IS_4X4) || defined(K_IS_4X2) || defined(K_IS_1X1)
      m2l_kern_ij_blocking_r32b4<<<Dg, Db>>>(d_Lanother, d_Kanother, d_Manother, level, Manotherstart);
#endif
      stopEventTimer(&timer_m2l_kernel);
    } else if (r == 256) {
      startEventTimer(&timer_m2l_kernel);
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_4X1_TEXTURE)
      m2l_kern_ij_blocking_r256b4<<<Dg, Db>>>(d_Lanother, d_Manother, level, Manotherstart);
#elif defined(K_IS_4X1) || defined(K_IS_8X1) || defined(K_IS_16X1) || defined(K_IS_8X2) || defined(K_IS_4X4) || defined(K_IS_4X2) || defined(K_IS_1X1)
      m2l_kern_ij_blocking_r256b4<<<Dg, Db>>>(d_Lanother, d_Kanother, d_Manother, level, Manotherstart);
#endif
      stopEventTimer(&timer_m2l_kernel);
    } else { 
      INFO("r=%d is not implemented. Exit.\n", r);
      exit(1);
    }
#endif

    /* Allocate another L */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
    real *Lanother = (real *)malloc(nc * r * sizeof(real));
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    real4 *Lanother = (real4 *)malloc(nc * (r / 4) * sizeof(real4));
#elif defined(K_IS_8X1) || defined(K_IS_8X2) 
    real8 *Lanother = (real8 *)malloc(nc * (r / 8) * sizeof(real8));
#elif defined(K_IS_16X1)
    real16 *Lanother = (real16 *)malloc(nc * (r / 16) * sizeof(real16));
#endif

    /* Copy another L from device */
    startEventTimer(&timer_m2l_get);
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
    CSC(cudaMemcpy(Lanother, d_Lanother, nc * r        * sizeof(real), cudaMemcpyDeviceToHost));
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    CSC(cudaMemcpy(Lanother, d_Lanother, nc * (r / 4)  * sizeof(real4), cudaMemcpyDeviceToHost));
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    CSC(cudaMemcpy(Lanother, d_Lanother, nc * (r / 8)  * sizeof(real8), cudaMemcpyDeviceToHost));
#elif defined(K_IS_16X1)
    CSC(cudaMemcpy(Lanother, d_Lanother, nc * (r / 16) * sizeof(real16), cudaMemcpyDeviceToHost));
#endif
    stopEventTimer(&timer_m2l_get);

    /* Free d_Lanother */
    CSC(cudaFree(d_Lanother));

    /* Convert Lanother to L */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_1X1)
    m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1(r, center, L0, level, B, Lanother, &(L[r * Fsta]));
#elif defined(K_IS_4X1_TEXTURE) || defined(K_IS_4X1) || defined(K_IS_4X4) || defined(K_IS_4X2)
    m2l_aux_convert_Lanother_to_L_for_ij_blocking_row4(r, center, L0, level, B, Lanother, &(L[r * Fsta]));
#elif defined(K_IS_8X1) || defined(K_IS_8X2)
    m2l_aux_convert_Lanother_to_L_for_ij_blocking_row8(r, center, L0, level, B, Lanother, &(L[r * Fsta]));
#elif defined(K_IS_16X1)
    m2l_aux_convert_Lanother_to_L_for_ij_blocking_row16(r, center, L0, level, B, Lanother, &(L[r * Fsta]));
#endif

    /* Free Lanother */
    free(Lanother);

  }

  /* Clean up */
#if defined(K_IS_1X2_TEXTURE) || defined(K_IS_4X1_TEXTURE)
  CSC(cudaUnbindTexture(texRefK));
  CSC(cudaFreeArray(d_Kanother));
  //#elif defined(K_IS_4X1) || defined(K_IS_8X1) || defined(K_IS_16X1) || defined(K_IS_8X2) || defined(K_IS_4X4) || defined(K_IS_4X2)
#else
  CSC(cudaFree(d_Kanother));
#endif
  CSC(cudaFree(d_Manother));

  free(Kanother);
  free(Manother);
  free(cellsta);
  free(cellend);

  /* Save the kernel execution time in the global variable */
  kernel_exec_time = getEventTimer(&timer_m2l_kernel);

  /* Finalize timers */
  INFO("timer_m2l_kernel = %14.7e\n", getEventTimer(&timer_m2l_kernel));
  INFO("timer_m2l_set = %14.7e\n", getEventTimer(&timer_m2l_set));
  INFO("timer_m2l_get = %14.7e\n", getEventTimer(&timer_m2l_get));
  finalizeEventTimer(&timer_m2l_kernel);
  finalizeEventTimer(&timer_m2l_set);
  finalizeEventTimer(&timer_m2l_get);

  stopEventTimer(&timer_m2l_all);
  INFO("timer_m2l_all = %14.7e\n", getEventTimer(&timer_m2l_all));
  finalizeEventTimer(&timer_m2l_all);
}

#endif /* M2L_HOST_IJ_BLOCKING_CU */
