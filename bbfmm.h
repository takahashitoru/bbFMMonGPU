/*
 * File: bbfmm.h
 * Description: Header file for bbfmm.c which contains functions used
 * in the implementation of the black-box fast multiple method.
 * ----------------------------------------------------------------------
 * 
 * Black-Box Fast Multipole Method (BBFMM)
 * William Fong
 * Stanford University
 *
 */

#ifndef _BBFMM_H
#define _BBFMM_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ORIGINAL
#error ORIGINAL is no longer supported. Neither ENABLE_ORIGINAL_FIELD_SOURCE_LISTS is.
#endif

/* Assumption of the degree of freedom */
#define DEGREE_OF_FREEDOM 1
#define DOF DEGREE_OF_FREEDOM

#include "vec234.h"
#include "real.h"

/* Uniform random number generator */
#define frand(xmin,xmax) ((real)(xmin)+(real)((xmax)-(xmin))*rand()/(real)RAND_MAX) 

#if defined(_cplusplus) || defined(CUDA)
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

/* Struct: fmmparam
 * -------------------------------------------------------------------
 * This struct stores all of the FMM parameters.
 */
typedef struct _fmmparam {
  int Ns;       // Number of sources
  int Nf;       // Number of field points
  int dof;        // Number of degrees of freedom
  real L;        // Length of one side of simulation cube
  int n;        // Number of Chebyshev nodes in one direction
  int levels;        // Maximum number of levels in octree
  int PBClevels;        // Number of PBC levels (images = 27^PBClevels)
  int PBCshells;        // Number of PBC shells (for direct calculation)
  int precomp;        // Turn on (1) or off (0) pre-computation
  int cutoff;       // Number of singular values to keep
  real homogen;        // Order of homogeneity of kernel
  char filesval[80];     // File name for storing singular values
  char filesvec[80];     // File name for storing singular vectors
} fmmparam;

// Declaration for LAPACK's SVD routine
EXTERN void agesvd_(char *jobu, char *jobvt, int *m, int *n, real *A,
		    int *lda, real *S, real *U, int *ldu, real *VT,
		    int *ldvt, real *work, int *lwork, int *info);

// Declaration for BLAS matrix-matrix multiply
EXTERN void agemm_(char *transa, char *transb, int *m, int *n, int *k,
		   real *alpha, real *A, int *lda, real *B,
		   int *ldb, real *beta, real *C, int *ldc);

// Declaration for BLAS matrix-vector multiply
EXTERN void agemv_(char *trans, int *m, int *n, real *alpha, real *A,
		   int *lda, real *x, int *incx, real *beta, 
		   real *y, int *incy);

// Declaration for BLAS dot product
EXTERN real adot_(int *n, real *dx, int *incx, real *dy, int *incy);

// Declaration for BLAS daxpy
EXTERN real aaxpy_(int *n, real *da, real *dx, int *incx, real *dy,
		     int *incy);

// Declaration for BLAS vector copy
EXTERN real acopy_(int *n, real *dx, int *incx, real *dy, int *incy);

/*
 * Function: bbfmm
 * -----------------------------------------------------------------
 * Given the source and field point locations, strength of the sources,
 * number of field points and sources, length of the computation cell, and the number of Chebyshev nodes, the field is computed.
 */
EXTERN void bbfmm(real3 *field, real3 *source, real *q, int Nf, int Ns,
		  real L, int n, real *phi, int l);

/*
 * Function: FMMSetup
 * -----------------------------------------------------------------
 * Prepare for the FMM calculation by setting the parameters, computing
 * the weight matrices, pre-computing the SVD (if necessary), reading 
 * in the necessary matrices, and building the FMM hierarchy.
 */
EXTERN void FMMSetup(real *Tkz, int *Ktable, real *Kweights,
		     real *Cweights, real L, real *homogen, int *cutoff, 
		     int n, int dof, int Ns, int *l, char *Kmat, char *Umat);

/*
 * Function: FMMReadMatrices
 * ------------------------------------------------------------------
 * Read in the kernel interaction matrix M and the matrix of singular
 * vectors U.
 */
EXTERN void FMMReadMatrices(real *K, real *U, int cutoff, int n, int dof, char *Kmat, char *Umat);

/*
 * Function: SetParam
 * ------------------------------------------------------------------
 * Read in the user-specified options from the options file.
 * 
 */
EXTERN int SetParam(char *options, fmmparam *param);

/*
 * Function: SetSources
 * -------------------------------------------------------------------
 * Distributes an equal number of positive and negative charges uniformly
 * in the simulation cell ([-0.5*L,0.5*L])^3 and takes these same locations
 * as the field points.
 */
EXTERN void SetSources(real3 *field, real3 *source, real *q, int N, int dof, real L, int seed);

/*
 * Function DirectCalc3D
 * -------------------------------------------------------------------
 * Computes the potential at the first field point and returns 
 * the result in phi.
 */
EXTERN void DirectCalc3D(real3 *field, real3 *source, real *q,
			 int Nf, int Ns, int dof, int start, real L, real *phi);


/*
 * Function: EwaldSolution
 * ---------------------------------------------------------------------
 * Computes the Ewald solultion for 1/r, r.x/r^3, and 1/r^4 kernels 
 * (for comparision to PBC with FMM).
 */
EXTERN void EwaldSolution(real3 *field, real3 *source, real *q, 
			  int Nf, int Ns, int nstart, int mstart, 
			  real L, real beta, real *phi, real *corr);

/*
 * Function: EvaluateKernel
 * -------------------------------------------------------------------
 * Evaluates the kernel given a source and a field point.
 */
EXTERN real EvaluateKernel(real3 fieldpos, real3 sourcepos);

/*
 * Function: EvaluateKernelCell
 * -------------------------------------------------------------------
 * Evaluates the kernel for interactions between a pair of cells.
 */
EXTERN void EvaluateKernelCell(real3 *fieldpos, real3 *sourcepos, 
			       int Nf, int Ns, int dof, real *kernel);

/*
 * Function: EvaluateField
 * -------------------------------------------------------------------
 * Evaluates the kernel for interactions between a pair of cells.
 */
EXTERN void EvaluateField(real3 *field, real3 *source, real *q, int Nf, 
			  int Ns, int dof, real *fieldval);

/*
 * Function: ComputeWeights
 * ------------------------------------------------------------------
 * Computes the weights for the Chebyshev nodes of all children cells
 * (identical for all cells and all levels so just compute once and
 * store in memory) and set up the lookup table.
 */
EXTERN void ComputeWeights(real *Tkz, int *Ktable, real *Kweights, real *Cweights, int n);

/*
 * Function: ComputeWeightsPBC
 * ------------------------------------------------------------------
 * Computes the weights for the Chebyshev nodes of all children cells
 * (identical for all cells and all levels so just compute once and
 * store in memory) (for PBC calculation each cell has 27 children
 * instead of 8).
 */
EXTERN void ComputeWeightsPBC(real *Wup, int n, int dof);

/*
 * Function: ComputeKernelSVD
 * -------------------------------------------------------------------
 * Computes the kernel for 316n^6 interactions between Chebyshev nodes
 * and then computes the SVD of the kernel matrix.
 */
EXTERN void ComputeKernelSVD(real *Kweights, int n, int dof,
			     char *Kmat, char *Umat);

/*
 * Function: ComputePeriodicKernel
 * ---------------------------------------------------------------------
 * Forms the matrix that describes the interactions of the computational
 * cell with its periodic images up to lpbc shells.
 */
EXTERN void ComputePeriodicKernel(real *KPBC, real *Wup, real L,
				  int n, int dof,int lpbc);

/*
 * Function: ComputeTk
 * -------------------------------------------------------------------
 * Computes T_k(x) for k between 0 and n-1 inclusive.
 */
EXTERN void ComputeTk(real x, int n, real *vec);

/*
 * Function: ComputeSn
 * ------------------------------------------------------------------
 * Computes S_n(x_m,x_i) for all Chebyshev node-point pairs using 
 * Clenshaw's recurrence relation.
 */
EXTERN void ComputeSn(real3 *point, real *Tkz, int n, int N, real3 *Sn);

EXTERN void ComputeSn2(real3 *point, real *Tkz, int n, int N, real *Snx, real *Sny, real *Snz);


/**************
   Takahashi
**************/
/* Assumptions */
/*
  The next option lets setSources copy sources[] to fields[].
  This is always true in Will's code.
*/
#define FIELDPOINTS_EQ_SOURCES (1)

/* Include headder files */
#include "another.h"
#include "timer.h"
#include "options.h"
#include "opts.h"
#include "envs.h"
#include "output.h"
#include "debugmacros.h"

/* Global variables */
timerType *timer_main;
timerType *timer_run;
timerType *timer_setup;
timerType *timer_comp;
timerType *timer_upward;
timerType *timer_interact;
timerType *timer_downward;
timerType *timer_output;
timerType *timer_cuda;

/* Workaround of M_PI for icc -std=c99 */
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/* Check performance by default */
#ifndef CHECK_PERFORMANCE
#define CHECK_PERFORMANCE
#endif

#endif /* _BBFMM_H */
