#ifndef ANOTHER_H
#define ANOTHER_H

#include <stdio.h>
#include <stdlib.h>
#include "vec234.h"
#include "real.h"
#include "anotherTree.h"
#include "mathmacros.h"
#include <assert.h>
#include "debugmacros.h"

#include "cell.h"

#include "anotherTree.h"

#include "timer.h"

/* Alignments for neighbors[] and interaction[] */
#ifndef ALIGN_SIZE
#define ALIGN_SIZE (64)
#endif

/* Define the minimum level of hierachy */
#ifndef MINLEV
#define MINLEV (2)
#endif

#if defined(_cplusplus) || defined(CUDA)
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

/* Function declarations */
EXTERN void anotherBuildFMMHierachy(real L, int n, int dof, int l, int cutoff,
				    anotherTree **atree,
				    int Nf, int Ns, real3 *field, real3 *source);
EXTERN void anotherFMMCleanup(anotherTree **atree);


EXTERN void anotherFMMCompute(anotherTree **atree, real3 *field, real3 *source, real *q, 
			      real *K, real *U, real *Tkz, int *Ktable, 
			      real *Kweights, real *Cweights, real homogen, 
			      int cutoff, int n, int dof, real *phi);

EXTERN void anotherFMMInteraction(anotherTree **atree, real *E, int *Ktable, 
				  real *U, real *Kweights, int n, int dof,
				  int cutoff, real homogen);
EXTERN void anotherUpwardPass(anotherTree **atree, real3 *source, real *Cweights, real *Tkz, 
			      real *q, real *V, real *Kweights, int cutoff, int n, int dof);
EXTERN void anotherDownwardPass(anotherTree **atree, real3 *field, real3 *source,
				real *Cweights, real *Tkz, real *q, real *U,
				int cutoff, int n, int dof, real homogen, real *phi);

EXTERN void checkTree(anotherTree *atree);
EXTERN double elapsed(void);
EXTERN void calc_performance(char *str, double flop, double sec);
EXTERN void estimate_performacne_bounded_by_bandwidth(char *str, double size_load_store_bytes,
						      double peak_bandwidth_giga_bytes, double flop);

EXTERN void anotherDownwardPassX(anotherTree **atree, real3 *field,
				 real *Cweights, real *Tkz, real *q, real *U,
				 int cutoff, int n, int dof, real homogen, real *phi);

EXTERN void anotherFMMDistribute(anotherTree **atree, real3 *field, real3 *source);
EXTERN void anotherFMMDistribute_check(anotherTree *atree);
EXTERN void anotherFMMDistribute_check2(anotherTree *atree);
EXTERN void anotherInteractionList(anotherTree **atree);
EXTERN void anotherInteractionListCheck(anotherTree *atree);

EXTERN void anotherNearField(anotherTree **atree, int dof, real homogen,
			     real3 *field, real3 *source, real *phi, real *q);

#ifdef CUDA
#include <cutil.h>

#ifdef _DEBUG
#define CSC(call) CUDA_SAFE_CALL(call)
#else
#define CSC(call) call
#endif

#endif /* CUDA */

/* Define the minimum level of hierachy */
#define MINLEV (2)

/* SI prefixes */
#define tera 1000000000000
#define giga 1000000000
#define mega 1000000
#define kilo 1000

#define GB giga /* giga bytes */
#define MB mega /* mega bytes */
#define KB kilo /* kilo bytes */

#endif /* ANOTHER_H */
