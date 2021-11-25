#include "another.h"

#define AIL_SAFE_FACTOR ( (real)1.5 ) // this must be greater than or equal to 1.0

//120208#if(1)

//120208#if defined(PBC)
//120208#error PBC is not supported.
//120208#endif

void anotherInteractionList(anotherTree **atree)
{
  const int maxlev = (*atree)->maxlev;
  const int *levsta = (*atree)->levsta;
  const int *levend = (*atree)->levend;
  const real *celeng = (*atree)->celeng;
  cell *c = (*atree)->c;
  int *neighbors = c->neighbors;
  int *interaction = c->interaction;
  const real3 *center = c->center;
  int *ineigh = c->ineigh; // this must be initialized beforehand
  int *iinter = c->iinter; // this must be initialized beforehand
  ///////////////////////////////////////////////////////////////
#if defined(MYDEBUG)
  const int ncell = (*atree)->ncell;
  for (int A = 0; A < ncell; A ++) {
    if (ineigh[A] != 0) {
      DBG("fatal error A=%d ineigh[A]=%d\n", A, ineigh[A]);
      abort();
    }
    if (iinter[A] != 0) {
      DBG("fatal error A=%d iinter[A]=%d\n", A, iinter[A]);
      abort();
    }
  }
#endif
  ///////////////////////////////////////////////////////////////
  const int pitch_neighbors = c->pitch_neighbors;
  const int pitch_interaction = c->pitch_interaction;

  /* Set up for the neighbors of the root cell (index 0) */
  neighbors[pitch_neighbors * 0 + 0] = 0;
  ineigh[0] = 1;
  
  /* Loop over levels excetp for the root level */
  for (int level = 1; level <= maxlev; level ++) {
    
    /* Set the cutoff between near and far for each dimension */
    const real Lsafe = celeng[level] * AIL_SAFE_FACTOR;
    const real Lsafe2 = Lsafe * Lsafe;
    
    /* Loop over cells in this level */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      
      /* Links to the lists for this cell */
      int *neighborsA = &(neighbors[pitch_neighbors * A]);
      int *interactionA = &(interaction[pitch_interaction * A]);

      /* Obtain the center of A */
      const real3 fcenter = center[A]; // field cell

      /* Obtain the parent of A */
      const int P = GET_PARENT_INDEX(A);
    
      /* Loop over P's neighbor cells (A's uncles; including P) */
      for (int i = 0; i < ineigh[P]; i ++) {
	
	/* Obtain an uncle U */
	const int U = neighbors[pitch_neighbors * P + i];
	
	/* Loop over U's children (A's cousins; including A) */
	for (int j = 0; j < 8; j ++) {

	  /* Obtain a cousin C */
	  const int C = GET_CHILD_INDEX(U, j);
 
	  /* Obtain the center of C */
	  const real3 scenter = center[C]; // source cell
	      
	  /* Difference from A to C */
	  const real diffx = scenter.x - fcenter.x;
	  const real diffy = scenter.y - fcenter.y;
	  const real diffz = scenter.z - fcenter.z;
	  
	  /* Check if C is close to A or not. Note that it is not
	     checked if A or U is empty or not */
	  if (diffx * diffx <= Lsafe2 && diffy * diffy <= Lsafe2 && diffz * diffz <= Lsafe2) {

	    /* Append C to A's neighbor-list */
	    neighborsA[ineigh[A]] = C;

	    /* Increase the length of A's neighbor-list */
	    ineigh[A] ++;

	  } else {

	    /* Append C to A's intearction-list */
	    interactionA[iinter[A]] = C;

	    /* Increase the length of A's interaction-list */
	    iinter[A] ++;

	  }
	} // j	
      } // i
    } // A
  } // level
}

//120208#else
//120208
//120208void anotherInteractionList(anotherTree **atree)
//120208{
//120208  const int ncell = (*atree)->ncell;
//120208  const int maxlev = (*atree)->maxlev;
//120208  const int *levsta = (*atree)->levsta;
//120208  const int *levend = (*atree)->levend;
//120208  const real *celeng = (*atree)->celeng;
//120208  cell *c = (*atree)->c;
//120208  int *neighbors = c->neighbors; // unknown
//120208  int *interaction = c->interaction; // unknown
//120208  const real3 *center = c->center;
//120208#ifdef PBC
//120208  real3 *cshiftneigh = c->cshiftneigh; // unknown
//120208  real3 *cshiftinter = c->cshiftinter; // unknown
//120208#endif
//120208  int *ineigh = c->ineigh; // unknown
//120208  int *iinter = c->iinter; // unknown
//120208  const int pitch_neighbors = c->pitch_neighbors;
//120208  const int pitch_interaction = c->pitch_interaction;
//120208
//120208  /* Initialise the numbers of neighbors and interactions */
//120208  for (int A = 0; A < ncell; A ++) {
//120208    ineigh[A] = 0;
//120208    iinter[A] = 0;
//120208  }
//120208
//120208  /* Set up for the neighbors of the root cell */
//120208  neighbors[pitch_neighbors * 0 + 0] = 0;
//120208  ineigh[0] = 1;
//120208#ifdef PBC
//120208  cshiftneigh[pitch_neighbors * 0 + 0].x = 0; // only for PBC?
//120208  cshiftneigh[pitch_neighbors * 0 + 0].y = 0; // only for PBC?
//120208  cshiftneigh[pitch_neighbors * 0 + 0].z = 0; // only for PBC?
//120208#endif
//120208  
//120208  /* Loop over levels except for maxlev */
//120208  for (int level = 0; level < maxlev; level ++) {
//120208
//120208    /* Sets the cutoff between near and far to be L (this is
//120208       equivalent to a one cell buffer) */
//120208    const real L = celeng[level];
//120208    const real iL = ONE / L;
//120208
//120208    const real LCsafe = (L / TWO) * AIL_SAFE_FACTOR; // L/2=celeng[level+1]
//120208    const real LCsafe2 = LCsafe * LCsafe;
//120208
//120208    /* Loop over cells */
//120208#ifdef _OPENMP
//120208#pragma omp parallel for
//120208#endif
//120208    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
//120208
//120208      if (level < maxlev) { // this guarantees that A is not a leaf, since perfect octree is assumed
//120208
//120208	/* 
//120208	   Finds all neighbors that are too close for the far field
//120208	   approximation and stores them in the neighbors array - Also
//120208	   finds the neighboring cells that are sufficiently far away
//120208	   and stores them in interaction array
//120208	*/
//120208	
//120208	/* Loop over A's neighbor cells */
//120208	for (int i = 0; i < ineigh[A]; i ++) {
//120208
//120208	  const int B = neighbors[pitch_neighbors * A + i];
//120208#ifdef PBC
//120208	  real3 cshift = cshiftneigh[pitch_neighbors * A + i]; // only for PBC?
//120208#endif
//120208	  
//120208	  for (int j = 0; j < 8; j ++) {
//120208
//120208	    const int D = GET_CHILD_INDEX(B, j); // source cell
//120208	    
//120208	    /* Note: Even if the current source cell D is empty, it is
//120208	       appended to either the neighbor list or interaction
//120208	       list of A's children (C) */
//120208
//120208	    real3 scenter = center[D]; // center of source cell
//120208#ifdef PBC
//120208	    scenter.x += cshift.x; // only for PBC?
//120208	    scenter.y += cshift.y;
//120208	    scenter.z += cshift.z;
//120208#endif
//120208	    
//120208	    for (int k = 0; k < 8; k ++) {
//120208
//120208	      const int C = GET_CHILD_INDEX(A, k); // field cell
//120208	      const real3 fcenter = center[C]; // center of field cell
//120208	      
//120208	      const real diffx = scenter.x - fcenter.x;
//120208	      const real diffy = scenter.y - fcenter.y;
//120208	      const real diffz = scenter.z - fcenter.z;
//120208
//120208	      /* Check if D is close to C or not */
//120208	      if (diffx * diffx <= LCsafe2 && diffy * diffy <= LCsafe2 && diffz * diffz <= LCsafe2) {
//120208
//120208		/* Append the source cell to neighbor-list */
//120208		neighbors[pitch_neighbors * C + ineigh[C]] = D;
//120208#ifdef PBC
//120208		cshiftneigh[pitch_neighbors * C + ineigh[C]] = cshift; // only for PBC?
//120208#endif
//120208		/* Increase the length of neighbor-list */
//120208		(ineigh[C]) ++;
//120208
//120208	      } else {
//120208
//120208		/* Append the source cell to intearction-list */
//120208		interaction[pitch_interaction * C + iinter[C]] = D;
//120208#ifdef PBC
//120208		cshiftinter[pitch_interaction * C + iinter[C]] = cshift; // only for PBC?
//120208#endif
//120208
//120208		/* Increase the length of interaction-list */
//120208		(iinter[C]) ++;
//120208
//120208	      }
//120208	    }
//120208	  } // i
//120208	} // j
//120208      }
//120208    } // A
//120208  } // level
//120208}
//120208
//120208#endif


void anotherInteractionListCheck(anotherTree *atree)
{
  const int maxlev = atree->maxlev;
  const int *levsta = atree->levsta;
  const int *levend = atree->levend;
  const cell *c = atree->c;

  const int *neighbors = c->neighbors;
  const int *interaction = c->interaction;
  const int *ineigh = c->ineigh;
  const int *iinter = c->iinter;
  const int pitch_neighbors = c->pitch_neighbors;
  const int pitch_interaction = c->pitch_interaction;

  for (int level = 0; level <= maxlev; level ++) {
    for (int A = levsta[level]; A <= levend[level]; A ++) {
      DBG("level=%d A=%d (%d): neigbors=", level, A, ineigh[A]);
      for (int i = 0; i < ineigh[A]; i++) {
	fprintf(stderr, " %d", neighbors[pitch_neighbors * A + i]);
      }
      fprintf(stderr, "\n");
    }
  }
  
  for (int level = 0; level <= maxlev; level ++) {
    for (int A = levsta[level]; A <= levend[level]; A ++) {
      DBG("level=%d A=%d (%d): interaction=", level, A, iinter[A]);
      for (int i = 0; i < iinter[A]; i++) {
	fprintf(stderr, " %d", interaction[pitch_interaction * A + i]);
      }
      fprintf(stderr, "\n");
    }
  }

}
