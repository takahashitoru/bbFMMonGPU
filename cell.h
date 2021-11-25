#ifndef CELL_H
#define CELL_H

#include "real.h"
#include "vec234.h"

#define NULLCELL (- 1)
#define NULL_CELL NULLCELL

typedef struct {
  //111123  int *leaves;
  //111123  int *parent;
  int *neighbors;
  int *interaction;
  real3 *center;
#ifdef PBC  
  real3 *cshiftneigh, *cshiftinter;
#endif
  int *fieldsta, *fieldend, *sourcesta, *sourceend;
  int *ineigh, *iinter;
  int pitch_neighbors, pitch_interaction;
} cell;

#include "mathmacros.h"

//#define GET_PARENT_INDEX(A) ( (((A) - 1) - ((A) - 1) % 8) / 8 )
#define GET_PARENT_INDEX(A) ( DIV8(((A) - 1) - MOD8((A) - 1)) )
#define GET_CHILD_INDEX(A, i) ( 8 * (A) + 1 + (i) )

#define GET_NUM_PARTICLES(sta, end, A) ( (end)[A] - (sta)[A] + 1 )

#define GET_SIBLING_INDEX(A) ( (A) - GET_CHILD_INDEX(GET_PARENT_INDEX(A), 0) )

#endif /* CELL_H */
