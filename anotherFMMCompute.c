#include "bbfmm.h"

void anotherFMMCompute(anotherTree **atree, real3 *field, real3 *source, real *q, 
		       real *K, real *U, real *Tkz, int *Ktable, 
		       real *Kweights, real *Cweights, real homogen, 
		       int cutoff, int n, int dof, real *phi)
{
  /* Upward pass */
  MSG("Enter anotherUpwardPass.\n");
#ifndef DISABLE_TIMING
  startTimer(timer_upward);
#endif
  anotherUpwardPass(atree, source, Cweights, Tkz, q, U, Kweights, cutoff, n, dof);
#ifndef DISABLE_TIMING
  stopTimer(timer_upward);
#endif
  MSG("Exit anotherUpwardPass.\n");

  /* Computes all of the cell interactions */
  MSG("Enter anotherFMMInteraction.\n");
#ifndef DISABLE_TIMING
  startTimer(timer_interact);
#endif
  anotherFMMInteraction(atree, K, Ktable, U, Kweights, n, dof, cutoff, homogen);
#ifndef DISABLE_TIMING
  stopTimer(timer_interact);
#endif
  MSG("Exit anotherFMMInteraction.\n");
  
  /* Downward pass */
  MSG("Enter anotherDownwardPass.\n");
#ifndef DISABLE_TIMING
  startTimer(timer_downward);
#endif
  anotherDownwardPass(atree, field, source, Cweights, Tkz, q, U, cutoff, n, dof, homogen, phi);
#ifndef DISABLE_TIMING
  stopTimer(timer_downward);
#endif
  MSG("Exit anotherDownwardPass.\n");
}
