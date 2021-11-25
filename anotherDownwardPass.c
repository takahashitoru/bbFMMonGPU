#include "bbfmm.h"

void anotherDownwardPass(anotherTree **atree, real3 *field, real3 *source,
			 real *Cweights, real *Tkz, real *q, real *U,
			 int cutoff, int n, int dof, real homogen, real *phi)
{

  /*
    Evaluation of far field interactions; L2L operation and evaluation by L
  */ 
  
  anotherDownwardPassX(atree, field, Cweights, Tkz, q, U, cutoff, n, dof, homogen, phi);

  /*
    Evaluation of near field interactions or direct computation;
    Assume that direct compution is done only at the maximum level
  */

  timerType *timer_downward_nearby;
  allocTimer(&timer_downward_nearby);
  initTimer(timer_downward_nearby);
  startTimer(timer_downward_nearby);

  anotherNearField(atree, dof, homogen, field, source, phi, q);

  stopTimer(timer_downward_nearby);
  printTimer(stderr, "downward_nearby", timer_downward_nearby);
  freeTimer(&timer_downward_nearby);
}
