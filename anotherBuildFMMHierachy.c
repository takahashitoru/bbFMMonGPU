#include "bbfmm.h" // defines FIELDPOINTS_EQ_SOURCES

static int globalIndex(int level, int localIndex)
{
  int s, n, i;
  /*
    local : 0  0,1,2,3,4,5,6,7   0, 1, 2,...,63   0, 1, 2,...,511
    s     : 0  1,1,1,1,1,1,1,1   9, 9, 9,..., 9  73,73,73,..., 73
    global: 0  1,2,3,4,5,6,7,8   9,10,11,...,72  73,74,75,...,584
  */

  if (level == 0) {
    s = 0;
  } else {
    s = 1;
    for (i = 1, n = 8; i < level; i++, n *= 8) {
      s += n;
    }
  }
  return (s + localIndex);
}  


void anotherBuildFMMHierachy(real L, int n, int dof, int l, int cutoff,
			     anotherTree **atree,
			     int Nf, int Ns, real3 *field, real3 *source)
{
  int maxlev;
  int dofn3 = dof * n * n * n;

  /* Allocate a tree of another type */
  *atree = (anotherTree *)malloc(sizeof(anotherTree));

  /* Set and allocate members of tree */
  (*atree)->minlev = MINLEV;
  (*atree)->maxlev = l; // not l + 1
  maxlev = (*atree)->maxlev;
  INFO("maxlev = %d\n", maxlev);
  (*atree)->levsta = (int *)malloc((maxlev + 1) * sizeof(int)); // first cell's index in each level
  (*atree)->levend = (int *)malloc((maxlev + 1) * sizeof(int)); // last cell's index in each level
  (*atree)->celeng = (real *)malloc((maxlev + 1) * sizeof(real)); // size of cells in each level
  (*atree)->c = (cell *)malloc(sizeof(cell));

  /* Compute the number of cells, the first and last cell's indices for each level */
  (*atree)->ncell = 0;
  int num = 1;
  real length = L;
  for (int level = 0; level <= maxlev; level ++) {
    (*atree)->levsta[level] = globalIndex(level,       0);
    (*atree)->levend[level] = globalIndex(level, num - 1);
    (*atree)->celeng[level] = length;
    (*atree)->ncell += num;
    num *= 8;
    length *= 0.5;
  }
  
  /* Copy */
  int ncell = (*atree)->ncell;
  INFO("ncell = %d\n", ncell);

  /* Allocate other members */
  cell *c = (*atree)->c;
  c->center = (real3 *)malloc(ncell * sizeof(real3));
  (*atree)->Nf = Nf;
  (*atree)->Ns = Ns;
  c->ineigh = (int *)calloc(ncell, sizeof(int)); // initialized
  c->iinter = (int *)calloc(ncell, sizeof(int)); // initialized

  /* Create an another type of octtree from scratch */
#ifndef DISABLE_TIMING
  timerType *timer_setup_tree;
  allocTimer(&timer_setup_tree);
  initTimer(timer_setup_tree);
  startTimer(timer_setup_tree);
#endif
  anotherFMMDistribute(atree, field, source);
#ifndef DISABLE_TIMING
  stopTimer(timer_setup_tree);
  printTimer(stderr, "setup_tree", timer_setup_tree);
  freeTimer(&timer_setup_tree);
#endif
#if(0)
  anotherFMMDistribute_check(*atree);
#endif
#if(0)
  anotherFMMDistribute_check2(*atree);
#endif

  /* Create neighbor and interactions lists */
  c->pitch_neighbors = 27; // set the theoretical maximum value; this looks fine everytime.
  c->pitch_interaction = 189; // set the theoretical maximum value; this looks fine everytime.
#ifndef DISABLE_ALIGN
  DBG("pitch_neighbors = %d\n", c->pitch_neighbors);
  DBG("pitch_interaction = %d\n", c->pitch_interaction);
  c->pitch_neighbors = FLOORUP(c->pitch_neighbors, ALIGN_SIZE);
  c->pitch_interaction = FLOORUP(c->pitch_interaction, ALIGN_SIZE);
#endif
  INFO("pitch_neighbors = %d\n", c->pitch_neighbors);
  INFO("pitch_interaction = %d\n", c->pitch_interaction);
  c->neighbors = (int *)malloc(ncell * c->pitch_neighbors * sizeof(int));
  c->interaction = (int *)malloc(ncell * c->pitch_interaction * sizeof(int));
#ifdef PBC
  c->cshiftneigh = (real3 *)malloc(ncell * c->pitch_neighbors * sizeof(real3));
  c->cshiftinter = (real3 *)malloc(ncell * c->pitch_interaction * sizeof(real3));
#endif
#ifndef DISABLE_TIMING
  timerType *timer_setup_list;
  allocTimer(&timer_setup_list);
  initTimer(timer_setup_list);
  startTimer(timer_setup_list);
#endif
  anotherInteractionList(atree);
#ifndef DISABLE_TIMING
  stopTimer(timer_setup_list);
  printTimer(stderr, "setup_list", timer_setup_list);
  freeTimer(&timer_setup_list);
#endif
#if(0)
  checkAnotherInteractionList(atree);
#endif

}


void anotherFMMCleanup(anotherTree **atree)
{
  cell *c = (*atree)->c;

  free(c->neighbors);
  free(c->interaction);
  free(c->center);
#ifdef PBC
  free(c->cshiftneigh);
  free(c->cshiftinter);
#endif
  free(c->fieldsta);
  free(c->fieldend);
#if !defined(FIELDPOINTS_EQ_SOURCES)
  free(c->sourcesta);
  free(c->sourceend);
#endif
  free(c->ineigh);
  free(c->iinter);
  
  free((*atree)->levsta);
  free((*atree)->levend);
  free((*atree)->celeng);
  free((*atree)->c);
  free((*atree)->fieldlist);
#if !defined(FIELDPOINTS_EQ_SOURCES)
  free((*atree)->sourcelist);
#endif

  free(*atree);
}
