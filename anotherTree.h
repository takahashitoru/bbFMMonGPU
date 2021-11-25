#ifndef ANOTHERTREE_H
#define ANOTHERTREE_H

#include "real.h"
#include "cell.h"

typedef struct {
  int ncell;
  int minlev;
  int maxlev;
  int *levsta;
  int *levend;
  real *celeng;
  cell *c;
  real *fieldval;
  real *sourceval;
  real *proxysval;
  int ncell0;
  int Nf, Ns, *fieldlist, *sourcelist;
} anotherTree;


#endif /* ANOTHERTREE_H */
