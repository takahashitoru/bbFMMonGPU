#ifndef ATRANSX_H
#define ATRANSX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "real.h"

#ifdef __cplusplus
extern "C" {
#endif
  void atransx(const int m, const int n, const real *atrans, const real *x, real *y);
#ifdef __cplusplus
}
#endif

#endif /* ATRANSX_H */
