#ifndef XDOT_H
#define XDOT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "real.h"

#ifdef __cplusplus
extern "C" {
#endif
  real xdot(const int n, const real *x, const int incx, const real *y, const int incy);
#ifdef __cplusplus
}
#endif

#endif /* XDOT_H */
