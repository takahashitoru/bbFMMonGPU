#ifndef OUTPUT_H
#define OUTPUT_H

#include <stdio.h>
#include <stdlib.h>
#include "vec234.h"
#include "real.h"

#ifdef __cplusplus
extern "C" {
#endif
  void output(real3 *field, int Nf, int dof, real *phi);
#ifdef __cplusplus
}
#endif

#endif /* OUTPUT_H */
