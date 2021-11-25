#ifndef OPTIONS_H
#define OPTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "debugmacros.h"

#ifdef __cplusplus
extern "C" {
#endif

  void bbfmm_options(int argc, char **argv, int *l, int *seed);

#ifdef __cplusplus
}
#endif

#endif /* OPTIONS_H */
