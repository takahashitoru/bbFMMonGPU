#!/bin/bash
CPU_CLOCK_GHZ=2.666849 # compute-x-x @ Certainty

CC=icc
#PLAT=_INTEL120
#LDFLAGS="-L/home/ttaka/lib -lqsort_mt${PLAT} -L/opt/intel/composerxe-2011/lib/intel64 -lirc -limf -lsvml -L/usr/lib64 -llapack -lblas -lm"
#LDFLAGS="-L/share/apps/intel/lib/intel64 -lirc -limf -lsvml -L/usr/lib64 -llapack -lblas -lm"
PLAT=_INTEL120_omp
LDFLAGS="-L/share/apps/intel/lib/intel64 -lifcoremt -lirc -limf -lsvml -L/home/ttaka/lib -llapack${PLAT} -lblas${PLAT} -lm"

OPTS="-O3 -xSSE4.2"
OPTS+=" -vec-report2"
#OPTS+=" -Wall -D_DEBUG -DMYDEBUG"
OPTS+=" -openmp"
OPTS+=" -openmp-report2"

OMP="-lintlc -liomp5 -lpthread"

make -f Makefile CC="${CC}" LDFLAGS="${LDFLAGS}" OPTS="${OPTS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" $*
