#!/bin/bash

CPU_CLOCK_GHZ=3.324837 # q.nuem.nagoya-u.ac.jp
CC=icc
PLAT=_INTEL120
#LDFLAGS="-L/home/ttaka/lib -lqsort_mt${PLAT} -L/opt/intel/composerxe-2011/lib/intel64 -lifcore -limf -lsvml -L/usr/lib64 -llapack -lblas -lm"
#LDFLAGS="-L/home/ttaka/lib -lqsort_mt${PLAT} -L/opt/intel/composerxe-2011/lib/intel64 -lirc -limf -lsvml -L/usr/lib64 -llapack -lblas -lm"

LDFLAGS="-L/opt/intel/composerxe-2011/lib/intel64 -lirc -limf -lsvml -L/usr/lib64 -llapack -lblas -lm"
##LDFLAGS="-L/opt/intel/composerxe-2011/lib/intel64 -lirc -limf -lsvml -L/usr/lib64 -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lm"


OPTS="-std=c99 -O3 -xSSE4.2"
#OPTS="-std=c99 -O3 -xSSE2" # for valgrind
OPTS+=" -vec-report2"
OPTS+=" -Wall"
#OPTS+=" -D_DEBUG -DMYDEBUG -g -check-uninit"

OPTS+=" -openmp"
OPTS+=" -openmp-report2"

OMP="-lintlc -liomp5 -lpthread"

#INCLUDES=" -I/home/ttaka/memcpy_DV"
#LIBS=" -L/home/ttaka/memcpy_DV -lmemcpy_DV${PLAT}"

make -f Makefile CC="${CC}" LDFLAGS="${LDFLAGS}" OPTS="${OPTS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" INCLUDES="${INCLUDES}" LIBS="${LIBS}" $*
